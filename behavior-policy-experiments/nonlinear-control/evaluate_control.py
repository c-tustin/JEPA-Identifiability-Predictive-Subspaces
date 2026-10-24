"""Goal reaching in the nonlinear world, with frozen observation encoders.

Shared versus native action data separate predictor coverage from representation
training. Actual distance is restricted to simulation, scoring and the explicit
known-dynamics reference. See EXTENSION_PROTOCOL.md.
"""

import argparse
import json
import sys
from pathlib import Path

import run_trajectories as study
from run_coverage import reset_queries, fingerprint
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "planning"))
import evaluate_planning as base


def nominal_step(distance, action, settings):
    noise = torch.zeros_like(distance)
    for _ in range(settings.substeps):
        distance = study.advance(distance, action - 1, noise, True, settings.dt)
    return distance


class KnownDynamics(torch.nn.Module):
    """Explicit physical-state reference; never used by the learned planner."""

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def forward(self, distance, action):
        return nominal_step(distance, action, self.settings)


def make_goals(settings, seed, cases):
    generator = torch.Generator().manual_seed(3300000 + seed)
    proposals = 2 + 13 * torch.rand(max(4096, 20 * cases), generator=generator)
    # Use goals requiring active control, rather than being carried there by
    # passive drift. This task was fixed after the execution-only pilot.
    feasible = proposals[(0.3 * torch.sin(proposals)).abs() <= 0.1]
    if len(feasible) < cases:
        raise RuntimeError("Insufficient controllable start proposals")
    start = feasible[:cases]
    action = torch.where(torch.sin(start) > 0, 0, 2).long()
    goal = nominal_step(nominal_step(start, action, settings), action, settings)
    return start, goal, action


def simulation_noise(settings, seed, cases, stochastic):
    generator = torch.Generator().manual_seed(3400000 + seed)
    noise = torch.randn(12, settings.substeps, cases, generator=generator)
    return noise if stochastic else torch.zeros_like(noise)


@torch.no_grad()
def rollout(
    settings,
    seed,
    bank,
    start,
    goal,
    stochastic,
    encoder=None,
    predictor=None,
    mean=0,
    scale=1,
    reference=None,
):
    distance = start.clone()
    path = [distance.numpy().copy()]
    noise = simulation_noise(settings, seed, len(start), stochastic)
    if encoder is not None:
        goal_y = base.encode(
            encoder, study.behavior.sensor(goal, study.behavior.Config()), mean, scale
        )
    for decision in range(12):
        if reference == "stay":
            action = torch.ones(len(start), dtype=torch.long)
        elif reference == "known_dynamics":
            action = base.choose_actions(KnownDynamics(settings), distance, goal, bank)
        else:
            y = base.encode(
                encoder,
                study.behavior.sensor(distance, study.behavior.Config()),
                mean,
                scale,
            )
            action = base.choose_actions(predictor, y, goal_y, bank)
        for substep in range(settings.substeps):
            distance = study.advance(
                distance, action - 1, noise[decision, substep], True, settings.dt
            )
        path.append(distance.numpy().copy())
    path = np.stack(path)
    # Exclude the initial state: a hit requires at least one executed decision.
    errors = np.abs(path[1:] - goal.numpy()[None])
    closest = errors.min(0)
    return {
        "mean_closest_error": float(closest.mean()),
        "success_within_point_one": float(np.mean(closest <= 0.1)),
        "mean_terminal_error": float(errors[-1].mean()),
    }, {"start": start.numpy(), "goal": goal.numpy(), "path": path}


def discover(root, quick=False):
    records = []
    for source in ["trajectories", "coverage"]:
        folder = root / f"output-{source}"
        result = json.loads((folder / "metrics.json").read_text())
        if result["quick"]:
            raise ValueError("Representation sources must be full runs")
        expected = 36 if source == "trajectories" else 12
        if len(result["runs"]) != expected:
            raise ValueError(f"Incomplete {source} encoders: expected {expected}")
        for row in result["runs"]:
            if not row["nonlinear"]:
                continue
            policy = row.get("policy", "reset_uniform")
            if quick and (
                source != "trajectories" or row["seed"] != 23 or policy != "random"
            ):
                continue
            records.append(
                {
                    "seed": row["seed"],
                    "policy": policy,
                    "sampler": row["sampler"],
                    "source": source,
                    "checkpoint": str((folder / row["checkpoint"]).relative_to(root)),
                }
            )
    return records


def write_report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Nonlinear goal-reaching evaluation",
        "",
        "Frozen encoders; independently trained action predictors. Smaller physical errors are better. Hit rate is the fraction reaching within 0.1 during 12 decisions. Goals have a two-action deterministic reachability witness. [Protocol](../EXTENSION_PROTOCOL.md).",
        "",
        "Two predictor seeds are nested inside three representation seeds. Means below average those six fits; they are not six independent representation replicates.",
        "",
    ]
    if result["quick"]:
        lines += ["**Execution check only.**", ""]
    for mode in ["deterministic", "stochastic"]:
        lines += [
            f"## {mode.capitalize()} execution",
            "",
            "| Encoder data | SIGReg | Predictor data | Fits | Closest error | Hit rate | Final error |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for policy, sampler, data_kind in sorted(
            {(r["policy"], r["sampler"], r["predictor_data"]) for r in result["runs"]}
        ):
            rows = [
                r
                for r in result["runs"]
                if (r["policy"], r["sampler"], r["predictor_data"])
                == (policy, sampler, data_kind)
            ]
            values = [
                np.mean([r["metrics"][mode][key] for r in rows])
                for key in [
                    "mean_closest_error",
                    "success_within_point_one",
                    "mean_terminal_error",
                ]
            ]
            lines.append(
                f"| {policy} | {sampler} | {data_kind} | {len(rows)} | {values[0]:.4f} | {values[1]:.3f} | {values[2]:.4f} |"
            )
        for reference in ["known_dynamics", "stay"]:
            rows = [r[mode][reference] for r in result["references"]]
            if rows:
                values = [
                    np.mean([r[key] for r in rows])
                    for key in [
                        "mean_closest_error",
                        "success_within_point_one",
                        "mean_terminal_error",
                    ]
                ]
                lines.append(
                    f"| {reference} | — | — | {len(rows)} | {values[0]:.4f} | {values[1]:.3f} | {values[2]:.4f} |"
                )
        lines += [""]
    lines += [
        "The known-dynamics nominal MPC uses the same search bank/cost, but it is not a global or stochastic optimum. Closest error and final error answer different questions when drift prevents holding a goal. True distance never enters the learned controller. No true-distance stopping rule is used.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output-control"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).parent
    settings = study.Settings()
    protocol = {
        "settings": vars(settings),
        "updates": 20 if args.quick else 1000,
        "cases": 4 if args.quick else 32,
        "horizon": 4 if args.quick else 8,
        "candidates": 32 if args.quick else 128,
        "replicates": 1 if args.quick else 2,
        "quick": args.quick,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"protocol": protocol, "quick": args.quick, "runs": [], "references": []}
    if args.resume or args.evaluate_only:
        result = json.loads((args.output / "metrics.json").read_text())
        if result["protocol"] != protocol:
            raise ValueError("Saved protocol differs")
    elif (args.output / "metrics.json").exists():
        raise ValueError("Use --resume or a new output directory")
    data_cache = {}
    for record in discover(root, args.quick):
        seed = record["seed"]
        if seed not in data_cache:
            count = 1024 if args.quick else settings.episodes * settings.horizon
            common = reset_queries(settings, 10000 + seed, True, count)
            test = reset_queries(settings, 20000 + seed, True, count)
            start, goal, witness = make_goals(settings, seed, protocol["cases"])
            replay = nominal_step(
                nominal_step(start, witness, settings), witness, settings
            )
            torch.testing.assert_close(replay, goal, atol=0, rtol=0)
            bank = base.candidate_bank(
                seed, protocol["candidates"], protocol["horizon"]
            )
            data_cache[seed] = {
                "common": common,
                "test": test,
                "start": start,
                "goal": goal,
                "bank": bank,
            }
            # Deterministic replay proves reachability independently of MPC.
            references = {
                "seed": seed,
                "witness_max_error": float((replay - goal).abs().max()),
            }
            for mode in ["deterministic", "stochastic"]:
                references[mode] = {}
                for name in ["known_dynamics", "stay"]:
                    metrics, _ = rollout(
                        settings,
                        seed,
                        bank,
                        start,
                        goal,
                        mode == "stochastic",
                        reference=name,
                    )
                    references[mode][name] = metrics
            found_ref = next(
                (r for r in result["references"] if r["seed"] == seed), None
            )
            if found_ref is None:
                result["references"].append(references)
            elif found_ref != references:
                raise RuntimeError("Reference rollout changed")
        cached = data_cache[seed]
        encoder = study.behavior.Encoder()
        encoder.load_state_dict(
            torch.load(root / record["checkpoint"], map_location="cpu")["encoder"]
        )
        encoder.eval()
        for kind in (
            ["common"] if record["source"] == "coverage" else ["common", "native"]
        ):
            data_key = record["policy"]
            if kind == "native" and data_key not in cached:
                cached[data_key] = study.collect(settings, seed, data_key, True)
            data = cached["common"] if kind == "common" else cached[data_key]
            source_hash = fingerprint(data)
            for replicate in range(protocol["replicates"]):
                filename = f"predictor_{record['policy']}_{record['sampler']}_{kind}_seed{seed}_fit{replicate}.pt"
                found = next(
                    (
                        r
                        for r in result["runs"]
                        if r["predictor_checkpoint"] == filename
                    ),
                    None,
                )
                if found is not None and not args.evaluate_only:
                    continue
                print(filename, flush=True)
                if args.evaluate_only:
                    if found is None or found["predictor_data_sha256"] != source_hash:
                        raise RuntimeError("Missing predictor or changed training data")
                    saved = torch.load(args.output / filename, map_location="cpu")
                    predictor = base.Predictor()
                    predictor.load_state_dict(saved["predictor"])
                    predictor.eval()
                    mean, scale = saved["mean"], saved["scale"]
                else:
                    predictor, mean, scale = base.train_predictor(
                        encoder,
                        data,
                        cached["common"],
                        100 * seed + replicate,
                        protocol["updates"],
                    )
                    torch.save(
                        {
                            "predictor": predictor.state_dict(),
                            "mean": mean,
                            "scale": scale,
                            "encoder_checkpoint": record["checkpoint"],
                            "protocol": protocol,
                        },
                        args.output / filename,
                    )
                with torch.no_grad():
                    y = base.encode(encoder, cached["test"]["x"], mean, scale)
                    yp = base.encode(encoder, cached["test"]["xp"], mean, scale)
                    metrics = {
                        "shared_heldout_latent_mse": float(
                            (predictor(y, cached["test"]["action"]) - yp)
                            .square()
                            .mean()
                        )
                    }
                arrays = {}
                for mode in ["deterministic", "stochastic"]:
                    scores, path = rollout(
                        settings,
                        seed,
                        cached["bank"],
                        cached["start"],
                        cached["goal"],
                        mode == "stochastic",
                        encoder,
                        predictor,
                        mean,
                        scale,
                    )
                    metrics[mode] = scores
                    arrays.update({f"{mode}_{k}": v for k, v in path.items()})
                if args.evaluate_only:
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for metric, number in value.items():
                                if not np.isclose(
                                    number,
                                    found["metrics"][key][metric],
                                    atol=1e-8,
                                    rtol=0,
                                ):
                                    raise RuntimeError(
                                        f"Changed rollout: {filename}: {key}: {metric}"
                                    )
                        elif not np.isclose(
                            value, found["metrics"][key], atol=1e-8, rtol=0
                        ):
                            raise RuntimeError(f"Changed predictor metric: {filename}")
                else:
                    np.savez_compressed(
                        args.output / filename.replace(".pt", ".npz"), **arrays
                    )
                    result["runs"].append(
                        {
                            **record,
                            "predictor_data": kind,
                            "predictor_replicate": replicate,
                            "predictor_checkpoint": filename,
                            "predictor_data_sha256": source_hash,
                            "metrics": metrics,
                        }
                    )
                    write_report(args.output, result)
    expected = 4 if args.quick else 84
    if len(result["runs"]) != expected:
        raise RuntimeError(f"Incomplete predictor comparison: expected {expected}")
    batches = 2 * len(result["runs"])
    print(
        f"{'Reproduced' if args.evaluate_only else 'Completed'} {len(result['runs'])} predictors / {batches} goal-reaching batches ({batches * protocol['cases']} episodes).",
        flush=True,
    )


if __name__ == "__main__":
    main()

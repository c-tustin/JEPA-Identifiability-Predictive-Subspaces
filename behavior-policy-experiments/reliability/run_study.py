"""Fixed reliability interventions and alternative policies, all in one distance.

Default validation seeds 8–12 were unused in the earlier curriculum study.
Seeds 3 and 7 are explicitly diagnostic, never independent validation.
"""

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "curriculum-controls"))
import run_controls as controls
from policies import behavior, collect, load, size_bin, train
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VALIDATION_SEEDS = [8, 9, 10, 11, 12]
REPAIR_CONDITIONS = [
    "ou",
    "random",
    "icm",
    "ou_floor",
    "earlier_ou",
    "icm_epsilon",
    "icm_coverage",
    "balanced_replay",
    "monotone",
    "ou_long",
    "random_long",
    "icm_long",
]
POLICY_CONDITIONS = ["rnd", "disagreement", "count", "action_balanced"]


class MonotoneEncoder(nn.Module):
    """Increasing spline of raw apparent height, not log-distance.

    Uses known observation ordering and changes the encoder class. This is a
    diagnostic architectural prior, not unconstrained learned identifiability.
    End intervals extrapolate with their positive learned slope.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("knots", torch.linspace(0.0, 2.0, 65))
        self.increments = nn.Parameter(torch.full((64,), -3.0))
        self.offset = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        height = x[:, 1].contiguous()
        ix = torch.bucketize(height, self.knots).sub(1).clamp(0, len(self.knots) - 2)
        values = self.offset + torch.cat(
            [torch.zeros(1, device=x.device), F.softplus(self.increments).cumsum(0)]
        )
        fraction = (height - self.knots[ix]) / (self.knots[ix + 1] - self.knots[ix])
        return values[ix] + fraction * (values[ix + 1] - values[ix])


def source_name(condition):
    if condition.startswith("ou") and condition != "ou_floor":
        return "ou"
    if condition.startswith("random"):
        return "random"
    if condition in POLICY_CONDITIONS + ["icm_epsilon", "icm_coverage"]:
        return condition
    return "icm"


def make_plan(c, data_seed, init_seed, condition):
    # Replay the historical pair index stream independently of encoder init.
    _, legacy = controls.make_plan(c, data_seed)
    indices = legacy["indices"]
    torch.manual_seed(40000 + init_seed)
    model = MonotoneEncoder() if condition == "monotone" else behavior.Encoder()
    n = c.train_episodes * c.horizon
    fractions = [
        controls.schedules.epoch_fraction(i, c.jepa_steps, n, c.batch)
        for i in range(c.jepa_steps)
    ]
    start, finish = (0.1, 0.5) if condition == "earlier_ou" else (0.25, 0.75)
    alpha = np.clip((np.array(fractions) - start) / (finish - start), 0, 1)
    if source_name(condition) == "ou":
        alpha[:] = 1
    if condition == "ou_floor":
        alpha = 0.2 + 0.8 * alpha
    # Preserve the exact historical mixture draws at transition steps. Draws at
    # the old zero/one endpoints use a separate stream for modified schedules.
    masks_rng = torch.Generator().manual_seed(700000 + data_seed)
    endpoint_rng = torch.Generator().manual_seed(800000 + data_seed)
    uniforms = torch.stack(
        [
            torch.rand(c.batch, generator=masks_rng if 0 < a < 1 else endpoint_rng)
            for a in legacy["alpha"]
        ]
    )
    masks = uniforms < torch.tensor(alpha, dtype=torch.float32)[:, None]
    return model, {
        "indices": indices,
        "masks": masks,
        "alpha": alpha,
        "sha256": hashlib.sha256(
            indices.numpy().tobytes() + masks.numpy().tobytes()
        ).hexdigest(),
    }


def fit(data, condition, c, data_seed, init_seed, reference):
    model, plan = make_plan(c, data_seed, init_seed, condition)
    ou = controls.ou_data(c, data_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer = behavior.SIGReg1D()
    history = []
    # Only observation pairs are retained by fitting; latent labels/actions unused.
    views = {key: data[key] for key in ["x", "xp"]}
    if condition == "balanced_replay":
        bins = size_bin(views["x"])
        counts = torch.bincount(bins, minlength=32).clamp_min(1)
        generator = torch.Generator().manual_seed(900000 + data_seed)
        selected = torch.multinomial(
            1 / counts[bins].float(), len(bins), replacement=True, generator=generator
        )
        views = {key: value[selected] for key, value in views.items()}
    for step in range(c.jepa_steps):
        losses = controls.update(
            model,
            optimizer,
            regularizer,
            *controls.batch_at(views, ou, plan, step),
            c,
            controls.learning_rate(step, c.jepa_steps),
        )
        if step % 512 == 0 or step == c.jepa_steps - 1:
            metrics, _ = controls.evaluate(model, reference, c, data_seed)
            history.append(
                {
                    "updates": step + 1,
                    "ou_fraction": float(plan["alpha"][step]),
                    **losses,
                    **metrics,
                }
            )
    return model.eval(), history, plan["sha256"]


def prepare_data(c, seed, kind, out, quick=False):
    folder = out / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{kind}_seed{seed}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu")
        if saved["config"] != asdict(c):
            raise RuntimeError("Cached source configuration differs")
        return saved["data"], saved["policy"]
    if kind in {"ou", "random"}:
        data = behavior.collect(c, kind, c.train_episodes, 10000 + seed)
        policy_saved = {"kind": kind}
    elif kind == "action_balanced":
        data = collect(c, kind, c.train_episodes, 10000 + seed)
        policy_saved = {"kind": kind}
    else:
        historical = (
            Path(controls.__file__).parent / "output" / f"behavior_seed{seed}.pt"
        )
        if kind == "icm" and historical.exists() and not quick:
            policy_saved = torch.load(historical, map_location="cpu")
            if policy_saved["config"] != asdict(c):
                raise RuntimeError("Historical policy configuration differs")
            policy_saved = {**policy_saved, "kind": "icm", "epsilon": 0.0}
            policy = load(policy_saved)
        else:
            print(f"seed={seed}: policy {kind}", flush=True)
            policy, policy_saved = train(c, seed, kind)
        data = collect(c, kind, c.train_episodes, 10000 + seed, policy)
    torch.save({"config": asdict(c), "policy": policy_saved, "data": data}, path)
    return data, policy_saved


def metric_record(
    model, history, plan_hash, condition, seed, init_seed, c, reference, out
):
    metrics, example = controls.evaluate(model, reference, c, seed)
    with torch.no_grad():
        z = torch.linspace(
            np.log(c.distance_min / 4) / c.sigma,
            np.log(c.distance_max / 4) / c.sigma,
            1024,
        )
        y = model(behavior.observe(z, c.sigma))
        sign = 1 if torch.cov(torch.stack([z, y]))[0, 1] >= 0 else -1
        normalized = sign * (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-8)
        metrics["reversed_grid_fraction"] = float(
            (torch.diff(normalized) < -1e-6).float().mean()
        )
    filename = f"encoder_{condition}_data{seed}_init{init_seed}.pt"
    torch.save(
        {
            "encoder": model.state_dict(),
            "condition": condition,
            "config": asdict(c),
            "data_seed": seed,
            "init_seed": init_seed,
            "plan_hash": plan_hash,
        },
        out / filename,
    )
    np.savez_compressed(
        out / filename.replace("encoder_", "evaluation_").replace(".pt", ".npz"),
        **example,
    )
    return {
        "condition": condition,
        "seed": seed,
        "init_seed": init_seed,
        "validation": seed in VALIDATION_SEEDS,
        "metrics": metrics,
        "history": history,
        "checkpoint": filename,
        "plan_hash": plan_hash,
        "config": asdict(c),
    }


def write_report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Reliability study results",
        "",
        "One distance coordinate; settings fixed before inspecting these results. "
        "Seeds 3 and 7 diagnose previous failures; seeds 8–12 are new validation runs. "
        "The new runner separates initialization and batch randomness. Its ordinary ICM and OU runs replay the earlier baseline streams when data and initialization seeds match.",
        "",
    ]
    if result["quick"]:
        lines += ["**Execution check only.**", ""]
    groups = [
        ("Validation", [r for r in result["runs"] if r["validation"]]),
        ("Diagnostic", [r for r in result["runs"] if not r["validation"]]),
    ]
    for label, records in groups:
        if not records:
            continue
        lines += [
            f"## {label}",
            "",
            "| Condition | Runs | Shared-range $R^2$ | Broader OU $R^2$ | Reversed grid steps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for condition in sorted({r["condition"] for r in records}):
            rows = [r for r in records if r["condition"] == condition]
            values = [
                [r["metrics"][key] for r in rows]
                for key in [
                    "common_grid_r2",
                    "reference_ou_r2",
                    "reversed_grid_fraction",
                ]
            ]
            lines.append(
                f"| {condition} | {len(rows)} | "
                + " | ".join(controls.summary(v) for v in values)
                + " |"
            )
    lines += [
        "",
        "## Individual runs",
        "",
        "| Data seed | Init seed | Condition | Shared $R^2$ | Broader $R^2$ |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for r in result["runs"]:
        lines.append(
            f"| {r['seed']} | {r['init_seed']} | {r['condition']} | {r['metrics']['common_grid_r2']:.6f} | {r['metrics']['reference_ou_r2']:.6f} |"
        )
    lines += [
        "",
        "Means ± sample standard deviations. No best-seed selection or distance-label tuning. "
        "Policy choices use the apparent-size geometry where stated; the monotone encoder adds an ordering prior. "
        "Long runs have twice the representation budget. This comparison is not a universal ranking of exploration algorithms.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def evaluate_saved(out):
    result = json.loads((out / "metrics.json").read_text())
    for r in result["runs"]:
        saved = torch.load(out / r["checkpoint"], map_location="cpu")
        c, seed = behavior.Config(**saved["config"]), r["seed"]
        model = (
            MonotoneEncoder() if r["condition"] == "monotone" else behavior.Encoder()
        )
        model.load_state_dict(saved["encoder"])
        reference = (
            behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
            behavior.collect(c, "ou", c.eval_episodes, 30000 + seed),
        )
        metrics, _ = controls.evaluate(model, reference, c, seed)
        for key, value in metrics.items():
            if not np.isclose(value, r["metrics"][key], atol=1e-7, rtol=0):
                raise RuntimeError(f"Evaluation changed: {r['checkpoint']} {key}")
    print(
        f"Reproduced {len(result['runs'])} saved reliability evaluations.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=["repairs", "policies", "crossed"], default="repairs"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    behavior.verify_upstream()
    out = args.output or Path(__file__).parent / f"output-{args.suite}"
    out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        evaluate_saved(out)
        return
    c = behavior.Config()
    if args.quick:
        c.jepa_steps, c.policy_updates, c.train_episodes, c.eval_episodes = 40, 2, 8, 4
        c.nenv, c.rollout, c.horizon, c.burnin, c.batch = 4, 4, 8, 4, 16
    seeds = (
        ([3, 7] + VALIDATION_SEEDS)
        if args.suite == "repairs"
        else VALIDATION_SEEDS if args.suite == "policies" else list(range(8))
    )
    if args.quick:
        seeds = [3]
    conditions = (
        REPAIR_CONDITIONS
        if args.suite == "repairs"
        else POLICY_CONDITIONS if args.suite == "policies" else ["icm", "ou"]
    )
    result = {
        "suite": args.suite,
        "quick": args.quick,
        "base_config": asdict(c),
        "seeds": seeds,
        "conditions": conditions,
        "runs": [],
    }
    if args.resume and (out / "metrics.json").exists():
        saved = json.loads((out / "metrics.json").read_text())
        for key in ["suite", "quick", "base_config", "seeds", "conditions"]:
            if saved[key] != result[key]:
                raise RuntimeError("Resume configuration differs")
        result = saved
    for seed in seeds:
        reference = (
            behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
            behavior.collect(c, "ou", c.eval_episodes, 30000 + seed),
        )
        for condition in conditions:
            init_seeds = [100, 101, 102] if args.suite == "crossed" else [seed]
            for init_seed in init_seeds:
                if any(
                    r["seed"] == seed
                    and r["condition"] == condition
                    and r["init_seed"] == init_seed
                    for r in result["runs"]
                ):
                    continue
                run_config = copy.deepcopy(c)
                if condition.endswith("_long"):
                    run_config.jepa_steps *= 2
                data, _ = prepare_data(c, seed, source_name(condition), out, args.quick)
                print(f"seed={seed} init={init_seed}: {condition}", flush=True)
                model, history, plan_hash = fit(
                    data, condition, run_config, seed, init_seed, reference
                )
                record = metric_record(
                    model,
                    history,
                    plan_hash,
                    condition,
                    seed,
                    init_seed,
                    run_config,
                    reference,
                    out,
                )
                result["runs"].append(record)
                write_report(out, result)
                print(
                    json.dumps(
                        {"seed": seed, "condition": condition, **record["metrics"]}
                    ),
                    flush=True,
                )
    print(f"Completed {args.suite}: {len(result['runs'])} models.", flush=True)


if __name__ == "__main__":
    main()

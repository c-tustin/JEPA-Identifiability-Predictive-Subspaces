"""Explain the ICM-to-OU gain without changing the one-distance environment.

Fixed, paired controls: early source x Adam reset x learning-rate restart,
plus OU-only runs with exactly the handover's OU pair occurrences.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "schedules"))
import run_schedules as schedules

behavior = schedules.behavior
import matplotlib.pyplot as plt
import numpy as np
import torch

SOURCES = ("ou", "random", "icm")
TAILS = ("carry", "reset_adam", "restart_lr", "reset_both")
CONDITIONS = [f"{source}_{tail}" for source in SOURCES for tail in TAILS] + [
    "ou_exposure_compressed_clock", "ou_exposure_original_clock"]
LABELS = {"ou": "OU → OU", "random": "Random → OU", "icm": "ICM → OU"}


def learning_rate(step, steps):
    """Original rule: hold 0.001 for half the budget, then cosine decay."""
    phase = max(0., (step - steps / 2) / max(1., steps / 2 - 1))
    return float(.001 * .5 * (1 + np.cos(np.pi * phase)))


def make_plan(c, seed):
    """Replay the legacy initialization/minibatches; mask RNG is independent.

    Evaluation and optimizer branching cannot alter this recorded pair plan.
    Only observations reach the optimizer; indices/masks are sampling metadata.
    """
    torch.manual_seed(40000 + seed)
    initial_model = behavior.Encoder()
    schedule = schedules.Schedule()
    n = c.train_episodes * c.horizon
    indices = torch.stack([torch.randint(n, (c.batch,)) for _ in range(c.jepa_steps)])
    fractions = [schedules.epoch_fraction(i, c.jepa_steps, n, c.batch) for i in range(c.jepa_steps)]
    alpha = [schedules.progress(f, schedule) for f in fractions]
    generator = torch.Generator().manual_seed(700000 + seed)
    masks = []
    for a in alpha:
        masks.append(torch.zeros(c.batch, dtype=torch.bool) if a == 0 else
                     torch.ones(c.batch, dtype=torch.bool) if a == 1 else
                     torch.rand(c.batch, generator=generator) < a)
    masks = torch.stack(masks)
    tail_start = next(i for i, a in enumerate(alpha) if a == 1)
    digest = hashlib.sha256(indices.numpy().tobytes() + masks.numpy().tobytes()).hexdigest()
    return initial_model, {"indices": indices, "masks": masks, "alpha": alpha,
                           "tail_start": tail_start, "sha256": digest}


def ou_data(c, seed):
    # Identical latent RNG to the old PairSampler, but return observations only.
    generator = torch.Generator().manual_seed(10000 + seed)
    n = c.train_episodes * c.horizon
    z, noise = torch.randn(n, generator=generator), torch.randn(n, generator=generator)
    zp = c.rho * z + (1 - c.rho**2)**.5 * noise
    return {"x": behavior.observe(z, c.sigma), "xp": behavior.observe(zp, c.sigma)}


def batch_at(data, ou, plan, step):
    ix, mask = plan["indices"][step], plan["masks"][step, :, None]
    return (torch.where(mask, ou["x"][ix], data["x"][ix]),
            torch.where(mask, ou["xp"][ix], data["xp"][ix]))


def update(model, optimizer, regularizer, x, xp, c, lr):
    optimizer.param_groups[0]["lr"] = lr
    embedding = model(torch.cat([x, xp]))
    alignment = (embedding[:len(x)] - embedding[len(x):]).square().mean()
    sigreg = regularizer(embedding)
    loss = (1 - c.sigreg_weight) * alignment + c.sigreg_weight * sigreg
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"alignment": float(alignment.detach()), "sigreg": float(sigreg.detach()), "lr": lr}


@torch.no_grad()
def evaluate(model, reference, c, seed):
    # Preserve RNG even though every training pair has already been fixed.
    with torch.random.fork_rng(devices=[]):
        metrics, example = behavior.evaluate_encoder(model.eval(), *reference, c, seed)
    return {"common_grid_r2": metrics["common_log_distance_grid_r2"],
            "reference_ou_r2": metrics["on_policy_log_distance_r2"],
            "reference_ou_raw_variance": metrics["raw_embedding_variance"],
            "reference_ou_gaussian_cf_error": metrics["heldout_gaussian_cf_discrepancy"],
            "reference_ou_alignment": metrics["raw_alignment"]}, example


def run_segment(model, optimizer, data, ou, plan, c, seed, reference, start, stop, restart_lr=False):
    history, regularizer = [], behavior.SIGReg1D()
    for step in range(start, stop):
        lr = learning_rate(step - start, stop - start) if restart_lr else learning_rate(step, c.jepa_steps)
        losses = update(model, optimizer, regularizer, *batch_at(data, ou, plan, step), c, lr)
        if step % 256 == 0 or step == stop - 1:
            metrics, _ = evaluate(model, reference, c, seed)
            history.append({"updates": step + 1, "ou_fraction": plan["alpha"][step], **losses, **metrics})
    return history


def exposure_stream(plan):
    """All and only handover OU occurrences, preserving their order."""
    ix = plan["indices"][plan["masks"]]
    steps = torch.arange(len(plan["indices"]))[:, None].expand_as(plan["indices"])[plan["masks"]]
    return ix, steps


def train_exposure(initial, ou, plan, c, seed, reference, original_clock):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    regularizer, history = behavior.SIGReg1D(), []
    ix, original_steps = exposure_stream(plan)
    updates = (len(ix) + c.batch - 1) // c.batch
    for step, offset in enumerate(range(0, len(ix), c.batch)):
        selected = ix[offset:offset + c.batch]
        # The last pair's collection step anchors each packed batch's clock.
        original_step = int(original_steps[min(offset + c.batch, len(ix)) - 1])
        lr = learning_rate(original_step, c.jepa_steps) if original_clock else learning_rate(step, updates)
        losses = update(model, optimizer, regularizer, ou["x"][selected], ou["xp"][selected], c, lr)
        if step % 256 == 0 or step == updates - 1:
            metrics, _ = evaluate(model, reference, c, seed)
            history.append({"updates": step + 1, "original_step": original_step, **losses, **metrics})
    return model, history, {"ou_pairs": len(ix), "updates": updates, "last_batch_size": len(selected)}


def save_condition(out, name, seed, model, history, metadata, reference, c):
    metrics, example = evaluate(model, reference, c, seed)
    torch.save({"encoder": model.state_dict(), "config": asdict(c), "condition": name,
                "metadata": metadata}, out / f"encoder_{name}_seed{seed}.pt")
    np.savez_compressed(out / f"evaluation_{name}_seed{seed}.npz", **example)
    print(json.dumps({"seed": seed, "condition": name, **metrics}), flush=True)
    return {"metrics": metrics, "history": history, **metadata}


def load_policy(c, seed, out, quick):
    original = Path(behavior.__file__).parent / "output"
    source_path = original / f"behavior_seed{seed}.pt"
    if not quick and source_path.exists():
        saved = torch.load(source_path, map_location="cpu")
        if saved["config"] != asdict(c):
            raise RuntimeError("Historical ICM configuration differs")
        policy = behavior.Policy()
        policy.load_state_dict(saved["policy"])
        history = next(r["policy_history"] for r in json.loads((original / "metrics.json").read_text())["runs"] if r["seed"] == seed)
        reused = True
    else:
        policy, icm, history, change = behavior.train_policy(c, seed)
        saved = {"policy": policy.state_dict(), "icm": icm.state_dict(), "config": asdict(c),
                 "weight_change_l2": change, "upstream_commit": behavior.verify_upstream()}
        reused = False
    torch.save(saved, out / f"behavior_seed{seed}.pt")
    return policy.eval(), {"reused_historical_policy": reused, "history": history}


def train_seed(c, seed, out, quick):
    began = time.monotonic()
    policy, policy_record = load_policy(c, seed, out, quick)
    reference = (behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
                 behavior.collect(c, "ou", c.eval_episodes, 30000 + seed))
    initial, plan = make_plan(c, seed)
    ou = ou_data(c, seed)
    record = {"seed": seed, "new_seed": seed not in [0, 1, 2], "policy": policy_record,
              "plan": {"sha256": plan["sha256"], "tail_start": plan["tail_start"],
                       "handover_ou_pairs": int(plan["masks"].sum()),
                       "handover_non_ou_pairs": int((~plan["masks"]).sum()),
                       "total_pair_slots": plan["masks"].numel()}, "conditions": {}}
    for source in SOURCES:
        print(f"seed={seed}: prefix {source}", flush=True)
        data = ou if source == "ou" else behavior.collect(c, source, c.train_episodes, 10000 + seed, policy)
        # Deliberately discard latent labels and actions before fitting.
        data = {key: data[key] for key in ["x", "xp"]}
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        prefix_history = run_segment(model, optimizer, data, ou, plan, c, seed, reference, 0, plan["tail_start"])
        prefix_weights = copy.deepcopy(model.state_dict())
        prefix_optimizer = copy.deepcopy(optimizer.state_dict())
        torch.save({"encoder": prefix_weights, "optimizer": prefix_optimizer, "config": asdict(c),
                    "next_step": plan["tail_start"], "plan_sha256": plan["sha256"]},
                   out / f"prefix_{source}_seed{seed}.pt")
        for tail in TAILS:
            name = f"{source}_{tail}"
            model = copy.deepcopy(initial)
            model.load_state_dict(prefix_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            reset_adam = tail in {"reset_adam", "reset_both"}
            restart_lr = tail in {"restart_lr", "reset_both"}
            if not reset_adam:
                optimizer.load_state_dict(copy.deepcopy(prefix_optimizer))
            tail_history = run_segment(model, optimizer, data, ou, plan, c, seed, reference,
                                       plan["tail_start"], c.jepa_steps, restart_lr)
            metadata = {"source": source, "reset_adam": reset_adam, "restart_lr": restart_lr,
                        "representation_updates": c.jepa_steps,
                        "ou_pair_occurrences": plan["masks"].numel() if source == "ou" else int(plan["masks"].sum()),
                        "historical_match": verify_historical(model, source, tail, c, seed, quick)}
            record["conditions"][name] = save_condition(out, name, seed, model, prefix_history + tail_history,
                                                        metadata, reference, c)
        del data
    for original_clock in [False, True]:
        name = "ou_exposure_original_clock" if original_clock else "ou_exposure_compressed_clock"
        model, history, exposure = train_exposure(initial, ou, plan, c, seed, reference, original_clock)
        metadata = {"source": "ou", "representation_updates": exposure["updates"],
                    "ou_pair_occurrences": exposure["ou_pairs"], "last_batch_size": exposure["last_batch_size"]}
        record["conditions"][name] = save_condition(out, name, seed, model, history, metadata, reference, c)
    record["elapsed_seconds"] = time.monotonic() - began
    return record


def verify_historical(model, source, tail, c, seed, quick):
    if quick or tail != "carry" or source not in {"ou", "icm"} or seed not in [0, 1, 2]:
        return None
    folder = Path(schedules.__file__).parent / "output"
    condition = "ou" if source == "ou" else "icm_to_ou"
    saved = torch.load(folder / f"encoder_{condition}_seed{seed}.pt", map_location="cpu")
    if saved["config"] != asdict(c):
        raise RuntimeError("Historical encoder configuration differs")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["encoder"][key], rtol=0, atol=0)
    return "Bit-for-bit identical encoder weights"


def summary(values):
    return f"{np.mean(values):.6f} ± {np.std(values, ddof=1) if len(values) > 1 else 0:.6f}"


def comparisons(runs):
    pairs = [("icm_carry", "ou_carry"), ("random_carry", "ou_carry"), ("icm_carry", "random_carry"),
             ("icm_carry", "ou_exposure_compressed_clock"), ("icm_carry", "ou_exposure_original_clock")]
    pairs += [(f"{s}_{t}", f"{s}_carry") for s in SOURCES for t in TAILS[1:]]
    pairs += [(f"icm_{t}", f"random_{t}") for t in TAILS[1:]]
    result = []
    for group, subset in [("all", runs), ("new", [r for r in runs if r["new_seed"]]),
                           ("historical", [r for r in runs if not r["new_seed"]])]:
        if not subset:
            continue
        for a, b in pairs:
            for metric in ["common_grid_r2", "reference_ou_r2"]:
                difference = np.array([r["conditions"][a]["metrics"][metric] - r["conditions"][b]["metrics"][metric] for r in subset])
                rng = np.random.default_rng(20261022)
                boot = difference[rng.integers(len(difference), size=(20000, len(difference)))].mean(1)
                result.append({"group": group, "a": a, "b": b, "metric": metric, "n": len(subset),
                               "mean_difference": float(difference.mean()), "wins": int((difference > 0).sum()),
                               "minimum_difference": float(difference.min()), "maximum_difference": float(difference.max()),
                               "paired_bootstrap_95_interval": np.quantile(boot, [.025, .975]).tolist()})
    return result


def report(out, result):
    runs = result["runs"]
    result["comparisons"] = comparisons(runs)
    lines = ["# What explains the curriculum gain?", "",
             "Same triangle, same single distance coordinate, same encoder and SIGReg. "
             "All choices were fixed before this study's results were inspected.", "",
             "**Execution check only; these are not trained results.**" if result["quick"] else
             f"Completed seeds: {', '.join(str(r['seed']) for r in runs)}. Seeds 0–2 were previously studied; seeds 3–7 are new to this study.", "",
             "Scores are mean ± sample standard deviation. R² closer to 1 means a straighter log-distance readout. "
             "The fixed Gaussian OU test includes distances beyond the shared range [1,16].", "",
             "| Condition | Shared-range R² | Fixed Gaussian OU R² | OU-test output variance |", "| --- | ---: | ---: | ---: |"]
    for name in CONDITIONS:
        lines.append(f"| {name} | " + " | ".join(summary([r["conditions"][name]["metrics"][k] for r in runs]) for k in
                     ["common_grid_r2", "reference_ou_r2", "reference_ou_raw_variance"]) + " |")
    lines += ["", "## Paired comparisons", "",
              "Differences are first condition minus second. Positive means the first scored higher. "
              "The interval resamples whole paired seeds (20,000 bootstrap draws, fixed random seed). "
              "These small-sample descriptive intervals are not corrected for multiple comparisons and do not establish a general policy ranking.", "",
              "| Seed group | Comparison | Test | Mean R² difference | 95% paired-bootstrap interval | Wins |",
              "| --- | --- | --- | ---: | --- | ---: |"]
    for row in result["comparisons"]:
        low, high = row["paired_bootstrap_95_interval"]
        lines.append(f"| {row['group']} | {row['a']} − {row['b']} | {row['metric']} | {row['mean_difference']:+.6f} | [{low:+.6f}, {high:+.6f}] | {row['wins']}/{row['n']} |")
    lines += ["", "## Exposure and timing checks", "",
              "Every source uses identical initial encoder weights, minibatch indices and OU pairs. "
              "ICM and random handovers use the same pair-selection masks. Every tail starts from its source's exact same prefix checkpoint; "
              "the final pure-OU minibatches are identical across all 12 source/reset conditions.", "",
              "OU→OU uses the same OU observations on both sides of the nominal handover, so its carry condition is exactly fixed OU. "
              "The historical carry controls are checked against their saved encoder weights bit for bit.", "",
              "| Seed | Historical encoder checks | Final pure-OU starts at update index | Handover OU pairs | Full-OU pairs | Exposure-only updates | Final exposure batch |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        checks = [r["conditions"][f"{s}_carry"]["historical_match"] for s in ["ou", "icm"]]
        e = r["conditions"]["ou_exposure_compressed_clock"]
        lines.append(f"| {r['seed']} | {'exact' if all(checks) else 'new run / smoke'} | {r['plan']['tail_start']} | {r['plan']['handover_ou_pairs']} | {r['plan']['total_pair_slots']} | {e['representation_updates']} | {e['last_batch_size']} |")
    lines += ["", "The two exposure-only controls discard all non-OU slots and pack the exact remaining OU occurrences in order into full batches, "
              "plus one possible final partial batch. Their clocks either compress the original learning-rate rule to that shorter budget, "
              "or use each packed batch's last pair's original step. Both match OU observations exactly but have fewer optimizer updates, "
              "different batch grouping, and different regularizer composition from the handover. They do not isolate every possible mechanism.", "",
              "At the final pure-OU boundary, Adam reset erases moment estimates and its step counter; LR restart independently starts "
              "the original hold-then-cosine rule over the remaining tail. Restarted runs spend longer at a larger learning rate; "
              "this is an intentional intervention, not equal integrated learning-rate exposure. The reset-both comparison tests the learned weights "
              "without inherited optimizer memory, given identical subsequent data and learning rates.", "",
              "Representation update counts exclude the separate cost of learning ICM. Random and OU have no policy-training cost. "
              "Evaluation uses labels only for held-out readouts, never for updates, early stopping or selecting checkpoints.", "",
              "## Individual scores", "", "| Seed | Condition | Shared-range R² | Fixed OU R² |", "| --- | --- | ---: | ---: |"]
    for r in runs:
        for name in CONDITIONS:
            m = r["conditions"][name]["metrics"]
            lines.append(f"| {r['seed']} | {name} | {m['common_grid_r2']:.6f} | {m['reference_ou_r2']:.6f} |")
    lines += ["", "![Controls](comparison.png)", "", "See [the protocol](../README.md) and [interpretation](../INTERPRETATION.md).", ""]
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RESULTS.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    names = ["ou_carry", "random_carry", "icm_carry", "ou_exposure_compressed_clock", "ou_exposure_original_clock"]
    for ax, metric, title in zip(axes, ["common_grid_r2", "reference_ou_r2"], ["Same distance range", "Broader fixed Gaussian OU test"]):
        for i, name in enumerate(names):
            values = [r["conditions"][name]["metrics"][metric] for r in runs]
            ax.scatter(values, np.full(len(values), i), s=22, alpha=.7)
            ax.scatter(np.mean(values), i, marker="|", s=240, color="black")
        ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Recovery R²", title=title)
        ax.invert_yaxis()
    fig.suptitle("Curriculum controls: dots are seeds, black marks are means")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def evaluate_saved(out):
    result = json.loads((out / "metrics.json").read_text())
    c = behavior.Config(**result["config"])
    for r in result["runs"]:
        seed = r["seed"]
        _, plan = make_plan(c, seed)
        if plan["sha256"] != r["plan"]["sha256"]:
            raise RuntimeError("Sampling plan changed")
        reference = (behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
                     behavior.collect(c, "ou", c.eval_episodes, 30000 + seed))
        for name in CONDITIONS:
            saved = torch.load(out / f"encoder_{name}_seed{seed}.pt", map_location="cpu")
            if saved["config"] != asdict(c):
                raise RuntimeError("Saved configuration mismatch")
            model = behavior.Encoder()
            model.load_state_dict(saved["encoder"])
            metrics, _ = evaluate(model, reference, c, seed)
            for key, value in metrics.items():
                if not np.isclose(value, r["conditions"][name]["metrics"][key], rtol=0, atol=1e-7):
                    raise RuntimeError(f"Evaluation mismatch: {seed}, {name}, {key}")
    report(out, result)
    print(f"Reproduced all {len(result['runs']) * len(CONDITIONS)} saved evaluations and sampling-plan hashes.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seeds only after verifying configuration")
    args = parser.parse_args()
    torch.set_num_threads(1)
    upstream = behavior.verify_upstream()
    if args.evaluate_only:
        evaluate_saved(args.output)
        return
    c = behavior.Config()
    if args.quick:
        c.policy_updates, c.jepa_steps = 2, 40
        c.nenv, c.rollout = 4, 4
        c.train_episodes, c.eval_episodes, c.horizon, c.burnin, c.batch = 8, 4, 8, 4, 16
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"config": asdict(c), "quick": args.quick, "seeds": args.seeds,
              "upstream_commit": upstream, "runs": []}
    if args.resume and (args.output / "metrics.json").exists():
        result = json.loads((args.output / "metrics.json").read_text())
        if result["config"] != asdict(c) or result["quick"] != args.quick or result["seeds"] != args.seeds:
            raise RuntimeError("Resume configuration differs")
    for seed in args.seeds:
        if any(r["seed"] == seed for r in result["runs"]):
            continue
        result["runs"].append(train_seed(c, seed, args.output, args.quick))
        report(args.output, result)
    print("Curriculum controls complete.", flush=True)


if __name__ == "__main__":
    main()

"""Policy zoo: many behavior policies under the reliability-study protocol.

Every fit reuses reliability.run_study exactly: encoder initialization
40000+seed, the historical pair-index stream, 4,000 Adam updates of batch 512
and the original learning-rate schedule. Two protocols:

  handover  policy data, then the original 25%->75% OU handover (as before)
  pure      policy data only, for the whole budget (no OU pairs at all)

Latent Gaussian sources are run "pure" only; pure fixed OU equals the earlier
fixed-OU condition. Runs are paired across conditions by data seed.
"""

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import zoo
from zoo import behavior, controls, learned, reliability

SEEDS = list(range(8, 28))
# Exploratory, added after the main results (see PROTOCOL.md): do negative-rho
# sources reach the lower-loss folded optimum with twice the budget?
LONG = ["ou", "ou_hp2_w1", "ou_hp2_w3", "iid_ou_hp2_w1", "iid_ou_hp2_w3"]
HERE = Path(__file__).resolve().parent
RELIABILITY = HERE.parent / "reliability"
EXISTING_CACHES = [RELIABILITY / "output-repairs", RELIABILITY / "output-policies"]


def jobs_for(seeds):
    jobs = []
    for seed in seeds:
        for name in zoo.PHYSICAL:
            jobs += [("handover", name, seed), ("pure", name, seed)]
        jobs += [("pure", name, seed) for name in zoo.LATENT]
        jobs += [("pure_long", name, seed) for name in LONG]
    return jobs


# ----------------------------------------------------------------- sources
def learned_source(c, seed, kind, out, quick):
    """Learned/adaptive sources; earlier reliability caches are reused."""
    filename = f"{kind}_seed{seed}.pt"
    for root in [] if quick else EXISTING_CACHES:
        path = root / "sources" / filename
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            if saved["config"] != asdict(c):
                raise RuntimeError(f"Cached source configuration differs: {path}")
            return saved["data"], {"reused": str(path.relative_to(HERE.parent))}
    path = out / "sources" / filename
    if path.exists():
        saved = torch.load(path, map_location="cpu")
        if saved["config"] != asdict(c):
            raise RuntimeError(f"Cached source configuration differs: {path}")
        return saved["data"], {"policy": f"policies/{filename}"}
    if kind == "task":
        policy, saved = learned.train(c, seed, "task", reward_fn=zoo.task_reward)
        data = learned.collect(c, "task", c.train_episodes, 10000 + seed, policy)
        (out / "sources").mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(c), "policy": saved, "data": data}, path)
    else:
        data, saved = reliability.prepare_data(c, seed, kind, out, quick)
    (out / "policies").mkdir(parents=True, exist_ok=True)
    # Datasets are large and regenerable; keep the small frozen policy only.
    torch.save(saved, out / "policies" / filename)
    return data, {"policy": f"policies/{filename}"}


def source(c, name, seed, out, quick=False):
    if name == "random":
        return behavior.collect(c, "random", c.train_episodes, 10000 + seed), {}
    if name in zoo.LEARNED or name in {"count", "action_balanced"}:
        return learned_source(c, seed, name, out, quick)
    if name in zoo.PHYSICAL_RULES:
        policy = zoo.PHYSICAL_RULES[name]()
        return zoo.collect_physical(c, policy, c.train_episodes, 10000 + seed), {}
    if name == "ou" or name.startswith("iid_"):
        return zoo.collect_pairs(c, zoo.source_rho(name), seed), {}
    return zoo.collect_latent(c, zoo.LATENT_FILTERS[name], seed), {}


def diagnostics(data, c):
    """Label-using descriptions of the source; never seen by fitting."""
    z, zp = data["z"], data["zp"]
    bins = learned.size_bin(data["x"])
    inside = (data["x"][:, 1] >= 1 / 16) & (data["x"][:, 1] <= 1)
    p = torch.bincount(bins[inside], minlength=32).float()
    p = p / p.sum().clamp_min(1)
    entropy = -(p[p > 0] * p[p > 0].log()).sum() / np.log(32)
    stats = behavior.distribution_stats(z, zp)
    return {
        "lag_correlation": stats["lag_correlation"],
        "gaussian_cdf_distance": stats["standardized_gaussian_cdf_distance"],
        "z_std": float(z.std()),
        "size_bin_entropy": float(entropy),
        "size_bin_coverage": float((p > 0).float().mean()),
        "max_size_bin_mass": float(p.max()),
        "fraction_in_sensor_range": float(inside.float().mean()),
        "stay_fraction": float(((zp - z).abs() < 1e-9).float().mean()),
        "mean_abs_step": float((zp - z).abs().mean()),
        "sha256": hashlib.sha256(
            data["x"].numpy().tobytes() + data["xp"].numpy().tobytes()
        ).hexdigest(),
    }


# ---------------------------------------------------------------- fitting
def fit_pure(views, c, seed, reference):
    """reliability.fit with the OU fraction held at zero for every update."""
    model, plan = reliability.make_plan(c, seed, seed, "icm")
    plan["alpha"][:] = 0
    plan["masks"][:] = False
    plan["sha256"] = hashlib.sha256(plan["indices"].numpy().tobytes()).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer, history = behavior.SIGReg1D(), []
    for step in range(c.jepa_steps):
        losses = controls.update(
            model,
            optimizer,
            regularizer,
            *controls.batch_at(views, views, plan, step),
            c,
            controls.learning_rate(step, c.jepa_steps),
        )
        if step % 512 == 0 or step == c.jepa_steps - 1:
            metrics, _ = controls.evaluate(model, reference, c, seed)
            history.append({"updates": step + 1, "ou_fraction": 0.0, **losses, **metrics})
    return model.eval(), history, plan["sha256"]


def run_job(job, c, out, quick):
    protocol, name, seed = job
    label = f"{protocol}-{name}"
    path = out / "runs" / f"{label}_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(1)
    reference = (
        behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
        behavior.collect(c, "ou", c.eval_episodes, 30000 + seed),
    )
    data, provenance = source(c, name, seed, out, quick)
    if protocol == "pure_long":
        c = copy.deepcopy(c)
        c.jepa_steps *= 2  # same doubling as the reliability *_long runs
    views = {k: data[k] for k in ["x", "xp"]}  # labels/actions never fitted
    if protocol == "handover":
        model, history, plan_hash = reliability.fit(views, "icm", c, seed, seed, reference)
    else:
        model, history, plan_hash = fit_pure(views, c, seed, reference)
    record = reliability.metric_record(
        model, history, plan_hash, label, seed, seed, c, reference, out / "checkpoints"
    )
    record.pop("validation")
    record.update(protocol=protocol, source=name, provenance=provenance,
                  source_stats=diagnostics(data, c))
    if name == "ou" or name in zoo.LATENT_FILTERS or name.startswith("iid_"):
        record["source_stats"]["theoretical_lag_correlation"] = zoo.source_rho(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"seed": seed, "run": label, "r2": record["metrics"]["common_grid_r2"]}),
          flush=True)
    return record


def _prepare(args):
    kind, seed, c, out, quick = args
    torch.set_num_threads(1)
    learned_source(c, seed, kind, out, quick)
    return kind, seed


def _run(args):
    return run_job(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--quick", action="store_true", help="Execution check only")
    args = parser.parse_args()
    torch.set_num_threads(1)
    behavior.verify_upstream()
    c = behavior.Config()
    if args.quick:
        c.jepa_steps, c.policy_updates, c.train_episodes, c.eval_episodes = 40, 2, 8, 4
        c.nenv, c.rollout, c.horizon, c.burnin, c.batch = 4, 4, 8, 4, 16
        args.seeds = args.seeds[:1]
    out = args.output
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"config": asdict(c), "seeds": args.seeds, "quick": args.quick,
         "physical": zoo.PHYSICAL, "latent": zoo.LATENT}, indent=2) + "\n")
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        # Learned policies first, so fit workers only ever read their caches.
        prep = [(k, s, c, out, args.quick) for s in args.seeds
                for k in zoo.LEARNED + ["count", "action_balanced"]]
        for kind, seed in pool.imap_unordered(_prepare, prep):
            print(f"source ready: {kind} seed={seed}", flush=True)
        jobs = [(job, c, out, args.quick) for job in jobs_for(args.seeds)]
        done = sum(1 for _ in pool.imap_unordered(_run, jobs))
    print(f"Completed {done} fits in {out}", flush=True)


if __name__ == "__main__":
    main()

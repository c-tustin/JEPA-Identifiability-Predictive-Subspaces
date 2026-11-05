"""Run discovery and fresh-seed confirmation of the targeted policy catalog."""

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_zoo
import targeted_policies
import zoo
from zoo import behavior, controls, reliability
import numpy as np
import torch

OUT = HERE / "output"
OLD = HERE.parent / "output"
DISCOVERY = tuple(range(8, 28))
CONFIRMATION = tuple(range(28, 48))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def source_path(name, seed):
    return OUT / "sources" / f"{name}-{seed}.pt"


def prepare_source(job):
    name, seed = job
    torch.set_num_threads(1)
    c = behavior.Config()
    path = source_path(name, seed)
    if path.exists():
        saved = load(path)
        assert saved["config"] == asdict(c)
        return saved["statistics"]
    if name in targeted_policies.POLICIES:
        policy = targeted_policies.POLICIES[name]()
        data = zoo.collect_physical(c, policy, c.train_episodes, 10000 + seed)
    else:
        data, _ = run_zoo.source(c, name, seed, OUT)
    statistics = run_zoo.diagnostics(data, c)
    save(path, dict(config=asdict(c), data=data, statistics=statistics))
    atomic_json(OUT / "source-statistics" / f"{name}-{seed}.json", statistics)
    print(f"SOURCE {name} seed {seed}", flush=True)
    return statistics


def reference(c, seed):
    return (
        behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
        behavior.collect(c, "ou", c.eval_episodes, 30000 + seed),
    )


def geometry(model, c):
    with torch.no_grad():
        z = torch.linspace(np.log(1 / 4) / c.sigma, np.log(16 / 4) / c.sigma, 1024)
        y = model(behavior.observe(z, c.sigma))
        sign = 1 if torch.cov(torch.stack([z, y]))[0, 1] >= 0 else -1
        display = sign * (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-8)
        return dict(
            distance=(4 * torch.exp(c.sigma * z)).numpy(),
            embedding=display.numpy(),
            reversed_grid_fraction=float((display.diff() < -1e-6).float().mean()),
        )


def fit(job):
    stage, protocol, name, seed = job
    torch.set_num_threads(1)
    c = behavior.Config()
    label = f"{protocol}-{name}"
    record_path = OUT / "runs" / stage / f"{label}_seed{seed}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    source = load(source_path(name, seed))
    data = source["data"]
    assert run_zoo.diagnostics(data, c)["sha256"] == source["statistics"]["sha256"]
    evaluation = reference(c, seed)
    model, plan = reliability.make_plan(c, seed, seed, "icm")
    if protocol == "pure":
        plan["masks"][:] = False
    elif protocol != "handover":
        raise ValueError(protocol)
    ou = controls.ou_data(c, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer = behavior.SIGReg1D()
    initial_metrics, _ = controls.evaluate(model, evaluation, c, seed)
    history, start = [], 0
    partial = OUT / "partials" / f"{stage}-{label}-{seed}.pt"
    if partial.exists():
        previous = load(partial)
        assert previous["source_sha256"] == source["statistics"]["sha256"]
        model.load_state_dict(previous["encoder"])
        optimizer.load_state_dict(previous["optimizer"])
        start, history = previous["step"], previous["history"]
    for step in range(start, c.jepa_steps):
        x, xp = controls.batch_at(data, ou, plan, step)
        losses = controls.update(
            model, optimizer, regularizer, x, xp, c,
            controls.learning_rate(step, c.jepa_steps),
        )
        if (step + 1) % 1000 == 0:
            metrics, _ = controls.evaluate(model, evaluation, c, seed)
            history.append(dict(updates=step + 1, **losses, **metrics))
            save(partial, dict(
                step=step + 1, encoder=model.state_dict(), optimizer=optimizer.state_dict(),
                history=history, source_sha256=source["statistics"]["sha256"],
            ))
    metrics, _ = controls.evaluate(model, evaluation, c, seed)
    curve = geometry(model, c)
    metrics["reversed_grid_fraction"] = curve["reversed_grid_fraction"]
    checkpoint = OUT / "checkpoints" / stage / f"{label}_seed{seed}.pt"
    save(checkpoint, dict(
        encoder=model.state_dict(), config=asdict(c), seed=seed,
        source=name, protocol=protocol, curve=curve,
    ))
    record = dict(
        stage=stage, protocol=protocol, source=name, seed=seed,
        metrics=metrics, initial_metrics=initial_metrics, history=history,
        source_statistics=source["statistics"],
        pair_indices_sha256=hashlib.sha256(plan["indices"].numpy().tobytes()).hexdigest(),
        masks_sha256=hashlib.sha256(plan["masks"].numpy().tobytes()).hexdigest(),
        checkpoint=str(checkpoint.relative_to(OUT)),
    )
    atomic_json(record_path, record)
    print(f"DONE {stage} {label} seed {seed}: {metrics['common_grid_r2']:.6f}", flush=True)
    return record


def preflight():
    rows = []
    for protocol, name in (("pure", "ou"), ("handover", "random")):
        prepare_source((name, 8))
        result = fit(("preflight", protocol, name, 8))
        new = load(OUT / result["checkpoint"])["encoder"]
        old = load(OLD / "checkpoints" / f"encoder_{protocol}-{name}_data8_init8.pt")["encoder"]
        difference = max(float((value - old[key]).abs().max()) for key, value in new.items())
        assert difference == 0.0, (protocol, name, difference)
        rows.append(dict(protocol=protocol, source=name, max_parameter_difference=difference))
    atomic_json(OUT / "preflight.json", dict(passed=True, checks=rows))


def select():
    selected = {}
    scores = {}
    for protocol in ("pure", "handover"):
        means = {}
        for name in targeted_policies.POLICIES:
            records = [json.loads((OUT / "runs" / "discovery" / f"{protocol}-{name}_seed{s}.json").read_text()) for s in DISCOVERY]
            means[name] = float(np.mean([r["metrics"]["common_grid_r2"] for r in records]))
        selected[protocol] = sorted(means, key=lambda name: (-means[name], name))[:2]
        scores[protocol] = means
    result = dict(selected=selected, discovery_means=scores,
                  rule="Top two by discovery mean R² within protocol; ties by name; no fold/p-value selection.")
    path = OUT / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        atomic_json(path, result)
    return selected


def run_stage(pool, stage, combinations, seeds):
    source_jobs = sorted({(name, seed) for _, name in combinations for seed in seeds})
    list(pool.imap_unordered(prepare_source, source_jobs))
    jobs = [(stage, protocol, name, seed) for protocol, name in combinations for seed in seeds]
    list(pool.imap_unordered(fit, jobs))
    atomic_json(OUT / f"{stage}-complete.json", dict(fits=len(jobs), seeds=list(seeds)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stage", choices=("all", "discovery", "confirmation", "preflight"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    code_paths = [Path(__file__), HERE.parent / "targeted_policies.py", Path(behavior.__file__),
                  Path(controls.__file__), Path(reliability.__file__), Path(zoo.__file__)]
    settings = dict(
        config=asdict(behavior.Config()), discovery=list(DISCOVERY), confirmation=list(CONFIRMATION),
        policies=list(targeted_policies.POLICIES), protocol_sha256=digest(HERE / "PROTOCOL.md"),
        code_sha256={str(path.relative_to(HERE.parents[2])): digest(path) for path in code_paths},
    )
    path = OUT / "settings.json"
    if path.exists():
        assert json.loads(path.read_text()) == settings, "Code/protocol changed; use a separate follow-up"
    else:
        atomic_json(path, settings)
    preflight()
    if args.stage == "preflight":
        return
    with mp.get_context("spawn").Pool(args.workers) as pool:
        if args.stage in ("all", "discovery"):
            combinations = [(p, n) for p in ("pure", "handover") for n in targeted_policies.POLICIES]
            run_stage(pool, "discovery", combinations, DISCOVERY)
        selected = select()
        if args.stage in ("all", "confirmation"):
            combinations = [(p, n) for p, names in selected.items() for n in names]
            combinations += [("pure", "ou"), ("pure", "count"), ("handover", "biased"),
                             ("pure", "random"), ("handover", "random")]
            run_stage(pool, "confirmation", combinations, CONFIRMATION)


if __name__ == "__main__":
    main()

"""Recreate every reported score and replay representative training runs.

Use --study targeted-policies or --study broad-policies. Verification reads the
recorded settings and never selects policies using confirmation performance.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from math import comb
import multiprocessing as mp
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def runner(study):
    sys.path.insert(0, str(HERE / study))
    return importlib.import_module("run_targeted" if study == "targeted-policies" else "run_broad")


def catalog(module):
    policy_module = getattr(module, "targeted_policies", None)
    if policy_module is None:
        policy_module = module.broad_policies
    return policy_module, getattr(policy_module, "COLLECTION_POLICIES", policy_module.POLICIES)


def replay(job):
    study, stage, protocol, name, seed = job
    module = runner(study)
    torch.set_num_threads(1)
    c = module.behavior.Config()
    source = module.load(module.source_path(name, seed))["data"]
    model, plan = module.reliability.make_plan(c, seed, seed, "icm")
    if protocol == "pure":
        plan["masks"][:] = False
    ou = module.controls.ou_data(c, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    regularizer = module.behavior.SIGReg1D()
    for step in range(c.jepa_steps):
        x, xp = module.controls.batch_at(source, ou, plan, step)
        module.controls.update(model, optimizer, regularizer, x, xp, c,
                               module.controls.learning_rate(step, c.jepa_steps))
    record = json.loads((module.OUT / "runs" / stage / f"{protocol}-{name}_seed{seed}.json").read_text())
    saved = module.load(module.OUT / record["checkpoint"])["encoder"]
    error = max(float((value - saved[key]).abs().max()) for key, value in model.state_dict().items())
    assert error == 0.0, (job, error)
    print(f"REPLAY {study} {stage} {protocol} {name} seed {seed}: exact", flush=True)
    return dict(stage=stage, protocol=protocol, policy=name, seed=seed, max_parameter_error=error)


def check_test_family(rows, records, seeds):
    for row in rows:
        a = [records[row["protocol"], row["policy"], s]["metrics"]["common_grid_r2"] for s in seeds]
        b = [records[row["reference_protocol"], row["reference"], s]["metrics"]["common_grid_r2"] for s in seeds]
        difference = np.asarray(a) - np.asarray(b)
        assert np.array_equal(difference, np.asarray(row["differences"]))
        wins = int((difference > 1e-10).sum())
        losses = int((difference < -1e-10).sum())
        assert (wins, losses) == (row["wins"], row["losses"])
        n = wins + losses
        expected = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
        assert abs(expected - row["p"]) < 1e-12
        assert abs(float(difference.mean()) - row["mean_difference"]) < 1e-14
    passing = True
    for rank, row in enumerate(sorted(rows, key=lambda r: r["p"])):
        passing = passing and row["p"] < .05 / (len(rows) - rank)
        assert (row["p_holm"] < .05) == passing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("targeted-policies", "broad-policies"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    module = runner(args.study)
    policy_module, collection = catalog(module)
    torch.set_num_threads(1)
    c = module.behavior.Config()
    settings = json.loads((module.OUT / "settings.json").read_text())
    assert settings["config"] == asdict(c)
    assert settings["protocol_sha256"] == module.digest(module.HERE / "PROTOCOL.md")
    for path, expected in settings["code_sha256"].items():
        assert module.digest(module.HERE.parents[2] / path) == expected, path

    selected = json.loads((module.OUT / "selection.json").read_text())
    for protocol, names in selected["selected"].items():
        means = {}
        for name in policy_module.POLICIES:
            values = [json.loads((module.OUT / "runs" / "discovery" / f"{protocol}-{name}_seed{s}.json").read_text())["metrics"]["common_grid_r2"] for s in module.DISCOVERY]
            means[name] = float(np.mean(values))
        assert names == sorted(means, key=lambda name: (-means[name], name))[:2]
        assert means == selected["discovery_means"][protocol]
    if args.study == "broad-policies":
        earlier = json.loads((HERE / "targeted-policies" / "output" / "selection.json").read_text())
        previous = json.loads((module.OUT / "previous-addition-references.json").read_text())
        assert previous == {p: names[0] for p, names in earlier["selected"].items()}
        assert set(module.CONFIRMATION).isdisjoint(range(8, 48))

    source_statistics = {}
    plans = {}
    references = {}
    initial_scores = {}
    counts = {}
    max_score_error = 0.0
    max_curve_error = 0.0
    for stage, seeds in (("discovery", module.DISCOVERY), ("confirmation", module.CONFIRMATION)):
        complete = json.loads((module.OUT / f"{stage}-complete.json").read_text())
        paths = sorted((module.OUT / "runs" / stage).glob("*.json"))
        assert len(paths) == complete["fits"]
        records = {}
        for path in paths:
            record = json.loads(path.read_text())
            seed, name, protocol = record["seed"], record["source"], record["protocol"]
            assert seed in seeds
            records[protocol, name, seed] = record
            key = (name, seed)
            if key not in source_statistics:
                source = module.load(module.source_path(name, seed))
                assert source["config"] == asdict(c)
                statistics = module.run_zoo.diagnostics(source["data"], c)
                assert statistics["sha256"] == source["statistics"]["sha256"]
                source_statistics[key] = statistics
                if name in collection:
                    data = source["data"]
                    distance = 4 * (c.sigma * data["z"]).exp()
                    proposed = distance + (data["action"].float() - 1) * .25
                    assert bool(((proposed >= 1 - 2e-5) & (proposed <= 16 + 2e-5)).all())
            assert record["source_statistics"]["sha256"] == source_statistics[key]["sha256"]
            if seed not in references:
                references[seed] = module.reference(c, seed)
                initial, plan = module.reliability.make_plan(c, seed, seed, "icm")
                initial_scores[seed] = module.controls.evaluate(initial, references[seed], c, seed)[0]
                plans[seed, "indices"] = hashlib.sha256(plan["indices"].numpy().tobytes()).hexdigest()
                plans[seed, "handover"] = hashlib.sha256(plan["masks"].numpy().tobytes()).hexdigest()
                plans[seed, "pure"] = hashlib.sha256(torch.zeros_like(plan["masks"]).numpy().tobytes()).hexdigest()
            assert record["pair_indices_sha256"] == plans[seed, "indices"]
            assert record["masks_sha256"] == plans[seed, protocol]
            assert record["initial_metrics"] == initial_scores[seed]
            saved = module.load(module.OUT / record["checkpoint"])
            model = module.behavior.Encoder()
            model.load_state_dict(saved["encoder"])
            metrics, _ = module.controls.evaluate(model, references[seed], c, seed)
            for field, value in metrics.items():
                error = abs(value - record["metrics"][field])
                assert error < 1e-10, (path, field, error)
                max_score_error = max(max_score_error, error)
            curve = module.geometry(model, c)
            assert curve["reversed_grid_fraction"] == record["metrics"]["reversed_grid_fraction"]
            error = float(np.max(np.abs(curve["embedding"] - saved["curve"]["embedding"])))
            assert error == 0.0
            max_curve_error = max(max_curve_error, error)
        summary = json.loads((module.OUT / f"{stage}-summary.json").read_text())
        # Discovery summaries also include historical comparison records.
        for row in summary["conditions"]:
            for seed in seeds:
                key = (row["protocol"], row["policy"], seed)
                if key not in records:
                    old_path = module.OLD / "runs" / f"{key[0]}-{key[1]}_seed{seed}.json"
                    if not old_path.exists():
                        old_path = HERE / "targeted-policies" / "output" / "runs" / stage / f"{key[0]}-{key[1]}_seed{seed}.json"
                    records[key] = json.loads(old_path.read_text())
            score = np.array([records[row["protocol"], row["policy"], seed]["metrics"]["common_grid_r2"] for seed in seeds])
            assert float(score.mean()) == row["mean"]
            folds = sum(records[row["protocol"], row["policy"], seed]["metrics"]["reversed_grid_fraction"] > 0 for seed in seeds)
            assert folds == row["folds"]
        for rows in summary["tests"].values():
            check_test_family(rows, records, seeds)
        counts[stage] = len(paths)

    regenerated = []
    regenerate_jobs = [(name, module.DISCOVERY[0]) for name in policy_module.POLICIES]
    regenerate_jobs += sorted({(name, module.CONFIRMATION[0]) for names in selected["selected"].values() for name in names})
    for name, seed in regenerate_jobs:
        data = module.zoo.collect_physical(c, collection[name](), c.train_episodes, 10000 + seed)
        actual = module.run_zoo.diagnostics(data, c)["sha256"]
        assert actual == source_statistics[name, seed]["sha256"]
        regenerated.append(dict(policy=name, seed=seed))

    jobs = [(args.study, "discovery", "pure", name, module.DISCOVERY[0]) for name in policy_module.POLICIES]
    jobs += [(args.study, "confirmation", p, name, module.CONFIRMATION[0]) for p, names in selected["selected"].items() for name in names]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        replays = list(pool.imap_unordered(replay, jobs))
    result = dict(
        passed=True, models_checked=counts, sources_checked=len(source_statistics),
        max_metric_difference=max_score_error, max_curve_difference=max_curve_error,
        regenerated_new_policy_sources=regenerated, full_training_replays=replays,
        scope="Every saved model/score/fold/curve, source fingerprints, seed pairing, frozen discovery selection, and every test family; representative source regeneration and full training replays, not every training history.",
    )
    module.atomic_json(module.OUT / "verification.json", result)
    print("VERIFIED", args.study, counts, flush=True)


if __name__ == "__main__":
    main()

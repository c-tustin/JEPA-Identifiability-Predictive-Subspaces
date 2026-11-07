"""Separate dataset/minibatch effects from encoder initialization effects."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_zoo
from zoo import behavior, controls, reliability
import numpy as np
import torch

OUT = HERE / "output"
ZOO_OUT = HERE.parent / "output"
LEVELS = (8, 9, 10, 11, 12)
POLICIES = ("icm", "random", "count", "ou")
EVAL_SEED = 20261104


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(model, c):
    q = torch.linspace(0, np.log(16), 1025, requires_grad=True)
    y = model(behavior.sensor(q.exp(), c))
    slope = torch.autograd.grad(y.sum(), q)[0]
    return dict(log_distance=q.detach().numpy(), curve=y.detach().numpy(), slope=slope.detach().numpy())


def reference(c, seed):
    return (behavior.collect(c, "ou", c.eval_episodes, 20000+seed),
            behavior.collect(c, "ou", c.eval_episodes, 30000+seed))


def prepare():
    c = behavior.Config(**json.loads((ZOO_OUT/"config.json").read_text())["config"])
    for policy in POLICIES:
        for seed in LEVELS:
            path = OUT/"sources"/f"{policy}-{seed}.pt"
            if path.exists():
                data = load(path)
            else:
                data, _ = run_zoo.source(c, policy, seed, OUT)
                save(path, data)
            expected = json.loads((ZOO_OUT/"runs"/f"pure-{policy}_seed{seed}.json").read_text())
            actual = run_zoo.diagnostics(data, c)["sha256"]
            assert actual == expected["source_stats"]["sha256"], (policy, seed)
    return c


def job(args):
    policy, data_seed, init_seed = args
    torch.set_num_threads(1)
    path = OUT/"crossed"/f"{policy}-data{data_seed}-init{init_seed}.pt"
    if path.exists():
        return str(path.name)
    c = behavior.Config(**json.loads((ZOO_OUT/"config.json").read_text())["config"])
    source = load(OUT/"sources"/f"{policy}-{data_seed}.pt")
    model, plan = reliability.make_plan(c, data_seed, init_seed, "icm")
    initial = geometry(model, c)
    common_reference = reference(c, EVAL_SEED)
    initial_metrics, _ = controls.evaluate(model, common_reference, c, EVAL_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    regularizer = behavior.SIGReg1D()
    partial = OUT/"partials"/path.name
    start = 0
    if partial.exists():
        old = load(partial)
        model.load_state_dict(old["encoder"])
        optimizer.load_state_dict(old["optimizer"])
        start = old["steps"]
    for step in range(start, c.jepa_steps):
        ix = plan["indices"][step]
        controls.update(model, optimizer, regularizer, source["x"][ix], source["xp"][ix], c,
                        controls.learning_rate(step, c.jepa_steps))
        if (step+1) % 1000 == 0:
            save(partial, dict(encoder=model.state_dict(), optimizer=optimizer.state_dict(), steps=step+1))
    original_metrics, _ = controls.evaluate(model, reference(c, data_seed), c, data_seed)
    common_metrics, _ = controls.evaluate(model, common_reference, c, EVAL_SEED)
    result = dict(policy=policy, data_seed=data_seed, init_seed=init_seed,
                  encoder=model.state_dict(), initial=initial, final=geometry(model, c),
                  initial_metrics=initial_metrics, metrics=common_metrics, original_metrics=original_metrics,
                  pair_indices_sha256=hashlib.sha256(plan["indices"].numpy().tobytes()).hexdigest())
    if data_seed == init_seed:
        old = load(ZOO_OUT/"checkpoints"/f"encoder_pure-{policy}_data{data_seed}_init{init_seed}.pt")
        difference = max(float((v-old["encoder"][k]).abs().max()) for k,v in model.state_dict().items())
        expected = json.loads((ZOO_OUT/"runs"/f"pure-{policy}_seed{data_seed}.json").read_text())
        metric_error = abs(original_metrics["common_grid_r2"]-expected["metrics"]["common_grid_r2"])
        assert difference == 0. and metric_error < 1e-7, (policy, data_seed, difference, metric_error)
        result["diagonal_verification"] = dict(max_parameter_difference=difference, r2_error=metric_error)
    save(path, result)
    print(f"DONE {policy} data {data_seed} init {init_seed}: {common_metrics['common_grid_r2']:.4f}", flush=True)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    settings = dict(levels=LEVELS, policies=POLICIES, common_evaluation_seed=EVAL_SEED,
                    code_sha256=sha(Path(__file__)), protocol_sha256=sha(HERE/"PROTOCOL.md"))
    settings = json.loads(json.dumps(settings))
    settings_path = OUT/"crossed-settings.json"
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == settings
    else:
        settings_path.write_text(json.dumps(settings, indent=2)+"\n")
    prepare()
    diagonal = [(p, d, d) for p in POLICIES for d in LEVELS]
    off_diagonal = [(p, d, i) for p in POLICIES for d in LEVELS for i in LEVELS if d != i]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        # No crossed fit starts until all 20 original fits reproduce.
        list(pool.imap_unordered(job, diagonal))
        list(pool.imap_unordered(job, off_diagonal))


if __name__ == "__main__":
    main()

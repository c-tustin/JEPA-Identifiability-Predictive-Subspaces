"""Trace pure-ICM learning and isolate the SIGReg sampling intervention."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_location import ScaledEncoder, encoder_gain, trials, ORIGINAL
import numpy as np
import torch
from torch.func import functional_call, jvp

OUT = HERE / "output"
GAINS = (2, 4, 8)
DIAGNOSTIC_SEEDS = (130, 131, 136)
SNAPSHOT_STEPS = (0, 25, 100, 250, 500, 1000, 2000, 3000)


def frozen_parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def cpu_copy(value):
    """Snapshot mutable optimizer tensors rather than retaining live references."""
    return copy.deepcopy(value)


def evaluator(model, c):
    log_distance = torch.linspace(0, np.log(16), 257)

    def evaluate(parameters):
        def forward(q):
            observation = trials.behavior.sensor(q.exp(), c)
            return functional_call(model, parameters, (observation,))
        return jvp(forward, (log_distance,), (torch.ones_like(log_distance),))

    return log_distance, evaluate


def adam_directions(optimizer, named, moment_a, moment_r, step, lr):
    """Split the actual Adam numerator, retaining its combined denominator."""
    a, r = {}, {}
    moment_error = 0.0
    for name, parameter in named:
        state = optimizer.state[parameter]
        denominator = (state["exp_avg_sq"] / (1 - 0.999 ** step)).sqrt() + 1e-8
        correction = 1 - 0.9 ** step
        a[name] = -lr * moment_a[name] / correction / denominator
        r[name] = -lr * moment_r[name] / correction / denominator
        error = (moment_a[name] + moment_r[name] - state["exp_avg"]).abs().max()
        moment_error = max(moment_error, float(error))
    return a, r, moment_error


def fit(c, seed, gain, condition, out):
    key = f"{condition}-g{gain}-{seed}"
    result_file = out / "runs" / f"{key}.pt"
    if result_file.exists():
        return trials.load(result_file)
    detailed = seed in DIAGNOSTIC_SEEDS
    source = trials.load(ORIGINAL / "sources" / f"icm-{seed}.pt")["data"]
    ou = trials.study.controls.ou_data(c, seed)
    with encoder_gain(gain):
        model, plan = trials.study.make_plan(c, seed, seed, "icm")
    plan["masks"] = torch.zeros_like(plan["masks"])
    reg_plan = copy.copy(plan)
    reg_plan["indices"] = torch.randint(
        len(source["x"]), plan["indices"].shape,
        generator=torch.Generator().manual_seed(1900000 + seed),
    )
    reg_source = trials.replay_pairs(source, seed, "log") if condition == "balanced" else source
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer = trials.behavior.SIGReg1D()
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    moment_a = {name: torch.zeros_like(p) for name, p in named}
    moment_r = {name: torch.zeros_like(p) for name, p in named}
    q, evaluate = evaluator(model, c)
    frames, snapshots, history = [], {}, []
    max_update_error, max_moment_error = 0.0, 0.0
    start = 0
    partial = out / "partials" / f"{key}.pt"
    if partial.exists():
        previous = trials.load(partial)
        if previous["config"] != asdict(c):
            raise RuntimeError("Partial mechanism configuration differs")
        model.load_state_dict(previous["encoder"])
        optimizer.load_state_dict(previous["optimizer"])
        moment_a, moment_r = previous["moment_a"], previous["moment_r"]
        frames, snapshots, history = previous["frames"], previous["snapshots"], previous["history"]
        max_update_error, max_moment_error = previous["max_update_error"], previous["max_moment_error"]
        start = previous["step"]
        torch.set_rng_state(previous["rng"])
    started = time.perf_counter()
    for step in range(start, c.jepa_steps):
        lr = trials.study.controls.learning_rate(step, c.jepa_steps)
        optimizer.param_groups[0]["lr"] = lr
        x, xp = trials.study.controls.batch_at(source, ou, plan, step)
        y = model(torch.cat([x, xp]))
        alignment = (y[:len(x)] - y[len(x):]).square().mean()
        rx, rxp = trials.study.controls.batch_at(reg_source, ou, reg_plan, step)
        penalty = regularizer(model(torch.cat([rx, rxp])))
        gradient_a = torch.autograd.grad(0.9 * alignment, parameters)
        gradient_r = torch.autograd.grad(0.1 * penalty, parameters)
        if not torch.isfinite(alignment + penalty):
            raise RuntimeError("Nonfinite mechanism loss")
        probe = step % 25 == 0
        before = frozen_parameters(model) if probe else None
        snapshot = None
        if detailed and step in SNAPSHOT_STEPS:
            snapshot = dict(
                before=before, optimizer_before=cpu_copy(optimizer.state_dict()),
                gradient_a={name: g.detach().clone() for (name, _), g in zip(named, gradient_a)},
                gradient_r={name: g.detach().clone() for (name, _), g in zip(named, gradient_r)},
                moment_a_before=cpu_copy(moment_a), moment_r_before=cpu_copy(moment_r),
            )
        optimizer.zero_grad(set_to_none=True)
        for (name, parameter), ga, gr in zip(named, gradient_a, gradient_r):
            parameter.grad = ga + gr
            moment_a[name].mul_(0.9).add_(ga, alpha=0.1)
            moment_r[name].mul_(0.9).add_(gr, alpha=0.1)
        optimizer.step()
        if probe:
            after = frozen_parameters(model)
            frame = dict(step=step, alignment=float(alignment.detach()), sigreg=float(penalty.detach()))
            if detailed:
                da, dr, error = adam_directions(optimizer, named, moment_a, moment_r, step+1, lr)
                max_moment_error = max(max_moment_error, error)
                difference = max(float((after[k]-before[k]-da[k]-dr[k]).abs().max()) for k in before)
                max_update_error = max(max_update_error, difference)
                (by, bs), (dy_a, ds_a) = jvp(evaluate, (before,), (da,))
                _, (dy_r, ds_r) = jvp(evaluate, (before,), (dr,))
                frame.update(
                    delta_curve_a=dy_a.detach().numpy(), delta_curve_r=dy_r.detach().numpy(),
                    delta_slope_a=ds_a.detach().numpy(), delta_slope_r=ds_r.detach().numpy(),
                    update_reconstruction_error=difference, moment_reconstruction_error=error,
                    gradient_a_norm=float(torch.cat([g.flatten() for g in gradient_a]).norm()),
                    gradient_r_norm=float(torch.cat([g.flatten() for g in gradient_r]).norm()),
                )
            else:
                by, bs = evaluate(before)
            ay, ass = evaluate(after)
            frame.update(curve_before=by.detach().numpy(), slope_before=bs.detach().numpy(),
                         curve_after=ay.detach().numpy(), slope_after=ass.detach().numpy())
            frames.append(frame)
            if snapshot is not None:
                snapshot.update(after=after, optimizer_after=cpu_copy(optimizer.state_dict()),
                                direction_a=da, direction_r=dr, lr=lr)
                snapshots[step] = snapshot
        if (step+1) % 500 == 0:
            history.append(dict(step=step+1, alignment=float(alignment.detach()), sigreg=float(penalty.detach())))
            trials.save_torch(partial, dict(
                config=asdict(c), step=step+1, encoder=model.state_dict(),
                optimizer=optimizer.state_dict(), moment_a=moment_a, moment_r=moment_r,
                frames=frames, snapshots=snapshots, history=history, rng=torch.get_rng_state(),
                max_update_error=max_update_error, max_moment_error=max_moment_error,
            ))
    final_y, final_s = evaluate(frozen_parameters(model))
    diagnostic = trials.representation(model.eval(), c, seed)
    result = dict(
        seed=seed, gain=gain, condition=condition, config=asdict(c), steps=c.jepa_steps,
        encoder=model.state_dict(), log_distance=q.numpy(), frames=frames, snapshots=snapshots,
        final_curve=final_y.detach().numpy(), final_slope=final_s.detach().numpy(),
        diagnostic=diagnostic, history=history, detailed=detailed,
        max_update_error=max_update_error, max_moment_error=max_moment_error,
        seconds=time.perf_counter()-started,
    )
    trials.save_torch(result_file, result)
    print(f"DONE {key}: folded={diagnostic['folded']}, R2={diagnostic['r2']:.4f}, seconds={result['seconds']:.1f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((ORIGINAL / "settings.json").read_text())
    c = trials.behavior.Config(**original["config"])
    metadata = dict(
        config=asdict(c), seeds=original["seeds"], gains=list(GAINS),
        diagnostic_seeds=list(DIAGNOSTIC_SEEDS), snapshot_steps=list(SNAPSHOT_STEPS),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((HERE/"PROTOCOL.md").read_bytes()).hexdigest(),
        exploratory=True,
    )
    path = OUT / "settings.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise RuntimeError("Mechanism settings changed; use a new output directory")
    trials.atomic_json(path, metadata)
    # Reproduction is a gate, not a result selected after trying several seeds.
    first = fit(c, 131, 4, "ordinary", OUT)
    previous = trials.load(ORIGINAL/"encoders"/"icm-131.pt")["encoder"]
    errors = [float((p-previous[k]).abs().max()) for k,p in first["encoder"].items()]
    maximum = max(errors)
    trials.atomic_json(OUT/"preflight.json", dict(
        seed=131, gain=4, max_parameter_difference=maximum,
        update_reconstruction_max=first["max_update_error"],
        moment_reconstruction_max=first["max_moment_error"],
        passed=maximum < 1e-6 and first["max_update_error"] < 1e-6,
    ))
    if maximum >= 1e-6 or first["max_update_error"] >= 1e-6:
        raise RuntimeError("Instrumented baseline did not reproduce; investigate before running controls")
    if args.preflight_only:
        return
    for seed in original["seeds"]:
        for gain in GAINS:
            if seed in DIAGNOSTIC_SEEDS:
                fit(c, seed, gain, "ordinary", OUT)
            fit(c, seed, gain, "balanced", OUT)
        trials.atomic_json(OUT/f"completed-{seed}.json", dict(seed=seed, completed=True))
    print("All mechanism controls complete", flush=True)


if __name__ == "__main__":
    main()

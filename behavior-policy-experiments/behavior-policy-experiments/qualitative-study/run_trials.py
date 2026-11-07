"""Paired fresh trials, binning controls and controlled planning-coverage removals."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
sys.path.insert(0, str(ROOT.parent / "planning"))
import evaluate_planning as planning
import numpy as np
import torch

study, behavior = planning.study, planning.behavior
ARMS = ("icm", "ou", "icm_to_ou", "balanced_log", "balanced_size")
SEEDS = tuple(range(130, 140))
EDGES = np.geomspace(1.0, 16.0, 33)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def distances(x):
    """Analysis-only geometry readout; never passed to a learned model."""
    return 1 / x[:, 1].clamp_min(1e-12)


def analysis_bins(x):
    distance = distances(x)
    ix = (torch.log(distance) / np.log(16) * 32).long().clamp(0, 31)
    valid = (distance >= 1 - 1e-6) & (distance <= 16 + 1e-6)
    return ix, valid


def bin_counts(x):
    ix, valid = analysis_bins(x)
    return torch.bincount(ix[valid], minlength=32).numpy()


def coverage(x):
    counts = bin_counts(x)
    probability = counts / max(counts.sum(), 1)
    positive = probability > 0
    return dict(
        counts=counts.tolist(),
        occupied=int(positive.sum()),
        entropy=float(-(probability[positive] * np.log(probability[positive])).sum()),
        outside_fraction=float(1 - counts.sum() / len(x)),
    )


def replay_pairs(data, seed, mode):
    height = data["x"][:, 1].clamp_min(1e-12)
    fraction = (
        (height.log() + np.log(16)) / np.log(16)
        if mode == "log"
        else (height - 1 / 16) / (1 - 1 / 16)
    )
    bins = (fraction * 32).long().clamp(0, 31)
    counts = torch.bincount(bins, minlength=32).clamp_min(1)
    generator = torch.Generator().manual_seed(900000 + seed)
    ix = torch.multinomial(
        1 / counts[bins].float(), len(bins), replacement=True, generator=generator
    )
    return {key: data[key][ix] for key in ("x", "xp")}


def source_data(c, seed, out):
    path = out / "sources" / f"icm-{seed}.pt"
    if path.exists():
        saved = load(path)
        if saved["config"] != asdict(c):
            raise RuntimeError("Saved ICM source configuration differs")
        return saved["data"]
    print(f"seed {seed}: train ICM actor", flush=True)
    policy, icm, history, movement = behavior.train_policy(c, seed)
    data = behavior.collect(c, "icm", c.train_episodes, 10000 + seed, policy)
    save_torch(
        path,
        dict(
            config=asdict(c),
            policy=policy.state_dict(),
            icm=icm.state_dict(),
            history=history,
            parameter_change=movement,
            data=data,
        ),
    )
    return data


def train_encoder(c, seed, arm, data, ou, out):
    path = out / "encoders" / f"{arm}-{seed}.pt"
    if path.exists():
        saved = load(path)
        if saved["config"] != asdict(c):
            raise RuntimeError("Saved encoder configuration differs")
        model = behavior.Encoder()
        model.load_state_dict(saved["encoder"])
        return model.eval(), saved
    model, plan = study.make_plan(c, seed, seed, "icm")
    if arm == "icm":
        plan["masks"] = torch.zeros_like(plan["masks"])
    elif arm == "ou":
        plan["masks"] = torch.ones_like(plan["masks"])
    regplan = copy.copy(plan)
    regplan["indices"] = torch.randint(
        len(data["x"]),
        plan["indices"].shape,
        generator=torch.Generator().manual_seed(1900000 + seed),
    )
    regdata = (
        replay_pairs(data, seed, "log" if arm == "balanced_log" else "size")
        if arm.startswith("balanced")
        else data
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer = behavior.SIGReg1D()
    counts, regcounts, history, start = (
        np.zeros(32, dtype=np.int64),
        np.zeros(32, dtype=np.int64),
        [],
        0,
    )
    partial = out / "partials" / path.name
    if partial.exists():
        old = load(partial)
        if old["config"] != asdict(c) or old["arm"] != arm or old["seed"] != seed:
            raise RuntimeError("Partial encoder configuration differs")
        model.load_state_dict(old["encoder"])
        optimizer.load_state_dict(old["optimizer"])
        torch.set_rng_state(old["rng"])
        counts, regcounts, history, start = (
            old["counts"],
            old["regcounts"],
            old["history"],
            old["step"],
        )
    started = time.perf_counter()
    for step in range(start, c.jepa_steps):
        optimizer.param_groups[0]["lr"] = study.controls.learning_rate(
            step, c.jepa_steps
        )
        x, xp = study.controls.batch_at(data, ou, plan, step)
        y = model(torch.cat([x, xp]))
        alignment = (y[: len(x)] - y[len(x) :]).square().mean()
        rx, rxp = study.controls.batch_at(regdata, ou, regplan, step)
        penalty = regularizer(model(torch.cat([rx, rxp])))
        loss = (1 - c.sigreg_weight) * alignment + c.sigreg_weight * penalty
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite encoder loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counts += bin_counts(torch.cat([x, xp]))
        regcounts += bin_counts(torch.cat([rx, rxp]))
        if (step + 1) % 500 == 0 or step == c.jepa_steps - 1:
            history.append(
                dict(
                    step=step + 1,
                    alignment=float(alignment.detach()),
                    sigreg=float(penalty.detach()),
                )
            )
            save_torch(
                partial,
                dict(
                    encoder=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(),
                    counts=counts,
                    regcounts=regcounts,
                    history=history,
                    step=step + 1,
                    config=asdict(c),
                    arm=arm,
                    seed=seed,
                ),
            )
    saved = dict(
        encoder=model.state_dict(),
        config=asdict(c),
        arm=arm,
        seed=seed,
        history=history,
        counts=counts,
        regcounts=regcounts,
        steps=c.jepa_steps,
        wall_seconds=time.perf_counter() - started,
        alpha_mean=float(plan["masks"].float().mean()),
    )
    save_torch(path, saved)
    print(
        f"seed {seed}: completed encoder {arm} ({saved['wall_seconds']:.1f}s)",
        flush=True,
    )
    return model.eval(), saved


@torch.no_grad()
def representation(encoder, c, seed):
    generator = torch.Generator().manual_seed(1800000 + seed)
    cal_d = torch.exp(torch.rand(4096, generator=generator) * np.log(16))
    d = torch.exp(torch.linspace(0, np.log(16), 1024))
    cal, raw = encoder(behavior.sensor(cal_d, c)), encoder(behavior.sensor(d, c))
    mean, scale = float(cal.mean()), max(float(cal.std(unbiased=False)), 1e-8)
    sign = 1 if float(torch.cov(torch.stack([cal, cal_d.log()]))[0, 1]) >= 0 else -1
    y = sign * (raw.numpy() - mean) / scale
    fit = np.linalg.lstsq(
        np.column_stack([cal.numpy(), np.ones(len(cal))]),
        cal_d.log().numpy(),
        rcond=None,
    )[0]
    estimate = np.column_stack([raw.numpy(), np.ones(len(raw))]) @ fit
    truth = d.log().numpy()
    r2 = 1 - np.mean((estimate - truth) ** 2) / np.var(truth)
    reversals = float(np.mean(np.diff(y) < -1e-5))
    slope = np.abs(np.diff(y) / np.diff(d.numpy()))
    return dict(
        distance=d.numpy(),
        embedding=y,
        r2=float(r2),
        reversal_fraction=reversals,
        folded=bool(reversals > 0.01),
        raw_variance=float(cal.var(unbiased=False)),
        sensitivity_p05=float(np.quantile(slope, 0.05)),
        sensitivity=slope,
    )


@torch.no_grad()
def plan_goals(encoder, predictor, mean, scale, c, seed, cases):
    start, goal = planning.goals(c, seed, cases)
    distance = start.clone()
    goal_y = planning.encode(encoder, behavior.sensor(goal, c), mean, scale)
    bank = planning.candidate_bank(seed)
    path, actions = [distance.numpy().copy()], []
    for _ in range(12):
        y = planning.encode(encoder, behavior.sensor(distance, c), mean, scale)
        action = planning.choose_actions(predictor, y, goal_y, bank)
        distance = behavior.move(distance, action, c)
        actions.append(action.numpy().copy())
        path.append(distance.numpy().copy())
    oracle = start.clone()
    oracle_path = [oracle.numpy().copy()]
    for _ in range(12):
        action = torch.where(
            goal - oracle > c.move / 2, 2, torch.where(oracle - goal > c.move / 2, 0, 1)
        ).long()
        oracle = behavior.move(oracle, action, c)
        oracle_path.append(oracle.numpy().copy())
    assert float((oracle - goal).abs().max()) < 1e-4
    error = (distance - goal).abs().numpy()
    return dict(
        mean_error=float(error.mean()),
        success=float(np.mean(error <= c.move + 1e-5)),
        error=error,
        start=start.numpy(),
        goal=goal.numpy(),
        path=np.stack(path),
        actions=np.stack(actions),
        oracle_path=np.stack(oracle_path),
    )


def predictor_result(
    encoder, data, common, c, seed, key, out, updates, screen_cases, test_cases
):
    checkpoint = out / "predictors" / (key + ".pt")
    resultpath = out / "planning" / (key + ".pt")
    if resultpath.exists():
        return load(resultpath)
    if checkpoint.exists():
        saved = load(checkpoint)
        predictor = planning.Predictor()
        predictor.load_state_dict(saved["predictor"])
        predictor.eval()
        mean, scale = saved["mean"], saved["scale"]
    else:
        predictor, mean, scale = planning.train_predictor(
            encoder, data, common, seed, updates
        )
        save_torch(
            checkpoint, dict(predictor=predictor.state_dict(), mean=mean, scale=scale)
        )
    result = dict(
        screen=plan_goals(
            encoder, predictor, mean, scale, c, 9000 + seed, screen_cases
        ),
        test=plan_goals(encoder, predictor, mean, scale, c, 10000 + seed, test_cases),
        data_coverage=coverage(data["x"]),
    )
    save_torch(resultpath, result)
    return result


def resampled_region(data, seed, region):
    ix, _ = analysis_bins(data["x"])
    keep = (
        torch.arange(len(ix)) if region is None else torch.where(ix // 4 != region)[0]
    )
    if len(keep) == 0:
        raise RuntimeError("Region removal left no physical predictor data")
    uniform = torch.rand(
        len(ix), generator=torch.Generator().manual_seed(2100000 + seed)
    )
    selected = keep[(uniform * len(keep)).long()]
    return {k: data[k][selected] for k in ("x", "xp", "action")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    c = behavior.Config()
    updates, screen_cases, test_cases = 1000, 24, 64
    if args.quick:
        c.policy_updates, c.jepa_steps, c.train_episodes, c.eval_episodes = 2, 20, 8, 4
        c.nenv, c.rollout, c.horizon, c.burnin, c.batch = 4, 4, 8, 4, 16
        updates, screen_cases, test_cases = 10, 4, 8
    settings = dict(
        config=asdict(c),
        seeds=args.seeds,
        arms=list(ARMS),
        predictor_updates=updates,
        screen_cases=screen_cases,
        test_cases=test_cases,
        quick=args.quick,
        numpy=np.__version__,
        torch=str(torch.__version__),
        protocol_sha256=hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest(),
    )
    settings["source_sha256"] = {
        str(Path(module.__file__).resolve()): hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in (sys.modules[__name__], planning, study, behavior, study.controls)
    }
    settingspath = out / "settings.json"
    if settingspath.exists() and json.loads(settingspath.read_text()) != settings:
        raise RuntimeError("Study settings differ from existing output")
    atomic_json(settingspath, settings)
    for seed in args.seeds:
        print(f"BEGIN seed {seed}", flush=True)
        data = source_data(c, seed, out)
        ou = study.controls.ou_data(c, seed)
        common = behavior.collect(c, "random", c.train_episodes, 510000 + seed)
        save_torch(out / "sources" / f"physical-common-{seed}.pt", common)
        models = {}
        for arm in ARMS:
            encoder, saved = train_encoder(c, seed, arm, data, ou, out)
            models[arm] = encoder
            diagnostic = representation(encoder, c, seed)
            save_torch(
                out / "curves" / f"{arm}-{seed}.pt",
                dict(
                    **diagnostic,
                    source_coverage=coverage(ou["x"] if arm == "ou" else data["x"]),
                    actual_training_counts=saved["counts"],
                    actual_regularizer_counts=saved["regcounts"],
                ),
            )
            for kind, pairs in [("shared", common)] + (
                [] if arm == "ou" else [("native", data)]
            ):
                result = predictor_result(
                    encoder,
                    pairs,
                    common,
                    c,
                    seed,
                    f"{arm}-{kind}-{seed}",
                    out,
                    updates,
                    screen_cases,
                    test_cases,
                )
                print(
                    f"seed {seed}: {arm}/{kind} test goal error {result['test']['mean_error']:.4f}",
                    flush=True,
                )
        # Only the OU encoder varies across these ten paired coverage experiments.
        for region in [None, *range(8)]:
            pairs = resampled_region(common, seed, region)
            tag = "all" if region is None else str(region)
            result = predictor_result(
                models["ou"],
                pairs,
                common,
                c,
                seed,
                f"remove-{tag}-{seed}",
                out,
                updates,
                screen_cases,
                test_cases,
            )
            print(
                f"seed {seed}: coverage removal {tag}, test error {result['test']['mean_error']:.4f}",
                flush=True,
            )
        atomic_json(
            out / f"completed-{seed}.json",
            dict(seed=seed, completed=True, quick=args.quick),
        )
        print(f"COMPLETED seed {seed}", flush=True)
    print("All requested trials completed", flush=True)


if __name__ == "__main__":
    main()

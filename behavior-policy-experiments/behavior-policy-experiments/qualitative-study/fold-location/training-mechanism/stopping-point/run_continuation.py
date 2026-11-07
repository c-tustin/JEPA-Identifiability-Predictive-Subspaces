"""Continue the saved ICM networks to test whether their turns really settled."""

import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_mechanism import OUT as PARENT_OUT, ORIGINAL, ScaledEncoder, trials

OUT = HERE / "output"
SEEDS = (131, 130, 136)
STEPS = 8000
ARMS = {"ordinary_lr1e-5": (1e-5, False),
        "ordinary_lr1e-4": (1e-4, False),
        "balanced_lr1e-4": (1e-4, True)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, config, source, reg_source, seed, step):
    """Measure geometry and loss; fixed batches do not affect training RNGs."""
    q = torch.linspace(0, np.log(16), 1025, requires_grad=True)
    y = model(trials.behavior.sensor(q.exp(), config))
    slope = torch.autograd.grad(y.sum(), q)[0]
    diagnostic = trials.representation(model, config, seed)
    regularizer = trials.behavior.SIGReg1D()
    generator = torch.Generator().manual_seed(2300000 + seed)
    with torch.no_grad():
        alignment = (model(source["x"]) - model(source["xp"])).square().mean()
        penalties = []
        for _ in range(16):
            ix = torch.randint(len(reg_source["x"]), (512,), generator=generator)
            penalties.append(float(regularizer(model(torch.cat([reg_source["x"][ix], reg_source["xp"][ix]])))))
    return dict(additional_updates=step, total_updates=4000+step,
                curve=y.detach().numpy(), slope=slope.detach().numpy(),
                folded=diagnostic["folded"], r2=diagnostic["r2"],
                reversal_fraction=diagnostic["reversal_fraction"],
                alignment=float(alignment), mean_sigreg=float(np.mean(penalties)),
                sigreg_batch_sd=float(np.std(penalties, ddof=1)),
                mean_objective=.9*float(alignment)+.1*float(np.mean(penalties)))


def start_state(seed):
    path = OUT / "starts" / f"seed-{seed}.pt"
    if path.exists():
        return trials.load(path)
    old = trials.load(PARENT_OUT / "partials" / f"ordinary-g4-{seed}.pt")
    final = trials.load(PARENT_OUT / "runs" / f"ordinary-g4-{seed}.pt")
    assert old["step"] == 4000
    assert all(torch.equal(v, final["encoder"][k]) for k, v in old["encoder"].items())
    state = dict(seed=seed, config=old["config"], step=4000,
                 encoder=old["encoder"], optimizer=old["optimizer"],
                 original_checkpoint_sha256=digest(PARENT_OUT / "runs" / f"ordinary-g4-{seed}.pt"))
    # Save the small optimizer state needed to reproduce these continuations;
    # the older ignored partial includes much more instrumented history.
    trials.save_torch(path, state)
    return state


def fit(seed, arm):
    target = OUT / "runs" / f"{arm}-{seed}.pt"
    if target.exists():
        return
    start = start_state(seed)
    config = trials.behavior.Config(**start["config"])
    source = trials.load(ORIGINAL / "sources" / f"icm-{seed}.pt")["data"]
    lr, balanced = ARMS[arm]
    reg_source = trials.replay_pairs(source, seed, "log") if balanced else source
    model = ScaledEncoder(4)
    model.load_state_dict(start["encoder"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(start["optimizer"])
    regularizer = trials.behavior.SIGReg1D()
    alignment_rng = torch.Generator().manual_seed(2100000 + seed)
    reg_rng = torch.Generator().manual_seed(2200000 + seed)
    frames = [evaluate(model, config, source, reg_source, seed, 0)]
    saved_states = {0: copy.deepcopy(model.state_dict())}
    elapsed, begin = 0., 0
    partial = OUT / "partials" / f"{arm}-{seed}.pt"
    if partial.exists():
        previous = trials.load(partial)
        model.load_state_dict(previous["encoder"])
        optimizer.load_state_dict(previous["optimizer"])
        alignment_rng.set_state(previous["alignment_rng"])
        reg_rng.set_state(previous["reg_rng"])
        frames, saved_states = previous["frames"], previous["saved_states"]
        begin, elapsed = previous["additional_updates"], previous["seconds"]
    optimizer.param_groups[0]["lr"] = lr
    started = time.perf_counter()
    for step in range(begin, STEPS):
        ix = torch.randint(len(source["x"]), (config.batch,), generator=alignment_rng)
        ri = torch.randint(len(reg_source["x"]), (config.batch,), generator=reg_rng)
        y = model(torch.cat([source["x"][ix], source["xp"][ix]]))
        alignment = (y[:config.batch]-y[config.batch:]).square().mean()
        penalty = regularizer(model(torch.cat([reg_source["x"][ri], reg_source["xp"][ri]])))
        loss = .9*alignment + .1*penalty
        assert torch.isfinite(loss), (seed, arm, step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 500 == 0:
            frames.append(evaluate(model, config, source, reg_source, seed, step+1))
            saved_states[step+1] = copy.deepcopy(model.state_dict())
            result = dict(seed=seed, arm=arm, lr=lr, balanced=balanced, config=start["config"],
                          additional_updates=step+1, encoder=model.state_dict(),
                          optimizer=optimizer.state_dict(), frames=frames, saved_states=saved_states,
                          alignment_rng=alignment_rng.get_state(), reg_rng=reg_rng.get_state(),
                          seconds=elapsed+time.perf_counter()-started)
            trials.save_torch(partial, result)
    trials.save_torch(target, result)
    print(f"DONE {arm} seed {seed}: R2={frames[-1]['r2']:.4f}, folded={frames[-1]['folded']}, seconds={result['seconds']:.1f}", flush=True)


def check_zero_lr(seed):
    start = start_state(seed)
    model = ScaledEncoder(4)
    model.load_state_dict(start["encoder"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(start["optimizer"])
    optimizer.param_groups[0]["lr"] = 0.
    source = trials.load(ORIGINAL / "sources" / f"icm-{seed}.pt")["data"]
    x, xp = source["x"][:512], source["xp"][:512]
    y = model(torch.cat([x, xp]))
    loss = .9*(y[:512]-y[512:]).square().mean()+.1*trials.behavior.SIGReg1D()(y)
    loss.backward()
    optimizer.step()
    maximum = max(float((v-start["encoder"][k]).abs().max()) for k, v in model.state_dict().items())
    assert maximum == 0
    return dict(seed=seed, max_parameter_change=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = dict(seeds=list(SEEDS), additional_updates=STEPS, arms=ARMS,
                    code_sha256=digest(Path(__file__)), protocol_sha256=digest(HERE/"PROTOCOL.md"),
                    sources={str(s): digest(ORIGINAL/"sources"/f"icm-{s}.pt") for s in SEEDS},
                    original_checkpoints={str(s): digest(PARENT_OUT/"runs"/f"ordinary-g4-{s}.pt") for s in SEEDS})
    import json
    # JSON turns tuples into lists; compare through a canonical round trip.
    metadata = json.loads(json.dumps(metadata))
    settings = OUT / "settings.json"
    if settings.exists():
        assert json.loads(settings.read_text()) == metadata, "Settings or source hashes changed"
    else:
        trials.atomic_json(settings, metadata)
    seeds = (args.seed,) if args.seed is not None else SEEDS
    for seed in seeds:
        trials.atomic_json(OUT/f"zero-lr-{seed}.json", check_zero_lr(seed))
        for arm in ARMS:
            fit(seed, arm)


if __name__ == "__main__":
    main()

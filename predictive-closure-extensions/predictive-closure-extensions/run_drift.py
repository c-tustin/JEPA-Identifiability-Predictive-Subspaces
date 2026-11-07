"""Independent-noise reset regression and calibrated-observation planning pilot."""

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from closure_controls import observe, calibrated_coordinate

ROOT = Path(__file__).parent
INTERVALS = np.array([0.02, 0.05, 0.1, 0.2])


def basis(d):
    d = np.asarray(d)
    return np.stack([np.ones_like(d), np.sin(d), np.cos(d), d / 16], -1)


def features(d, a, delta, multi):
    b = basis(d)
    main = np.concatenate([b, np.asarray(a)[..., None]], -1)
    if not multi:
        return main
    return np.concatenate(
        [main, delta[..., None] * b, delta[..., None] * a[..., None] * b], -1
    )


def simulate(d, a, delta, rng, noisy=True, steps=64, coef=None):
    x = np.array(d, dtype=float, copy=True)
    dt = np.asarray(delta) / steps
    for _ in range(steps):
        drift = (
            0.3 * np.sin(x) + 0.25 * a
            if coef is None
            else basis(x) @ coef[:4] + coef[4] * a
        )
        diffusion = 0.05 * (1 + 0.1 * np.cos(x))
        noise = rng.normal(size=x.shape) if noisy else 0
        x = x + dt * drift + np.sqrt(dt) * diffusion * noise
    return x


def fit(seed, multi, weighted=False):
    rng = np.random.default_rng(seed)
    d = rng.uniform(2, 15, 8192)
    a = rng.choice([-1.0, 1.0], len(d))
    interval = np.resize(INTERVALS, len(d)) if multi else np.full(len(d), 0.2)
    # Recover from the triangle; this known calibration is a stated privilege.
    d = 4 * np.exp(0.7 * calibrated_coordinate(observe(np.log(d / 4) / 0.7)))
    successor = simulate(d, a, interval, rng)
    design = features(d, a, interval, multi)
    target = (successor - d) / interval
    weights = np.sqrt(interval) / (1 + 0.1 * np.cos(d)) if weighted else np.ones_like(d)
    coef = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)[0]
    residual = successor - d - interval * (design @ coef)
    # Estimate one scalar diffusion amplitude; its state shape is assumed known.
    amplitude = np.sqrt(np.mean(residual**2 / (interval * (1 + 0.1 * np.cos(d)) ** 2)))
    return coef, float(amplitude)


def plan(d, goal, coef, amplitude, penalty, bank, seed):
    rng = np.random.default_rng(seed)
    # Same noise for every candidate; independent draws across rollout particles.
    x = np.full((len(bank), 16), d)
    for t in range(4):
        a = bank[:, t, None]
        for _ in range(16):
            noise = rng.normal(size=(1, 16))
            x = (
                x
                + (basis(x) @ coef[:4] + coef[4] * a) / 16
                + amplitude * (1 + 0.1 * np.cos(x)) * noise / 4
            )
    costs = (x - goal) ** 2
    score = costs.mean(-1) + penalty * costs.std(-1)
    return bank[np.argmin(score), 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted-followup", action="store_true")
    args = parser.parse_args()
    seeds = list(range(301, 306)) if args.weighted_followup else list(range(201, 206))
    kinds = (
        ["single", "multi", "multi_wls"]
        if args.weighted_followup
        else ["single", "multi"]
    )
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    bank = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=4)))
    records = {
        "settings": dict(
            seeds=seeds,
            weighted_followup=args.weighted_followup,
            endpoints=8192,
            substeps=64,
            intervals=INTERVALS.tolist(),
            goals=16,
            decisions=8,
            horizon=4,
            particles=16,
        ),
        "fits": [],
        "planning": [],
    }
    for seed in seeds:
        rng = np.random.default_rng(seed + 100000)
        grid = np.linspace(2, 15, 1001)
        fitted = {}
        for kind in kinds:
            coef, amp = fit(seed, kind != "single", weighted=kind == "multi_wls")
            fitted[kind] = (coef, amp)
            expected = basis(grid) @ coef[:4]
            pred = simulate(grid, np.ones_like(grid), 1.0, rng, noisy=False, coef=coef)
            truth = simulate(grid, np.ones_like(grid), 1.0, rng, noisy=False)
            records["fits"].append(
                dict(
                    seed=seed,
                    kind=kind,
                    coefficients=coef.tolist(),
                    diffusion_amplitude=amp,
                    drift_rmse=float(
                        np.sqrt(np.mean((expected - 0.3 * np.sin(grid)) ** 2))
                    ),
                    action_coefficient=float(coef[4]),
                    transition_rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
                )
            )
        starts = []
        while len(starts) < 16:
            d = rng.uniform(2, 15)
            if abs(0.3 * np.sin(d)) <= 0.1:
                starts.append(d)
        for index, start in enumerate(starts):
            a = -np.sign(np.sin(start))
            goal = float(simulate(np.array(start), a, 2.0, rng, noisy=False, steps=128))
            for kind in kinds + ["stay"]:
                for penalty in ([0.0, 1.0] if kind != "stay" else [0.0]):
                    state = start
                    closest = abs(state - goal)
                    path = [state]
                    actions = []
                    # Same real execution noise across every controller.
                    execution_rng = np.random.default_rng(seed * 10000 + index)
                    for t in range(8):
                        action = (
                            0.0
                            if kind == "stay"
                            else plan(
                                float(
                                    4
                                    * np.exp(
                                        0.7
                                        * calibrated_coordinate(
                                            observe(np.log(state / 4) / 0.7)
                                        )[0]
                                    )
                                ),
                                float(
                                    4
                                    * np.exp(
                                        0.7
                                        * calibrated_coordinate(
                                            observe(np.log(goal / 4) / 0.7)
                                        )[0]
                                    )
                                ),
                                *fitted[kind],
                                penalty,
                                bank,
                                seed * 100000 + index * 100 + t
                            )
                        )
                        state = float(
                            simulate(np.array(state), action, 1.0, execution_rng)
                        )
                        closest = min(closest, abs(state - goal))
                        path.append(state)
                        actions.append(float(action))
                    records["planning"].append(
                        dict(
                            seed=seed,
                            goal_index=index,
                            kind=kind,
                            penalty=penalty,
                            start=start,
                            goal=goal,
                            closest_error=closest,
                            final_error=abs(state - goal),
                            hit=bool(closest <= 0.1),
                            path=path,
                            actions=actions,
                        )
                    )
        print("drift/planning seed", seed, flush=True)
    (
        out / ("drift-weighted.json" if args.weighted_followup else "drift.json")
    ).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Compare ordinary one-distance trajectories under nonlinear versus zero drift.

The flexible passive encoder sees observation pairs only. No paired interventions,
distance targets, action predictor, or monotonicity constraint enter fitting.
See TRAJECTORY_PROTOCOL.md for the fixed comparison and its limits.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from check_actuation import behavior
from policies import size_bin
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class Settings:
    episodes: int = 256
    horizon: int = 64
    burnin: int = 32
    substeps: int = 50
    dt: float = 0.02
    updates: int = 4000
    batch: int = 512


def reflect(distance):
    """Reflect any overshoot into [1, 16], including multiple crossings."""
    remainder = torch.remainder(distance - 1, 30)
    return 1 + torch.minimum(remainder, 30 - remainder)


def advance(distance, action, noise, nonlinear, dt):
    drift = 0.3 * torch.sin(distance) if nonlinear else torch.zeros_like(distance)
    diffusion = 0.05 * (1 + 0.1 * torch.cos(distance))
    return reflect(
        distance + (drift + 0.25 * action) * dt + diffusion * math.sqrt(dt) * noise
    )


def collect(settings, seed, policy, nonlinear):
    """Return observation pairs/actions; simulator coordinates are discarded."""
    actions_rng = torch.Generator().manual_seed(2300000 + seed)
    world_rng = torch.Generator().manual_seed(2400000 + seed)
    distance = 1 + 15 * torch.rand(settings.episodes, generator=world_rng)
    score = torch.randn(settings.episodes, generator=actions_rng)
    counts = torch.zeros(32, 3)
    rows = []
    for step in range(settings.burnin + settings.horizon):
        x = behavior.sensor(distance, behavior.Config())
        if policy == "random":
            action = torch.randint(3, (settings.episodes,), generator=actions_rng)
        elif policy == "persistent":
            score = 0.9 * score + math.sqrt(1 - 0.9**2) * torch.randn(
                settings.episodes, generator=actions_rng
            )
            action = (score > -0.43072729929545756).long() + (
                score > 0.43072729929545756
            ).long()
        elif policy == "action_balanced":
            bins = size_bin(x)
            probabilities = 0.9 * torch.softmax(-counts[bins], -1) + 0.1 / 3
            action = torch.multinomial(
                probabilities, 1, generator=actions_rng
            ).flatten()
            counts += torch.bincount(bins * 3 + action, minlength=96).reshape(32, 3)
        else:
            raise ValueError(policy)
        for _ in range(settings.substeps):
            noise = torch.randn(settings.episodes, generator=world_rng)
            distance = advance(distance, action - 1, noise, nonlinear, settings.dt)
        if step >= settings.burnin:
            rows.append((x, behavior.sensor(distance, behavior.Config()), action))
    return {
        key: torch.cat([row[i] for row in rows])
        for i, key in enumerate(["x", "xp", "action"])
    }


def sampling_weights(data):
    bins = size_bin(data["x"])
    counts = torch.bincount(bins, minlength=32).clamp_min(1)
    return 1 / counts[bins].float()


def coverage(data):
    bins = size_bin(data["x"])
    counts = torch.bincount(bins, minlength=32)
    joint = (
        torch.bincount(bins * 3 + data["action"], minlength=96).reshape(32, 3).float()
    )
    probabilities = joint / counts.clamp_min(1)[:, None]
    mean = probabilities[:, 2] - probabilities[:, 0]
    variance = probabilities[:, 0] + probabilities[:, 2] - mean.square()
    return {
        "visited_source_bins": int((counts > 0).sum()),
        "source_bin_counts": counts.tolist(),
        "min_visited_bin_action_variance": float(variance[counts > 0].min()),
    }


def fit(data, settings, seed, sampler):
    torch.manual_seed(2500000 + seed)
    model = behavior.Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer = behavior.SIGReg1D()
    alignment_rng = torch.Generator().manual_seed(2600000 + seed)
    regularizer_rng = torch.Generator().manual_seed(2700000 + seed)
    weights = (
        sampling_weights(data) if sampler == "balanced" else torch.ones(len(data["x"]))
    )
    # Resampling uses the same uniform random stream in both sampler conditions.
    reg_indices = torch.multinomial(
        weights,
        settings.updates * settings.batch,
        replacement=True,
        generator=regularizer_rng,
    ).reshape(settings.updates, settings.batch)
    for step in range(settings.updates):
        fraction = max(
            0, (step - settings.updates / 2) / max(1, settings.updates / 2 - 1)
        )
        optimizer.param_groups[0]["lr"] = (
            0.001 * 0.5 * (1 + math.cos(math.pi * fraction))
        )
        indices = torch.randint(
            len(weights), (settings.batch,), generator=alignment_rng
        )
        y = model(torch.cat([data["x"][indices], data["xp"][indices]]))
        alignment = (y[: settings.batch] - y[settings.batch :]).square().mean()
        indices = reg_indices[step]
        penalty = regularizer(
            model(torch.cat([data["x"][indices], data["xp"][indices]]))
        )
        loss = 0.9 * alignment + 0.1 * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def largest_reversal_fraction(oriented):
    """Largest drop while distance increases, divided by total output range."""
    output_range = float(np.ptp(oriented))
    if output_range == 0:
        return 0.0
    drop = np.maximum.accumulate(oriented) - oriented
    return float(drop.max()) / output_range


@torch.no_grad()
def evaluate(model, seed):
    generator = torch.Generator().manual_seed(2800000 + seed)
    calibration = 1.2 + 14.6 * torch.rand(4096, generator=generator)
    grid = torch.linspace(1.2, 15.8, 1024)
    yc = model(behavior.sensor(calibration, behavior.Config())).numpy()
    y = model(behavior.sensor(grid, behavior.Config())).numpy()
    metrics = {}
    for name, transform in [("distance", lambda x: x), ("log_distance", np.log)]:
        metrics[name + "_r2"] = behavior.affine_score(
            yc, transform(calibration.numpy()), y, transform(grid.numpy())
        )
    sign = 1 if np.cov(grid.numpy(), y)[0, 1] >= 0 else -1
    normalized = sign * (y - y.mean()) / max(float(y.std()), 1e-8)
    metrics["reversed_grid_fraction"] = float(np.mean(np.diff(normalized) < -1e-6))
    metrics["largest_reversal_fraction"] = largest_reversal_fraction(normalized)
    metrics["grid_std"] = float(y.std())
    return metrics, {"distance": grid.numpy(), "embedding": y}


def report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Flexible JEPA on nonlinear trajectories",
        "",
        "Same triangle and one distance. Raw alignment in every cell; only SIGReg sampling changes. Three fixed seeds per cell. Details: [protocol](../TRAJECTORY_PROTOCOL.md).",
        "",
        "| Drift | Policy | SIGReg data | Runs | Distance $R^2$ | Log-distance $R^2$ | Runs with reversals | Worst reversal (% of output range) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nonlinear in [False, True]:
        for policy in ["random", "persistent", "action_balanced"]:
            for sampler in ["raw", "balanced"]:
                rows = [
                    r
                    for r in result["runs"]
                    if (r["nonlinear"], r["policy"], r["sampler"])
                    == (nonlinear, policy, sampler)
                ]
                if not rows:
                    continue
                scores = [
                    np.mean([r["metrics"][key] for r in rows])
                    for key in ["distance_r2", "log_distance_r2"]
                ]
                folds = sum(r["metrics"]["reversed_grid_fraction"] > 0 for r in rows)
                worst = 100 * max(
                    r["metrics"]["largest_reversal_fraction"] for r in rows
                )
                lines.append(
                    f"| {'Nonlinear' if nonlinear else 'Zero'} | {policy} | {sampler} | {len(rows)} | {scores[0]:.4f} | {scores[1]:.4f} | {folds} | {worst:.4f} |"
                )
    if result["quick"]:
        lines.insert(2, "**Execution check only; not research results.**")
    lines += [
        "",
        "Means use identical interior evaluation grids; per-seed scores, coverage, and checkpoints are saved. Worst reversal measures the largest cumulative drop along increasing distance after orienting the embedding, relative to its full grid range. This supplementary size diagnostic was added after observing shallow reversals; the original reversal counts are retained. Neither affine target is guaranteed by this passive objective. No action-conditioned predictor or planning evaluation is included.",
        "",
        "![Learned coordinates](trajectories.png)",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def plot(out, result):
    fig, axes = plt.subplots(
        2, 3, figsize=(11, 6), sharex=True, sharey=True, layout="constrained"
    )
    for row in result["runs"]:
        ax = axes[
            int(row["nonlinear"]),
            ["random", "persistent", "action_balanced"].index(row["policy"]),
        ]
        data = np.load(out / row["checkpoint"].replace(".pt", ".npz"))
        y, d = data["embedding"], data["distance"]
        sign = 1 if np.cov(d, y)[0, 1] >= 0 else -1
        ax.plot(
            d,
            sign * (y - y.mean()) / max(float(y.std()), 1e-8),
            color={"raw": "#c45c32", "balanced": "#287bb5"}[row["sampler"]],
            alpha=0.6,
            linewidth=1,
        )
    for i in range(2):
        for j, policy in enumerate(["Random", "Persistent", "Action-balanced"]):
            axes[i, j].set_title(f"{'Nonlinear' if i else 'Zero'} drift / {policy}")
            axes[i, j].set_xlabel("Physical distance (evaluation only)")
        axes[i, 0].set_ylabel("Oriented, standardized embedding")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color="#c45c32", label="Raw SIGReg"),
            Line2D([0], [0], color="#287bb5", label="Balanced SIGReg"),
        ],
        loc="outside upper center",
        ncol=2,
    )
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output-trajectories"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = Settings()
    if args.quick:
        settings = Settings(
            episodes=8, horizon=8, burnin=2, substeps=2, updates=20, batch=16
        )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"settings": asdict(settings), "quick": args.quick, "runs": []}
    if args.resume or args.evaluate_only:
        result = json.loads((args.output / "metrics.json").read_text())
        if result["settings"] != asdict(settings) or result["quick"] != args.quick:
            raise ValueError("Saved configuration differs")
    elif (args.output / "metrics.json").exists():
        raise ValueError("Output already exists; use --resume or a new --output")
    for seed in [23] if args.quick else [23, 24, 25]:
        for nonlinear in [False, True]:
            for policy in ["random", "persistent", "action_balanced"]:
                data = None
                for sampler in ["raw", "balanced"]:
                    filename = f"encoder_{'nonlinear' if nonlinear else 'zero'}_{policy}_{sampler}_seed{seed}.pt"
                    found = next(
                        (r for r in result["runs"] if r["checkpoint"] == filename), None
                    )
                    if args.evaluate_only:
                        if found is None:
                            raise RuntimeError(f"Missing evaluation: {filename}")
                        model = behavior.Encoder()
                        model.load_state_dict(
                            torch.load(args.output / filename, map_location="cpu")[
                                "encoder"
                            ]
                        )
                        metrics, _ = evaluate(model, seed)
                        for key in metrics:
                            if not np.isclose(
                                metrics[key], found["metrics"][key], atol=1e-8, rtol=0
                            ):
                                raise RuntimeError(
                                    f"Evaluation changed: {filename}: {key}"
                                )
                    elif found is None:
                        print(filename, flush=True)
                        if data is None:
                            data = collect(settings, seed, policy, nonlinear)
                        model = fit(data, settings, seed, sampler)
                        metrics, arrays = evaluate(model, seed)
                        torch.save(
                            {
                                "encoder": model.state_dict(),
                                "settings": asdict(settings),
                                "seed": seed,
                                "nonlinear": nonlinear,
                                "policy": policy,
                                "sampler": sampler,
                            },
                            args.output / filename,
                        )
                        np.savez_compressed(
                            args.output / filename.replace(".pt", ".npz"), **arrays
                        )
                        result["runs"].append(
                            {
                                "seed": seed,
                                "nonlinear": nonlinear,
                                "policy": policy,
                                "sampler": sampler,
                                "checkpoint": filename,
                                "coverage": coverage(data),
                                "metrics": metrics,
                            }
                        )
                        report(args.output, result)
    if not args.evaluate_only:
        plot(args.output, result)
    print(
        f"{'Reproduced' if args.evaluate_only else 'Completed'} {len(result['runs'])} trajectory evaluations.",
        flush=True,
    )


if __name__ == "__main__":
    main()

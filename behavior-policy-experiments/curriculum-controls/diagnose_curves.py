"""Inspect saved distance curves and behavior coverage; never update a model.

This is a post-result diagnostic, not a preregistered primary outcome or a
checkpoint-selection rule. It cannot identify the causal source of a failure.
"""
import argparse
import json
from pathlib import Path

import run_controls as controls

import matplotlib.pyplot as plt
import numpy as np
import torch

behavior = controls.behavior


@torch.no_grad()
def curve(state, c):
    model = behavior.Encoder()
    model.load_state_dict(state)
    model.eval()
    z = torch.linspace(np.log(c.distance_min / 4) / c.sigma,
                       np.log(c.distance_max / 4) / c.sigma, 1024)
    y = model(behavior.observe(z, c.sigma))
    sign = 1 if torch.cov(torch.stack([z, y]))[0, 1] >= 0 else -1
    y = sign * (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-8)
    difference = torch.diff(y)
    return {"wrong_way_grid_step_fraction": float((difference < -1e-6).float().mean()),
            "total_variation": float(difference.abs().sum()),
            "endpoint_difference": float(y[-1] - y[0])}, z.numpy(), y.numpy()


def main(out):
    torch.set_num_threads(1)
    result = json.loads((out / "metrics.json").read_text())
    c = behavior.Config(**result["config"])
    diagnostics = []
    fig, axes = plt.subplots(len(result["runs"]), 3, figsize=(12, 2.6 * len(result["runs"])),
                             squeeze=False, layout="constrained")
    for row, run in enumerate(result["runs"]):
        seed = run["seed"]
        policy = behavior.Policy()
        saved = torch.load(out / f"behavior_seed{seed}.pt", map_location="cpu")
        policy.load_state_dict(saved["policy"])
        policy.eval()
        distances = torch.linspace(c.distance_min, c.distance_max, 1024)
        probs = policy(behavior.sensor(distances, c))[0].probs.mean(0).tolist()
        record = {"seed": seed, "mean_action_probabilities_on_distance_grid": probs, "sources": {}}
        for col, source in enumerate(controls.SOURCES):
            data = behavior.collect(c, source, c.train_episodes, 10000 + seed, policy)
            d = 4 * torch.exp(c.sigma * data["z"])
            inside = (d >= c.distance_min) & (d <= c.distance_max)
            counts = torch.histc(d[inside], bins=32, min=c.distance_min, max=c.distance_max)
            record["sources"][source] = {"distance_bin_coverage": float((counts > 0).float().mean()),
                                          "fraction_distances_1_to_2": float(((d >= 1) & (d <= 2)).float().mean()),
                                          "curves": {}}
            for label, filename in [("prefix", f"prefix_{source}_seed{seed}.pt"),
                                     ("carry", f"encoder_{source}_carry_seed{seed}.pt"),
                                     ("reset_both", f"encoder_{source}_reset_both_seed{seed}.pt")]:
                state = torch.load(out / filename, map_location="cpu")["encoder"]
                metrics, z, y = curve(state, c)
                record["sources"][source]["curves"][label] = metrics
                axes[row, col].plot(z, y, label=label, alpha=.85)
            axes[row, col].set(title=f"Seed {seed}: {controls.LABELS[source]}", xlabel="True log-distance z",
                               ylabel="Standardized output")
            if row == 0:
                axes[row, col].legend(fontsize=7)
        diagnostics.append(record)
    fig.suptitle("Saved distance curves: sign and scale adjusted only for display; no model retraining")
    fig.savefig(out / "distance_curves.png", dpi=140)
    plt.close(fig)
    # Choose a descriptive illustration after evaluation; keep every seed in
    # the primary report. This selection does not tune or select a model.
    worst = min(result["runs"], key=lambda r: r["conditions"]["icm_carry"]["metrics"]["common_grid_r2"])
    seed = worst["seed"]
    example, example_axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for ax, source in zip(example_axes, controls.SOURCES):
        for label, filename in [("Before final OU phase", f"prefix_{source}_seed{seed}.pt"),
                                 ("Final: keep optimizer", f"encoder_{source}_carry_seed{seed}.pt"),
                                 ("Final: reset Adam + LR", f"encoder_{source}_reset_both_seed{seed}.pt")]:
            state = torch.load(out / filename, map_location="cpu")["encoder"]
            _, z, y = curve(state, c)
            ax.plot(z, y, label=label)
        ax.set(title=controls.LABELS[source], xlabel="True log-distance z", ylabel="Standardized output")
    example_axes[-1].legend(fontsize=7)
    example.suptitle(f"Post-result illustration: seed {seed}, the lowest ICM-handover recovery score in this study")
    example.savefig(out / "failure_example.png", dpi=160)
    plt.close(example)
    (out / "curve_diagnostics.json").write_text(json.dumps({
        "note": "Post-result descriptive checks. Wrong-way steps orient output by positive covariance with z; finite-grid checks are not a global monotonicity proof.",
        "runs": diagnostics}, indent=2) + "\n")
    print("Saved distance curves and post-result coverage/monotonicity diagnostics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    main(parser.parse_args().output)

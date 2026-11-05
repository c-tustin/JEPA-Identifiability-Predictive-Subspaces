"""Plot the saved histories and independent tail interventions; no retraining."""
import argparse
import json
from pathlib import Path

import run_controls as controls

import matplotlib.pyplot as plt
import numpy as np


def plot(out):
    result = json.loads((out / "metrics.json").read_text())
    runs = result["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    colors = {"ou": "#0e7490", "random": "#7c3aed", "icm": "#c2410c"}
    for ax, metric, title in [(axes[0, 0], "common_grid_r2", "Recovery on the shared distance range"),
                              (axes[0, 1], "reference_ou_r2", "Recovery on the broader OU test")]:
        for source in controls.SOURCES:
            histories = [r["conditions"][source + "_carry"]["history"] for r in runs]
            steps = [h["updates"] for h in histories[0]]
            scores = np.array([[h[metric] for h in history] for history in histories])
            ax.plot(steps, scores.mean(0), label=controls.LABELS[source], color=colors[source])
        ax.axvline(runs[0]["plan"]["tail_start"], color="gray", ls="--", label="Final all-OU phase")
        ax.set(xlabel="Representation updates", ylabel="Mean R²", title=title)
        ax.legend(fontsize=8)
    for ax, metric, title in [(axes[1, 0], "common_grid_r2", "Final score after tail interventions: shared range"),
                              (axes[1, 1], "reference_ou_r2", "Final score after tail interventions: broader OU")]:
        for source in controls.SOURCES:
            values = [[r["conditions"][f"{source}_{tail}"]["metrics"][metric] for r in runs] for tail in controls.TAILS]
            ax.plot(range(len(controls.TAILS)), np.mean(values, axis=1), marker="o", color=colors[source], label=controls.LABELS[source])
            for index, seeds in enumerate(values):
                ax.scatter(np.full(len(seeds), index), seeds, color=colors[source], alpha=.25, s=12)
        ax.set(xticks=range(4), xticklabels=["Keep both", "Reset Adam", "Restart LR", "Reset both"], ylabel="Recovery R²", title=title)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle(("SMOKE CHECK — " if result["quick"] else "") +
                 f"Same distance-only world: {len(runs)} paired seeds; dots are individual runs")
    fig.savefig(out / "training_and_resets.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    plot(parser.parse_args().output)

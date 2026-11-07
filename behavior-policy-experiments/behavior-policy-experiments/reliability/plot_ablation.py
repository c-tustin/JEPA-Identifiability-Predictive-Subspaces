"""Show all new-seed outcomes in the two-by-two sampling test."""

import json
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kempner-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).parent / "output-loss-ablation"
    result = json.loads((out / "metrics.json").read_text())
    cells = [
        ("raw", "raw"),
        ("raw", "balanced"),
        ("balanced", "raw"),
        ("balanced", "balanced"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for seed in range(18, 23):
        values = [
            next(
                r["metrics"]["common_grid_r2"]
                for r in result["runs"]
                if r["seed"] == seed and r["alignment"] == a and r["regularizer"] == b
            )
            for a, b in cells
        ]
        ax.plot(range(4), values, marker="o", alpha=0.7, label=f"Seed {seed}")
    ax.set(
        xticks=range(4),
        xticklabels=[
            "Raw / raw",
            "Raw / balanced",
            "Balanced / raw",
            "Balanced / balanced",
        ],
        xlabel="Alignment sampling / SIGReg sampling",
        ylabel="Shared-range $R^2$",
        title="Balancing the regularizer prevents the observed fold",
    )
    ax.legend(fontsize=8)
    fig.savefig(out / "sampling_ablation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

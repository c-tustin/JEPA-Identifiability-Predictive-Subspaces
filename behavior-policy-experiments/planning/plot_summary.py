"""Plot goal errors on the same five validation seeds; no retraining."""

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
    root = Path(__file__).parent
    rows = []
    for source in ["repairs", "policies"]:
        rows += json.loads((root / f"output-{source}" / "metrics.json").read_text())[
            "runs"
        ]
    labels = {
        "ou": "Fixed OU",
        "random": "Random → OU",
        "icm": "ICM → OU",
        "count": "Coverage → OU",
        "action_balanced": "Action-balanced → OU",
        "rnd": "RND → OU",
        "disagreement": "Disagreement → OU",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, kind, title in zip(
        axes,
        ["common", "native"],
        ["Shared predictor-training data", "Policy-native predictor-training data"],
    ):
        for index, (name, label) in enumerate(labels.items()):
            values = [
                r["metrics"]["mean_terminal_distance_error"]
                for r in rows
                if r["condition"] == name
                and r["predictor_data"] == kind
                and r["seed"] in range(8, 13)
            ]
            if values:
                ax.scatter(
                    values,
                    np.full(len(values), index),
                    alpha=0.65,
                    s=28,
                    color=f"C{index}",
                )
                ax.scatter(np.mean(values), index, marker="|", s=250, color="black")
            else:
                ax.text(
                    0.05,
                    index,
                    "No physical action labels",
                    va="center",
                    fontsize=9,
                    color="gray",
                )
        ax.set(
            yticks=range(len(labels)),
            yticklabels=list(labels.values()),
            xlabel="Actual terminal distance error ↓",
            title=title,
        )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Same five validation seeds; points are seed means; black marks are group means"
    )
    fig.savefig(root / "planning_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

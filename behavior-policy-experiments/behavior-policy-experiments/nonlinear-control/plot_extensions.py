"""Plot saved nonlinear control results; dots are representation-seed means.

Average the two predictor fits before plotting each of the three seed points.
No training or additional statistical inference is performed here.
"""

import json
from pathlib import Path

import run_trajectories as study
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).parent
    result = json.loads((root / "output-control" / "metrics.json").read_text())
    if len(result["runs"]) != 84:
        raise ValueError("Expected the complete 84-predictor comparison")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, kind in zip(axes, ["common", "native"]):
        policies = ["random", "persistent", "action_balanced"]
        if kind == "common":
            policies.append("reset_uniform")
        for index, policy in enumerate(policies):
            for sampler, offset, color in [
                ("raw", -0.12, "#c45c32"),
                ("balanced", 0.12, "#287bb5"),
            ]:
                rows = [
                    r
                    for r in result["runs"]
                    if (r["policy"], r["sampler"], r["predictor_data"])
                    == (policy, sampler, kind)
                ]
                values = [
                    np.mean(
                        [
                            r["metrics"]["stochastic"]["mean_closest_error"]
                            for r in rows
                            if r["seed"] == seed
                        ]
                    )
                    for seed in [23, 24, 25]
                ]
                x = index + offset
                ax.scatter(
                    x + np.array([-0.035, 0, 0.035]),
                    values,
                    color=color,
                    alpha=0.75,
                    s=25,
                )
                ax.plot(
                    [x - 0.07, x + 0.07],
                    [np.mean(values)] * 2,
                    color=color,
                    linewidth=3,
                )
        ax.set_xticks(
            range(len(policies)),
            [
                p.replace("action_balanced", "Action\nbalanced")
                .replace("reset_uniform", "Reset\nqueries")
                .capitalize()
                for p in policies
            ],
        )
        ax.set_title(
            "Shared predictor data"
            if kind == "common"
            else "Policy-native predictor data"
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Closest physical goal error (noisy execution) ↓")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color="#c45c32", label="Raw SIGReg"),
            Line2D([0], [0], color="#287bb5", label="Balanced SIGReg"),
        ],
        loc="outside upper center",
        ncol=2,
    )
    output = root / "output-control" / "comparison.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

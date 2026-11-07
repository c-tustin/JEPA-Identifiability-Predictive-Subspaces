"""Aggregate completed main-paper runs. No training and no fabricated partial curves."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys
from launch import ROOT, SEEDS, STUDIES, verify_source


def aggregate(upstream, output):
    verify_source(upstream)
    sys.path.insert(0, str(upstream))
    from plot import collect, plot
    from jepa.artifacts import verify_completion
    rows = []
    for seed in SEEDS:
        seed_root = output / f"seed{seed}"
        plan = json.loads((seed_root / "plan.json").read_text())
        for job in plan:
            folder = seed_root / job["tag"]
            for index in range(3 if job["study"] == "traffic" else 2):
                config = json.loads((folder / f"config_{index}.json").read_text())
                verify_completion(folder / f"complete_{index}.json", config)
        rows.extend(collect(seed_root))
        plot(seed_root)
    expected = {"traffic": 15, "predictability": 15, "subset": 12,
                "reconstruction": 20, "side_input": 15}
    assert len(rows) == 77
    assert {name: sum(r["study"] == name for r in rows) for name in STUDIES} == expected
    assert all(r["weight"] == .1 for r in rows)
    assert {r["seed"] for r in rows if r["study"] == "reconstruction"} == {303}
    groups = {}
    for row in rows:
        for feature, accuracy in row["values"].items():
            key = (row["study"], row["x"], row["beta"], feature)
            groups.setdefault(key, []).append((row["seed"], accuracy))
    summary = []
    for (study, x, beta, feature), values in sorted(groups.items()):
        summary.append(dict(study=study, x=x, beta=beta, feature=feature,
                            seeds=[s for s, _ in values], mean=mean(v for _, v in values),
                            sem=stdev(v for _, v in values) / math.sqrt(len(values))
                            if len(values) > 1 else None))
    target = output / "aggregate"
    target.mkdir(exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (target / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for study, features in [("traffic", ["current_light"]),
                            ("predictability", list("ABCD")),
                            ("subset", ["AB", "CD"])]:
        fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
        for feature in features:
            points = [r for r in summary if r["study"] == study and r["feature"] == feature]
            ax.errorbar([r["x"] for r in points], [100*r["mean"] for r in points],
                        yerr=[100*r["sem"] for r in points], marker="o", label=feature)
        ax.set(xlabel=study + " parameter", ylabel="Probe accuracy (%)")
        ax.axhline(50, color="gray", linestyle="--")
        ax.legend()
        for suffix in ["png", "pdf"]:
            fig.savefig(target / f"{study}.{suffix}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), layout="constrained", sharey=True)
    for ax, feature in zip(axes, "ABCD"):
        for study in ["predictability", "side_input"]:
            points = [r for r in summary if r["study"] == study and r["feature"] == feature]
            ax.errorbar([r["x"] for r in points], [100*r["mean"] for r in points],
                        yerr=[100*r["sem"] for r in points], marker="o", label=study)
        ax.set_title(feature)
        ax.set_xlabel("pD")
        ax.axhline(50, color="gray", linestyle="--")
    axes[0].set_ylabel("Probe accuracy (%)")
    axes[-1].legend()
    for suffix in ["png", "pdf"]:
        fig.savefig(target / f"side_information.{suffix}")
    plt.close(fig)
    print("Aggregated 77 completed encoders. Reconstruction heatmaps: seed303/figures/main/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=ROOT / "upstream")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    aggregate(args.upstream.resolve(), args.output.resolve())

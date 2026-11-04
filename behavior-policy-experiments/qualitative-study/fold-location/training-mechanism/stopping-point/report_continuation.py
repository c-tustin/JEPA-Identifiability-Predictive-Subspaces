"""Plot the continuation controls without assuming a turn must persist."""

import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_continuation import ARMS, HERE, OUT, SEEDS, PARENT_OUT, trials
from report_mechanism import load_run, normalize, roots

LABELS = {"ordinary_lr1e-5": "Continue: learning rate 0.00001",
          "ordinary_lr1e-4": "Continue: learning rate 0.0001",
          "balanced_lr1e-4": "Change SIGReg sampling: rate 0.0001"}
COLORS = {"ordinary_lr1e-5": "#0072B2", "ordinary_lr1e-4": "#D55E00", "balanced_lr1e-4": "#009E73"}


def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder/f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def turns(frame):
    q = np.linspace(0, np.log(16), len(frame["slope"]))
    return [float(np.exp(v)) for v in roots(q, frame["slope"])]


def main():
    plt.rcParams.update({"font.size": 10})
    runs = {(seed, arm): trials.load(OUT/"runs"/f"{arm}-{seed}.pt") for seed in SEEDS for arm in ARMS}
    rows, outcomes = [], []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained", sharey=True)
    for ax, seed in zip(axes, SEEDS):
        original = load_run("ordinary", 4, seed)
        original_points = []
        for frame in original["frames"]:
            for v in roots(original["log_distance"], frame["slope_before"]):
                original_points.append((frame["step"], np.exp(v)))
        if original_points:
            points = np.asarray(original_points)
            ax.scatter(points[:, 0], points[:, 1], s=7, color=".5", alpha=.6, label="Original training")
        initial = runs[seed, "ordinary_lr1e-4"]["frames"][0]
        initial_turns = turns(initial)
        for value in initial_turns:
            ax.plot([4000, 12000], [value, value], color=".3", ls="--", label="Freeze at original endpoint")
        for arm in ARMS:
            run = runs[seed, arm]
            unique = []
            for frame in run["frames"]:
                locations = turns(frame)
                row = {k: frame[k] for k in ("additional_updates", "total_updates", "folded", "r2", "reversal_fraction", "alignment", "mean_sigreg", "mean_objective")}
                row.update(seed=seed, arm=arm, turning_distances=json.dumps(locations))
                rows.append(row)
                unique.append(locations[0] if len(locations) == 1 else np.nan)
                if len(locations) > 1:
                    ax.scatter([frame["total_updates"]]*len(locations), locations, color=COLORS[arm], marker="x")
            ax.plot([f["total_updates"] for f in run["frames"]], unique, color=COLORS[arm], lw=2, label=LABELS[arm])
            for frame in (run["frames"][-5], run["frames"][-1]):
                if len(turns(frame)) == 0:
                    ax.scatter(frame["total_updates"], 1.03, marker="v", color=COLORS[arm])
            last = run["frames"][-1]
            outcomes.append(dict(seed=seed, arm=arm, starting_turns=initial_turns,
                                 final_turns=turns(last), r2=last["r2"], folded=last["folded"],
                                 initial_objective=run["frames"][0]["mean_objective"],
                                 final_objective=last["mean_objective"],
                                 turns_at_total_10000=turns(run["frames"][-5])))
        ax.axvline(4000, color=".3", lw=1)
        ax.axhspan(2, 4, color=".7", alpha=.15)
        ax.set_title(f"Seed {seed}" + (" · primary example" if seed == 131 else " · diagnostic control"))
        ax.set_xlabel("Total training updates")
        ax.set_ylim(1, 16)
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Turning distance (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("The original endpoint is not a universal stopping location\nAll continuations carry the original Adam state. Lines break when there is no unique turn; ▼ marks no detected turn at selected checkpoints.")
    save(fig, "turns_after_original_cutoff")

    # A focused panel makes the small late motion visible without losing the
    # full-range, three-seed comparison above.
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    original = load_run("ordinary", 4, 131)
    points = [(f["step"], float(np.exp(r))) for f in original["frames"]
              if f["step"] >= 2000 for r in roots(original["log_distance"], f["slope_before"])]
    ax.plot(*np.array(points).T, color=".5", lw=2, label="Original training")
    initial_turn = turns(runs[131, "ordinary_lr1e-4"]["frames"][0])[0]
    ax.plot([4000, 12000], [initial_turn]*2, color=".3", ls="--", label=f"Keep learning rate zero: {initial_turn:.3f}")
    for arm in ARMS:
        frames = runs[131, arm]["frames"]
        locations = [turns(f) for f in frames]
        unique = [v[0] if len(v) == 1 else np.nan for v in locations]
        ax.plot([f["total_updates"] for f in frames], unique, color=COLORS[arm], lw=2.5, label=LABELS[arm])
        if len(locations[-1]) == 1:
            ax.annotate(f"{locations[-1][0]:.3f}", (12000, locations[-1][0]), xytext=(7, 0),
                        textcoords="offset points", color=COLORS[arm], va="center", weight="bold")
    ax.axvline(4000, color=".4", lw=1)
    ax.text(4050, 2.75, "Original run ends", color=".3")
    ax.set(xlim=(2000, 12900), ylim=(1.8, 2.8), xlabel="Total training updates",
           ylabel="Turning distance", title="Seed 131: the turn did not settle at 2.4\nRestore a nonzero learning rate and it keeps moving.")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "stopping_point_seed131")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    d = np.exp(np.linspace(0, np.log(16), 1025))
    for ax, seed in zip(axes, SEEDS):
        factor, offset = normalize(load_run("ordinary", 4, seed))
        initial = runs[seed, "ordinary_lr1e-4"]["frames"][0]
        ax.plot(d, factor*initial["curve"]+offset, color=".25", lw=2, ls="--", label="Original endpoint: 4,000 updates")
        for arm in ARMS:
            final = runs[seed, arm]["frames"][-1]
            y = factor*final["curve"]+offset
            ax.plot(d, y, color=COLORS[arm], lw=2, label=LABELS[arm])
            for value in turns(final):
                ax.scatter(value, np.interp(np.log(value), np.log(d), y), color=COLORS[arm], s=35)
        ax.axvspan(2, 4, color=".7", alpha=.15)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set(xlabel="Physical distance", ylabel="Learned coordinate", title=f"Seed {seed}")
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("What another 8,000 updates change\nSolid curves: 12,000 total updates. One fixed affine normalization per seed across every condition; dots mark turns.")
    save(fig, "representations_after_continuation")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    original_lr = [trials.study.controls.learning_rate(i, 4000) for i in range(4000)]
    axes[0].plot(range(4000), original_lr, color=".3", label="Original schedule")
    for arm in ARMS:
        axes[0].plot([4000, 12000], [ARMS[arm][0]]*2, color=COLORS[arm], ls="--" if "balanced" in arm else "-", label=LABELS[arm])
    axes[0].set(xlabel="Total updates", ylabel="Learning rate", title="The original learning rate reaches zero")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for arm in ARMS:
        run = runs[131, arm]
        frames = run["frames"]
        ratio = np.array([f["mean_objective"] for f in frames])/frames[0]["mean_objective"]
        axes[1].plot([f["total_updates"] for f in frames], ratio, color=COLORS[arm], lw=2)
    axes[1].set(xlabel="Total updates", ylabel="Objective / its own value at continuation start", title="Seed 131: does training still reduce its loss?")
    for ax in axes:
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stopping the optimizer is different from reaching a stationary representation\nBalanced sampling changes its loss measure; normalized losses compare progress within an arm, not quality across arms.")
    save(fig, "learning_rate_and_remaining_progress")
    with (OUT/"trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    trials.atomic_json(OUT/"summary.json", dict(outcomes=outcomes, original_last_lr=original_lr[-1],
                                             interpretation="Finite continuations, not a convergence theorem; three diagnostic seeds, not population-level significance."))
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()

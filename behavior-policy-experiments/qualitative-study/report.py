"""Readable plots and seed-level statistics from the completed paired study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from run_trials import ARMS, EDGES, ROOT, analysis_bins, atomic_json, load
from trial_stats import TIE, curriculum_comparisons, holm_adjust, paired_summary

PRIMARY = ("icm", "ou", "icm_to_ou")
NAMES = {
    "icm": "ICM",
    "ou": "OU",
    "icm_to_ou": "ICM → OU",
    "balanced_log": "Balanced log-size",
    "balanced_size": "Balanced raw size",
}
COLORS = {
    "icm": "#D55E00",
    "ou": "#0072B2",
    "icm_to_ou": "#009E73",
    "balanced_log": "#CC79A7",
    "balanced_size": "#E69F00",
}
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])


def entropy(counts):
    p = np.asarray(counts, dtype=float) / max(np.sum(counts), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def normalized(counts):
    return np.asarray(counts) / max(np.sum(counts), 1)


def save_figure(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def collect(out, seeds):
    curves, plans, rows = {}, {}, []
    for seed in seeds:
        for arm in ARMS:
            curve = load(out / "curves" / f"{arm}-{seed}.pt")
            curves[seed, arm] = curve
            for kind in ("shared",) if arm == "ou" else ("shared", "native"):
                plans[seed, arm, kind] = load(
                    out / "planning" / f"{arm}-{kind}-{seed}.pt"
                )
            shared = plans[seed, arm, "shared"]
            reversed_steps = np.diff(curve["embedding"]) < -1e-5
            route = shared["test"]["oracle_path"].ravel()
            step = np.searchsorted(curve["distance"], route, side="right") - 1
            route_reversed = reversed_steps[np.clip(step, 0, len(reversed_steps) - 1)]
            reversed_distance = curve["distance"][1:][reversed_steps]
            counts, reg = (
                curve["actual_training_counts"],
                curve["actual_regularizer_counts"],
            )
            rows.append(
                dict(
                    seed=seed,
                    arm=arm,
                    folded=curve["folded"],
                    r2=curve["r2"],
                    reversal_fraction=curve["reversal_fraction"],
                    oracle_reversal_exposure=float(route_reversed.mean()),
                    reversal_low=(
                        float(reversed_distance.min())
                        if len(reversed_distance)
                        else None
                    ),
                    reversal_high=(
                        float(reversed_distance.max())
                        if len(reversed_distance)
                        else None
                    ),
                    sensitivity_p05=curve["sensitivity_p05"],
                    raw_variance=curve["raw_variance"],
                    source_occupied=curve["source_coverage"]["occupied"],
                    source_entropy=curve["source_coverage"]["entropy"],
                    training_occupied=int((counts > 0).sum()),
                    training_entropy=entropy(counts),
                    regularizer_entropy=entropy(reg),
                    screen_error=shared["screen"]["mean_error"],
                    test_error=shared["test"]["mean_error"],
                    test_success=shared["test"]["success"],
                    native_error=(
                        None
                        if arm == "ou"
                        else plans[seed, arm, "native"]["test"]["mean_error"]
                    ),
                )
            )
    return curves, plans, rows


def curve_grid(seeds, curves, rows, arms, folder, name, title):
    lookup = {(r["seed"], r["arm"]): r for r in rows}
    ncols = 5 if len(seeds) > 4 else min(len(seeds), 4)
    nrows = (len(seeds) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.1 * nrows),
        squeeze=False,
        layout="constrained",
    )
    for ax, seed in zip(axes.flat, seeds):
        for arm in arms:
            curve = curves[seed, arm]
            d, y = curve["distance"], curve["embedding"]
            ax.plot(d, y, color=COLORS[arm], lw=1.6, label=NAMES[arm])
            if len(arms) == 1:
                reversed = np.r_[False, np.diff(y) < -1e-5]
                ax.plot(d, np.where(reversed, y, np.nan), color="#CC0000", lw=2.8)
        ideal = (np.log(d) - np.log(4)) / (np.log(16) / np.sqrt(12))
        ax.plot(
            d, ideal, "--", color=".65", lw=0.8, label="Affine log-distance reference"
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("Physical distance (log scale)")
        ax.set_ylabel("Normalized learned coordinate")
        if len(arms) == 1:
            row = lookup[seed, arms[0]]
            ax.set_title(
                f"Seed {seed} · {'FOLDED' if row['folded'] else 'no grid fold'}\n"
                f"R²={row['r2']:.3f}; goal error={row['test_error']:.3f}",
                fontsize=9,
            )
        else:
            ax.set_title(f"Seed {seed}")
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(seeds) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(labels), 4),
        frameon=False,
    )
    fig.suptitle(
        title
        + "\nOnly affine display normalization; no unfolding or nonlinear correction",
        fontsize=13,
    )
    save_figure(fig, folder, name)


def planning_plots(seeds, rows, statistics, folder):
    lookup = {(r["seed"], r["arm"]): r for r in rows}
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained", sharey=True)
    for arm in PRIMARY:
        axes[0].plot(
            x,
            [lookup[s, arm]["test_error"] for s in seeds],
            "o-",
            color=COLORS[arm],
            label=NAMES[arm],
        )
    for arm in ("icm", "icm_to_ou"):
        axes[1].plot(
            x,
            [lookup[s, arm]["native_error"] for s in seeds],
            "o-",
            color=COLORS[arm],
            label=NAMES[arm],
        )
    for ax, title in zip(
        axes, ["Same physical predictor data", "Predictor uses ICM's own physical data"]
    ):
        ax.set_xticks(x, [str(s) for s in seeds])
        ax.set_xlabel("Paired trial seed")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean terminal physical goal error ↓")
    fig.suptitle(
        "Planning errors, not latent prediction losses\nOU-native is undefined: passive OU pairs have no physical action labels"
    )
    save_figure(fig, folder, "planning_by_trial")
    delta = np.array(
        [
            lookup[s, "icm_to_ou"]["test_error"] - lookup[s, "ou"]["test_error"]
            for s in seeds
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.bar(x, delta, color=np.where(delta < -TIE, COLORS["icm_to_ou"], COLORS["icm"]))
    selected = statistics["screen_selected_seeds"]
    for i, s in enumerate(seeds):
        if s in selected:
            ax.scatter(i, delta[i], marker="D", color="black", s=55, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set(
        xlabel="Seed · black diamond = passed separate validation screen",
        ylabel="ICM→OU error − OU error",
        title="Held-out wins and losses: negative favors ICM→OU",
    )
    save_figure(fig, folder, "curriculum_wins_and_losses")


def coverage_plots(seeds, curves, rows, plans, out, folder):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    metrics = [
        ("training_occupied", "Occupied training bins"),
        ("source_occupied", "Occupied source bins"),
        ("regularizer_entropy", "SIGReg exposure entropy"),
        ("reversal_fraction", "Fraction of reversed grid steps"),
        ("oracle_reversal_exposure", "Oracle positions inside reversed intervals"),
        ("sensitivity_p05", "5th percentile |dh/dd|"),
    ]
    for ax, (key, label) in zip(axes.flat, metrics):
        for arm in ARMS:
            chosen = [r for r in rows if r["arm"] == arm]
            ax.scatter(
                [
                    max(r[key], 1e-9) if key == "sensitivity_p05" else r[key]
                    for r in chosen
                ],
                [r["test_error"] for r in chosen],
                color=COLORS[arm],
                label=NAMES[arm],
                s=35,
                alpha=0.8,
            )
        if key == "sensitivity_p05":
            ax.set_xscale("log")
        ax.set(xlabel=label, ylabel="Shared-data physical goal error ↓")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Coverage, folds and sensitivity versus actual planning\nRepeated methods within each seed; descriptive associations"
    )
    save_figure(fig, folder, "coverage_and_geometry_vs_planning")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for arm in ("icm_to_ou", "balanced_log", "balanced_size"):
        p = np.stack(
            [normalized(curves[s, arm]["actual_regularizer_counts"]) for s in seeds]
        )
        axes[0].plot(CENTERS, p.mean(0), color=COLORS[arm], label=NAMES[arm])
        records = [r for r in rows if r["arm"] == arm]
        index = ("icm_to_ou", "balanced_log", "balanced_size").index(arm)
        axes[1].scatter(
            np.full(len(records), index),
            [r["test_error"] for r in records],
            color=COLORS[arm],
            s=30,
        )
        axes[1].plot(
            index, np.mean([r["test_error"] for r in records]), "k_", ms=20, mew=2
        )
    for seed in seeds:
        lookup = {r["arm"]: r for r in rows if r["seed"] == seed}
        axes[1].plot(
            range(3),
            [
                lookup[a]["test_error"]
                for a in ("icm_to_ou", "balanced_log", "balanced_size")
            ],
            color=".7",
            lw=0.6,
            zorder=0,
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    axes[0].set(
        xlabel="Physical distance",
        ylabel="Fraction of SIGReg training exposure",
        title="Same stored transitions, different replay weights",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks(range(3), ["Ordinary", "Log-size bins", "Raw-size bins"])
    axes[1].set(
        ylabel="Shared-data goal error ↓",
        title="Changing replay can change planning geometry",
    )
    save_figure(fig, folder, "binning_exposure_and_planning")
    selected = [r for r in rows if r["arm"] in PRIMARY]
    probabilities = np.stack(
        [
            normalized(curves[r["seed"], r["arm"]]["actual_training_counts"])
            for r in selected
        ]
    )
    fig, (ax, bars) = plt.subplots(
        1,
        2,
        figsize=(12, 10),
        gridspec_kw={"width_ratios": [5, 1]},
        layout="constrained",
    )
    image = ax.imshow(
        np.log10(probabilities + 1e-6),
        aspect="auto",
        origin="upper",
        vmin=-6,
        vmax=-0.5,
        cmap="viridis",
    )
    ax.set_yticks(
        range(len(selected)),
        [f"{r['seed']} {NAMES[r['arm']]}" for r in selected],
        fontsize=7,
    )
    ax.set_xticks([0, 7, 15, 23, 31], [f"{CENTERS[i]:.1f}" for i in [0, 7, 15, 23, 31]])
    ax.set(
        xlabel="Distance-bin center",
        title="Actual representation-training exposure (log₁₀ fraction)",
    )
    bars.barh(
        range(len(selected)),
        [r["test_error"] for r in selected],
        color=[COLORS[r["arm"]] for r in selected],
    )
    bars.set_ylim(len(selected) - 0.5, -0.5)
    bars.set_yticks([])
    bars.set(xlabel="Goal error ↓", title="Same row")
    fig.colorbar(image, ax=ax, label="log₁₀ exposure; −6 includes empty bins")
    save_figure(fig, folder, "coverage_heatmap")
    # Correlations are centered within seed and have no uncorrected p-value claims.
    logp = np.log10(
        np.stack(
            [
                normalized(curves[r["seed"], r["arm"]]["actual_regularizer_counts"])
                for r in rows
            ]
        )
        + 1e-6
    )
    error = np.array([r["test_error"] for r in rows])
    centered = logp.copy()
    centered_error = error.copy()
    for seed in seeds:
        ix = np.array([r["seed"] == seed for r in rows])
        centered[ix] -= centered[ix].mean(0)
        centered_error[ix] -= centered_error[ix].mean()
    correlation = [
        (
            float(spearmanr(centered[:, b], centered_error).statistic)
            if np.std(centered[:, b]) > 1e-12 and np.std(centered_error) > 1e-12
            else 0.0
        )
        for b in range(32)
    ]
    demand = np.zeros(32)
    for seed in seeds:
        oracle = plans[seed, "ou", "shared"]["test"]["oracle_path"]
        demand += np.histogram(oracle, EDGES)[0] / oracle.size
    demand /= len(seeds)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained", sharex=True)
    axes[0].bar(CENTERS, correlation, width=np.diff(EDGES) * 0.8, color="#666666")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(
        ylabel="Within-seed Spearman correlation",
        title="Exploratory: bin exposure versus error (positive = worse)",
    )
    axes[1].bar(CENTERS, demand, width=np.diff(EDGES) * 0.8, color=COLORS["ou"])
    axes[1].set(
        ylabel="Oracle-route occupancy",
        xlabel="Physical distance",
        title="Where these particular goal tasks need to travel or stay",
    )
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    save_figure(fig, folder, "bin_associations_and_task_demand")
    atomic_json(
        out / "coverage-associations.json",
        dict(
            correlations=correlation,
            oracle_bin_demand=demand.tolist(),
            interpretation="Exploratory within-seed associations, not independent observations or causal bin effects",
        ),
    )


def action_coverage(seeds, out, folder):
    totals = []
    for source in ("icm", "physical-common"):
        counts = np.zeros((32, 3), dtype=float)
        for seed in seeds:
            saved = load(out / "sources" / f"{source}-{seed}.pt")
            data = saved["data"] if source == "icm" else saved
            bins, valid = analysis_bins(data["x"])
            ix = (bins[valid] * 3 + data["action"][valid]).numpy()
            counts += np.bincount(ix, minlength=96).reshape(32, 3)
        totals.append(counts)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), layout="constrained")
    for ax, counts, title in zip(
        axes[:2], totals, ["ICM's physical pairs", "Shared random-action pairs"]
    ):
        probability = np.divide(
            counts,
            counts.sum(1, keepdims=True),
            out=np.full_like(counts, np.nan),
            where=counts.sum(1, keepdims=True) > 0,
        )
        image = ax.imshow(
            probability, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma"
        )
        ax.set_xticks(range(3), ["approach", "stay", "retreat"], rotation=20)
        ax.set_yticks(
            [0, 7, 15, 23, 31], [f"{CENTERS[i]:.1f}" for i in [0, 7, 15, 23, 31]]
        )
        ax.set(title=title, ylabel="Distance-bin center")
    fig.colorbar(
        image, ax=axes[:2], label="Action fraction within bin; blank = no data"
    )
    for counts, color, label in zip(
        totals, [COLORS["icm"], COLORS["ou"]], ["ICM", "Shared"]
    ):
        axes[2].plot(CENTERS, counts.sum(1) / counts.sum(), color=color, label=label)
    axes[2].set_xscale("log")
    axes[2].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    axes[2].set(
        xlabel="Physical distance",
        ylabel="Fraction of physical predictor data",
        title="State coverage also differs",
    )
    axes[2].legend(frameon=False)
    save_figure(fig, folder, "physical_state_action_coverage")


def trial_diagnostics(seeds, curves, plans, out, folder):
    """Show each ICM source distribution and one concrete failed planning task."""
    ncols = min(len(seeds), 5)
    nrows = (len(seeds) + ncols - 1) // ncols
    for kind in ("sampling", "paths"):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols, 3.1 * nrows),
            squeeze=False,
            layout="constrained",
        )
        examples = []
        for ax, seed in zip(axes.flat, seeds):
            if kind == "sampling":
                curve = curves[seed, "icm"]
                ax.plot(
                    CENTERS,
                    normalized(curve["source_coverage"]["counts"]),
                    color=COLORS["icm"],
                    label="ICM collected pairs",
                )
                ax.plot(
                    CENTERS,
                    normalized(curve["actual_regularizer_counts"]),
                    "--",
                    color="black",
                    label="ICM SIGReg exposure",
                )
                ax.plot(
                    CENTERS,
                    normalized(curves[seed, "ou"]["actual_regularizer_counts"]),
                    color=COLORS["ou"],
                    label="OU SIGReg exposure",
                )
                ax.set_xscale("log")
                ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
                ax.set(
                    xlabel="Physical distance",
                    ylabel="Within-range fraction",
                    title=f"Seed {seed}",
                )
            else:
                reference = plans[seed, "icm", "shared"]["test"]
                # Selection is only for illustrating failures, never for a statistical test.
                index = int(np.argmax(reference["error"]))
                goal = float(reference["goal"][index])
                for arm in PRIMARY:
                    case = plans[seed, arm, "shared"]["test"]
                    ax.plot(
                        range(13),
                        case["path"][:, index],
                        color=COLORS[arm],
                        label=NAMES[arm],
                    )
                ax.axhline(goal, color="black", ls="--", label="Goal")
                ax.set(
                    xlabel="Executed decision",
                    ylabel="Physical distance",
                    title=f"Seed {seed} · ICM's worst test task",
                )
                examples.append(
                    dict(
                        seed=seed,
                        case=index,
                        start=float(reference["start"][index]),
                        goal=goal,
                        errors={
                            arm: float(
                                plans[seed, arm, "shared"]["test"]["error"][index]
                            )
                            for arm in PRIMARY
                        },
                    )
                )
            ax.grid(alpha=0.15)
        for ax in list(axes.flat)[len(seeds) :]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=len(labels),
            frameon=False,
        )
        fig.suptitle(
            "ICM sampling varies across training runs"
            if kind == "sampling"
            else "Concrete trajectories on the same held-out tasks\nCases chosen to illustrate ICM failures; not used to select the success subgroup"
        )
        save_figure(
            fig,
            folder,
            (
                "icm_sampling_by_trial"
                if kind == "sampling"
                else "planning_failure_examples"
            ),
        )
        if kind == "paths":
            atomic_json(out / "illustrative-failure-cases.json", examples)


def region_analysis(seeds, out, folder):
    summaries = []
    matrix = np.zeros((len(seeds), 8))
    demand = np.zeros((len(seeds), 8))
    per_seed = []
    for i, seed in enumerate(seeds):
        base = load(out / "planning" / f"remove-all-{seed}.pt")["test"]
        for region in range(8):
            changed = load(out / "planning" / f"remove-{region}-{seed}.pt")["test"]
            np.testing.assert_array_equal(base["goal"], changed["goal"])
            np.testing.assert_array_equal(base["start"], changed["start"])
            low, high = EDGES[4 * region], EDGES[4 * (region + 1)]
            crossing = (
                (base["oracle_path"] >= low) & (base["oracle_path"] < high)
            ).any(0)
            delta = changed["error"] - base["error"]
            matrix[i, region] = delta.mean()
            demand[i, region] = crossing.mean()
            per_seed.append(
                dict(
                    seed=seed,
                    region=region,
                    low=float(low),
                    high=float(high),
                    error_change=float(delta.mean()),
                    control_error=base["mean_error"],
                    removed_error=changed["mean_error"],
                    crossing_tasks=int(crossing.sum()),
                    crossing_error_change=(
                        float(delta[crossing].mean()) if crossing.any() else None
                    ),
                )
            )
    for region in range(8):
        chosen = [r for r in per_seed if r["region"] == region]
        summary = paired_summary(
            [r["removed_error"] for r in chosen],
            [r["control_error"] for r in chosen],
            seed=4300 + region,
        )
        summary.update(
            region=region,
            low=float(EDGES[4 * region]),
            high=float(EDGES[4 * (region + 1)]),
            oracle_crossing_fraction=float(demand[:, region].mean()),
        )
        summaries.append(summary)
    adjusted = holm_adjust([s["sign_test_p"] for s in summaries])
    for row, p in zip(summaries, adjusted):
        row["sign_test_holm_p"] = p
    atomic_json(
        out / "region-importance.json",
        dict(
            summaries=summaries,
            per_seed=per_seed,
            interpretation="Fixed OU encoder; paired predictor-data removal; positive change means removing this region harmed planning",
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained", sharex=True)
    means = matrix.mean(0)
    intervals = np.array([s["bootstrap_95"] for s in summaries])
    axes[0].bar(range(8), means, color=COLORS["ou"], alpha=0.7)
    axes[0].vlines(range(8), intervals[:, 0], intervals[:, 1], color="black", lw=1.5)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(
        ylabel="Increase in physical goal error",
        title="Remove one training-data region; freeze the OU encoder\nBars: paired means; lines: seed-bootstrap 95% intervals",
    )
    axes[1].bar(range(8), demand.mean(0), color="#888888")
    axes[1].set(
        ylabel="Fraction of tasks crossing region",
        title="Oracle routes: which regions do the evaluation tasks actually need?",
    )
    limit = max(float(abs(matrix).max()), 1e-4)
    image = axes[2].imshow(
        matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axes[2].set_yticks(range(len(seeds)), [str(s) for s in seeds])
    axes[2].set_ylabel("Paired seed")
    axes[2].set_xticks(
        range(8), [f"{EDGES[4*g]:.2g}–{EDGES[4*(g+1)]:.2g}" for g in range(8)]
    )
    axes[2].set_xlabel("Removed distance region (four of the 32 bins)")
    fig.colorbar(image, ax=axes[2], label="Error increase: red harmful, blue improved")
    save_figure(fig, folder, "controlled_region_importance")
    return summaries


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def write_results(out, seeds, rows, statistics, regions, quick):
    lookup = {(r["seed"], r["arm"]): r for r in rows}
    all_trials, screen = statistics["comparisons"]
    lines = ["# Ten-trial results: representation folds, planning and coverage", ""]
    if quick:
        lines += ["**Smoke check only: reduced budgets, not scientific results.**", ""]
    lines += [
        "Smaller physical goal error is better. Every row is a matched seed; the same held-out goals and predictor-training data compare the three main encoders. ICM + OU means ICM→OU.",
        "",
        "| Seed | ICM error | OU error | ICM→OU error | Curriculum − OU | Passed separate screen? |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in seeds:
        a, b, h = [lookup[s, arm] for arm in PRIMARY]
        lines.append(
            f"| {s} | {a['test_error']:.4f} | {b['test_error']:.4f} | {h['test_error']:.4f} | {h['test_error']-b['test_error']:+.4f} | {'yes' if s in statistics['screen_selected_seeds'] else 'no'} |"
        )
    lines += [
        "",
        "## Statistical comparison",
        "",
        "The seed is the independent unit. The exact sign test asks whether the curriculum wins more often than it loses; it does not test the mean error. Both sign-test p-values below are Holm-adjusted. Mean changes and paired bootstrap intervals are also reported.",
        "",
        "| Comparison | Seeds | Wins / losses / ties | Mean error change | 95% paired-seed bootstrap interval | Adjusted sign-test p |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for result in statistics["comparisons"]:
        interval = result["bootstrap_95"]
        ci = "—" if interval is None else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
        lines.append(
            f"| {result['name']} | {result['n']} | {result['wins']} / {result['losses']} / {result['ties']} | {fmt(result['mean_difference'])} | {ci} | {result['sign_test_holm_p']:.4f} |"
        )
    lines += [
        "",
        "The subgroup is selected using separate validation goals and a fold check, never the held-out errors in the table. Its conclusion is conditional on passing this particular screen. A small subgroup has little power; testing only the held-out winners would be invalid. Supplementary paired t-tests and their assumptions are in [statistics.json](statistics.json).",
        "",
        "## What to inspect qualitatively",
        "",
    ]
    for arm in PRIMARY:
        selected = [r for r in rows if r["arm"] == arm]
        lines.append(
            f"- **{NAMES[arm]}:** {sum(r['folded'] for r in selected)}/{len(selected)} runs have a substantial grid reversal; average shared-data goal error {np.mean([r['test_error'] for r in selected]):.4f}."
        )
    lines += [
        "",
        "Each representation panel shows the learned coordinate against physical distance, with only affine normalization. Red curve segments reverse direction. A fold maps different distances to similar outputs; a monotone curve can still compress useful distances or make action prediction difficult. These plots diagnose geometry; they do not prove an optimization mechanism.",
        "",
        "Fold location also matters: a reversal mostly outside the routes used by our local goals can look less damaging than a reversal across the center of the task range. The following uses oracle routes only as an evaluation diagnostic, not as training data or a success screen.",
        "",
        "| ICM seed | Reversed distance span | Oracle positions in reversed intervals | Shared-data goal error |",
        "| --- | --- | ---: | ---: |",
    ]
    for s in seeds:
        r = lookup[s, "icm"]
        span = (
            "none"
            if r["reversal_low"] is None
            else f"{r['reversal_low']:.2f}–{r['reversal_high']:.2f}"
        )
        lines.append(
            f"| {s} | {span} | {r['oracle_reversal_exposure']:.1%} | {r['test_error']:.4f} |"
        )
    lines += [
        "",
        "## Binning and planning",
        "",
        "| SIGReg sampling in ICM→OU | Shared-data mean goal error | Native-data mean goal error | Mean SIGReg exposure entropy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ("icm_to_ou", "balanced_log", "balanced_size"):
        selected = [r for r in rows if r["arm"] == arm]
        lines.append(
            f"| {NAMES[arm]} | {np.mean([r['test_error'] for r in selected]):.4f} | {np.mean([r['native_error'] for r in selected]):.4f} | {np.mean([r['regularizer_entropy'] for r in selected]):.3f} |"
        )
    ranked = sorted(regions, key=lambda r: r["mean_difference"], reverse=True)
    lines += [
        "",
        "Replay reweights existing observations; it does not add unseen states or missing action examples. Log-size and raw-size bins define different target sampling measures for SIGReg. Shared-data planning holds the predictor data fixed; native-data planning also exposes the physical action-coverage problem. Passive OU has no physical action labels, so an OU-native predictor is not fabricated.",
        "",
        "## Which distance regions matter?",
        "",
        "The controlled intervention freezes each OU encoder and removes one region from predictor training. Positive error change means that removal harmed planning. This is different from correlating a coverage histogram with error.",
        "",
        "| Removed distance region | Mean error increase | Tasks whose oracle route crosses it | Adjusted sign-test p |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in ranked:
        lines.append(
            f"| {r['low']:.2f}–{r['high']:.2f} | {r['mean_difference']:+.4f} | {r['oracle_crossing_fraction']:.1%} | {r['sign_test_holm_p']:.4f} |"
        )
    lines += [
        "",
        "These are eight four-bin regions, not separately identified individual-bin effects. The result is specific to these local goals, frozen OU encoders, predictor family and budget. Removing data can also change finite-fit behavior globally; negative changes are retained. Per-bin correlations are exploratory and do not establish causal importance.",
        "",
        "[Representation curves](figures/representations_overlay.png) · [Planning wins/losses](figures/curriculum_wins_and_losses.png) · [Binning comparison](figures/binning_exposure_and_planning.png) · [Controlled region importance](figures/controlled_region_importance.png)",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    out = args.output.resolve()
    settings = json.loads((out / "settings.json").read_text())
    seeds = settings["seeds"]
    if not all((out / f"completed-{s}.json").exists() for s in seeds):
        raise RuntimeError(
            "Requested cohort is incomplete; do not report partial trials as the full study"
        )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    curves, plans, rows = collect(out, seeds)
    statistics = curriculum_comparisons(rows)
    statistics["quick"] = settings["quick"]
    atomic_json(out / "statistics.json", statistics)
    atomic_json(out / "trials.json", rows)
    with (out / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for arm in PRIMARY:
        curve_grid(
            seeds,
            curves,
            rows,
            [arm],
            folder,
            f"representations_{arm}",
            NAMES[arm] + f": {len(seeds)} representation trials",
        )
    curve_grid(
        seeds,
        curves,
        rows,
        PRIMARY,
        folder,
        "representations_overlay",
        "Matched ICM, OU and ICM→OU representations",
    )
    curve_grid(
        seeds,
        curves,
        rows,
        ("icm_to_ou", "balanced_log", "balanced_size"),
        folder,
        "representations_binning",
        "How the SIGReg binning changes the coordinate",
    )
    planning_plots(seeds, rows, statistics, folder)
    coverage_plots(seeds, curves, rows, plans, out, folder)
    action_coverage(seeds, out, folder)
    trial_diagnostics(seeds, curves, plans, out, folder)
    regions = region_analysis(seeds, out, folder)
    write_results(out, seeds, rows, statistics, regions, settings["quick"])
    print(json.dumps(statistics, indent=2))
    print("Saved trial table, statistics and figure gallery to", out)


if __name__ == "__main__":
    main()

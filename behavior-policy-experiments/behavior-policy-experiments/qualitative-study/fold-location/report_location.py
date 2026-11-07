"""Readable plots separating fold roots, tapering, sampling and input scale."""

import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_location import (
    BASE_ENCODER, GAINS, HERE, ORIGINAL, OUT, ScaledEncoder, trials,
)

COLORS = {2: "#0072B2", 4: "#D55E00", 8: "#009E73"}
ARM_COLORS = {"icm": "#D55E00", "ou": "#0072B2", "balanced_log": "#CC79A7", "initial": "#777777"}
LABELS = {"icm": "ICM", "ou": "OU", "balanced_log": "Balanced SIGReg", "initial": "Before training"}


def load_profile(arm, gain, seed):
    return trials.load(OUT / "profiles" / f"{arm}-gain-{gain}-{seed}.pt")


def axis_distance(ax):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.axvspan(2, 4, color="#bbbbbb", alpha=.17, zorder=-2)
    ax.set_xlabel("Physical distance (log scale)")
    ax.grid(alpha=.15)


def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def grid(title):
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), layout="constrained")
    fig.suptitle(title, fontsize=13)
    return fig, axes


def legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=len(labels), frameon=False)


def input_scale():
    d = np.geomspace(1, 16, 1024)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for v, label, color in [(0.5, "|V| = 0.5", "#0072B2"), (np.sqrt(3)/2, "|V| = 0.866", "#D55E00"), (1., "|V| = 1", "#009E73")]:
        u = np.tanh(4*v/d)
        axes[0,0].plot(d, v/d, label=label, color=color)
        axes[0,1].plot(d, u, label=label, color=color)
        axes[1,0].plot(d, 4*v/d*(1-u*u), label=label, color=color)
        axes[0,1].axvline(4*v, color=color, ls=":", alpha=.7)
    for gain in GAINS:
        u = np.tanh(gain/d)
        axes[1,1].plot(d, gain/d*(1-u*u), color=COLORS[gain], label=f"Input gain {gain}")
    titles = ["Raw nonzero input magnitudes", "Actual fixed tanh preprocessing\nDotted lines: |4V/d| = 1", "Preprocessing sensitivity to log-distance", "Changing gain shifts the input sensitivity\nShown for the height coordinate, |V|=1"]
    ylabels = ["|V| / d", "tanh(4 |V| / d)", "|du / d(log d)|", "|du / d(log d)|"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        axis_distance(ax)
        ax.set(title=title, ylabel=ylabel)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("A smooth scale exists before any training; it does not force a fold")
    save(fig, "input_scale")


def baseline_grids(seeds):
    fig, axes = grid("Same input gain, different learned geometry · gray band: distances 2–4")
    fig2, axes2 = grid("Signed output sensitivity · zero crossing = an extremum; near zero = local flattening")
    for i, (ax, sensitivity, seed) in enumerate(zip(axes.flat, axes2.flat, seeds)):
        source = trials.load(ORIGINAL / "sources" / f"icm-{seed}.pt")["data"]
        counts = trials.bin_counts(source["x"])
        twin = ax.twinx()
        edges = trials.EDGES
        twin.stairs(counts / counts.sum(), edges, color=".7", alpha=.6, fill=True)
        twin.set_ylim(0, .65)
        twin.set_yticks([0,.3,.6])
        twin.tick_params(axis="y",labelsize=6,colors=".5")
        if i % 5 == 4:
            twin.set_ylabel("ICM visit fraction",fontsize=7,color=".5")
        twin.set_zorder(-1)
        ax.patch.set_visible(False)
        for arm in ("icm", "ou", "balanced_log"):
            p = load_profile(arm, 4, seed)
            ax.plot(p["distance"], p["embedding"], label=LABELS[arm], color=ARM_COLORS[arm])
        for arm in ("initial", "icm", "ou", "balanced_log"):
            p = load_profile(arm, 4, seed)
            sensitivity.plot(p["distance"], p["derivative_log_distance"], label=LABELS[arm], color=ARM_COLORS[arm], lw=1.3)
        for a in (ax, sensitivity):
            axis_distance(a)
            a.set_title(f"Seed {seed}")
        ax.set_ylabel("Learned coordinate")
        sensitivity.axhline(0, color="black", lw=.6)
        sensitivity.set_ylabel("dh / d(log d)")
    legend(fig, axes.flat[0])
    legend(fig2, axes2.flat[0])
    fig.text(.5, -.02, "Gray histogram: source ICM visit fractions (separate right-hand scale); not a derivative or training weight", ha="center")
    save(fig, "location_and_sampling")
    save(fig2, "signed_sensitivity")


def gain_grids(seeds):
    for arm in ("icm", "ou"):
        fig, axes = grid(f"{LABELS[arm]} · retrain on the same pairs at input gains 2, 4 and 8")
        for ax, seed in zip(axes.flat, seeds):
            notes = []
            for gain in GAINS:
                p = load_profile(arm, gain, seed)
                ax.plot(p["distance"], p["embedding"], color=COLORS[gain], label=f"Gain {gain}")
                roots = p["turning_distances"]
                notes.append(", ".join(f"{r:.2f}" for r in roots) if roots else "—")
            ax.set_title(f"Seed {seed}\nTurn g2/4/8: " + " / ".join(notes), fontsize=8)
            axis_distance(ax)
            ax.set_ylabel("Normalized learned coordinate")
        legend(fig, axes.flat[0])
        save(fig, f"{arm}_retrained_gain_comparison")


def unit_diagnostics():
    seeds = (130, 131, 136, 139)
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), layout="constrained")
    for row, seed in zip(axes, seeds):
        p = load_profile("icm", 4, seed)
        d = p["distance"]
        row[0].plot(d, p["embedding"], color=ARM_COLORS["icm"])
        row[0].set(title=f"Seed {seed}: learned curve", ylabel="Normalized coordinate")
        terms = p["last_layer_derivative_terms"]
        largest = np.argsort(np.mean(abs(terms), axis=0))[-6:]
        for j in largest:
            row[1].plot(d, terms[:,j], lw=.7, alpha=.7)
        row[1].plot(d, p["derivative_log_distance"], color="black", lw=2, label="Total (all 48 units)")
        row[1].axhline(0, color="black", lw=.5)
        row[1].set(title="Six largest individual contributions + total", ylabel="Signed output derivative")
        row[1].legend(fontsize=7, frameon=False)
        row[2].plot(d, p["cancellation_ratio"], color="black", label="|sum| / sum of |terms|")
        row[2].plot(d, p["hidden1_saturated"], color="#0072B2", label="Hidden layer 1 saturated fraction")
        row[2].plot(d, p["hidden2_saturated"], color="#CC79A7", label="Hidden layer 2 saturated fraction")
        row[2].set(ylim=(-.03,1.03), title="Cancellation versus tanh saturation", ylabel="Fraction or ratio")
        row[2].legend(fontsize=6, frameon=False)
        for ax in row:
            axis_distance(ax)
    fig.suptitle("A zero output slope can come from opposing unit contributions, not every unit being saturated\nSaturated = |tanh activation| > 0.95; six units shown are selected only for illustration")
    save(fig, "hidden_derivative_diagnostics")


def turn_zoom():
    p = load_profile("icm",4,131)
    d = p["distance"]
    turn = p["turning_distances"][0]
    terms = p["last_layer_derivative_terms"]
    positive = np.maximum(terms,0).sum(1)
    negative = np.minimum(terms,0).sum(1)
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].plot(d,p["embedding"],color=ARM_COLORS["icm"])
    axes[0].set(ylabel="Normalized learned coordinate",title="A smooth minimum, not a physical boundary")
    local = (d>1.7)&(d<3.1)
    y = p["embedding"][local]
    axes[0].set_ylim(float(y.min())-.03,float(y.max())+.03)
    axes[1].plot(d,positive,color="#0072B2",label="Sum of positive contributions")
    axes[1].plot(d,negative,color="#D55E00",label="Sum of negative contributions")
    axes[1].plot(d,positive+negative,color="black",label="Total derivative")
    limit = max(float(positive[local].max()),float(-negative[local].min()))*1.1
    axes[1].set(ylim=(-limit,limit),ylabel="Contribution to dh / d(log d)",title="All 48 units: contributions cancel at the turn")
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set(xlim=(1.7,3.1),xlabel="Physical distance (linear zoom)")
        ax.axvline(turn,color=".5",ls=":")
        ax.grid(alpha=.15)
    fig.suptitle(f"Seed 131 · original gain 4 · turning distance {turn:.3f}")
    save(fig,"seed131_turn_zoom")


def locations(seeds):
    fig, axes = plt.subplots(1, 2, figsize=(12,5), layout="constrained")
    colors = plt.get_cmap("tab10")
    for i, seed in enumerate(seeds):
        for k, gain in enumerate(GAINS):
            p = load_profile("icm", gain, seed)
            axes[0].scatter(np.full(len(p["turning_distances"]),k), p["turning_distances"], color=colors(i), s=30)
            half = p["far_half_sensitivity_distance"]
            if half is not None:
                axes[1].scatter(k, half, color=colors(i), s=30)
        # Connect only adjacent gains when both have a single extremum.
        for a,b in ((0,1),(1,2)):
            pa,pb = [load_profile("icm", GAINS[k],seed) for k in (a,b)]
            if len(pa["turning_distances"]) == len(pb["turning_distances"]) == 1:
                axes[0].plot([a,b], [pa["turning_distances"][0],pb["turning_distances"][0]], color=colors(i), alpha=.5, lw=.7)
    for ax in axes:
        ax.set_xticks(range(3), [str(g) for g in GAINS])
        ax.set(xlabel="Input gain", ylabel="Physical distance", ylim=(.8,16.5))
        ax.axhspan(2,4,color=".8",alpha=.2)
        ax.grid(alpha=.15)
    axes[0].set_title("ICM extrema: missing points mean no detected extremum\nLines match a seed only across adjacent gains")
    axes[1].set_title("ICM monotone curves: slope falls to half its peak\nFolded curves excluded; missing = not reached in [1,16]")
    save(fig,"turning_and_taper_locations")


def frozen_control(c, seeds):
    fig, axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    max_error = 0.
    d = torch.exp(torch.linspace(0,np.log(16),1024))
    for ax, seed in zip(axes,(130,131,136)):
        state = trials.load(ORIGINAL/"encoders"/f"icm-{seed}.pt")["encoder"]
        base = ScaledEncoder(4)
        base.load_state_dict(state)
        with torch.no_grad():
            baseline = base(trials.behavior.sensor(d,c))
            mean, scale = baseline.mean(), baseline.std(unbiased=False)
            for gain in GAINS:
                changed = ScaledEncoder(gain)
                changed.load_state_dict(state)
                y = changed(trials.behavior.sensor(d,c))
                expected = base(trials.behavior.sensor(4*d/gain,c))
                error = float((y-expected).abs().max())
                max_error = max(max_error,error)
                torch.testing.assert_close(y,expected,atol=1e-5,rtol=1e-5)
                ax.plot(d,(y-mean)/scale,color=COLORS[gain],label=f"Gain {gain}")
        axis_distance(ax)
        ax.set(title=f"Fixed trained weights · seed {seed}",ylabel="Common affine display scale")
    legend(fig,axes[0])
    fig.suptitle("Frozen weights must shift horizontally: h_g(d) = h_4(4d/g)\nThis is an algebraic control, not evidence that retraining must shift the fold")
    save(fig,"frozen_weight_rescaling")
    return max_error


def tables(seeds):
    rows = []
    for seed in seeds:
        data = trials.load(ORIGINAL/"sources"/f"icm-{seed}.pt")["data"]
        distance = trials.distances(data["x"]).numpy()
        for gain in GAINS:
            for arm in ("initial","icm","ou","balanced_log") if gain == 4 else ("initial","icm","ou"):
                p = load_profile(arm,gain,seed)
                rows.append(dict(seed=seed,arm=arm,gain=gain,
                    folded=p["folded"], turning_distances=p["turning_distances"],
                    peak_sensitivity_distance=p["peak_sensitivity_distance"],
                    far_half_sensitivity_distance=p["far_half_sensitivity_distance"],
                    r2=p["r2"], icm_source_fraction_d_le_2=float(np.mean(distance<=2)),
                    icm_source_fraction_d_le_4=float(np.mean(distance<=4)),
                    derivative_sum_max_error=p["derivative_sum_max_error"]))
    trials.atomic_json(OUT/"metrics.json",rows)
    with (OUT/"metrics.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for arm in ("icm","ou","initial"):
        for gain in GAINS:
            subset = [r for r in rows if r["arm"]==arm and r["gain"]==gain]
            roots = [d for r in subset for d in r["turning_distances"]]
            halves = [r["far_half_sensitivity_distance"] for r in subset if r["far_half_sensitivity_distance"] is not None]
            summaries.append(dict(arm=arm,gain=gain,folded=sum(r["folded"] for r in subset),
                extrema=len(roots), median_turn=float(np.median(roots)) if roots else None,
                median_half_sensitivity=float(np.median(halves)) if halves else None,
                median_peak_sensitivity=float(np.median([r["peak_sensitivity_distance"] for r in subset])),
                mean_r2=float(np.mean([r["r2"] for r in subset]))))
    trials.atomic_json(OUT/"summary.json",summaries)
    return rows,summaries


def main():
    torch.set_num_threads(1)
    settings = json.loads((OUT/"settings.json").read_text())
    seeds = settings["seeds"]
    if not all((OUT/f"completed-{s}.json").exists() for s in seeds):
        raise RuntimeError("Gain cohort is incomplete")
    c = trials.behavior.Config(**settings["config"])
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    rows,summaries = tables(seeds)
    input_scale()
    baseline_grids(seeds)
    gain_grids(seeds)
    unit_diagnostics()
    turn_zoom()
    locations(seeds)
    error = frozen_control(c,seeds)
    trials.atomic_json(OUT/"checks.json",dict(
        passed=True,profiles=len(rows),new_encoders=40,original_encoders=30,
        derivative_reconstruction_max_error=max(r["derivative_sum_max_error"] for r in rows),
        frozen_rescaling_max_raw_error=error,
        baseline_reproduction=json.loads((OUT/"baseline-reproduction.json").read_text()),
        note="No new planning fits or new independent validation seeds; exploratory same-data controls"))
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()

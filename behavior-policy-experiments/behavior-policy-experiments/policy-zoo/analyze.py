"""Aggregate zoo runs: reproduction check, paired statistics, report, figures.

All tests follow PROTOCOL.md. Seeds are the pairing unit: every condition
shares data-seed-specific encoder initialization and minibatch indices.
"""

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zoo

HERE = Path(__file__).resolve().parent
RELIABILITY = HERE.parent / "reliability"
METRIC = "common_grid_r2"
REPRODUCE = {
    "handover-random": ("output-repairs", "random"),
    "handover-icm": ("output-repairs", "icm"),
    "pure-ou": ("output-repairs", "ou"),
    "handover-rnd": ("output-policies", "rnd"),
    "handover-disagreement": ("output-policies", "disagreement"),
    "handover-count": ("output-policies", "count"),
    "handover-action_balanced": ("output-policies", "action_balanced"),
}
NAMES = {
    "random": "Random", "icm": "ICM", "rnd": "RND", "disagreement": "Disagreement",
    "count": "Count coverage", "action_balanced": "Action-balanced",
    "sticky": "Sticky random", "lazy": "Lazy random", "biased": "Approach-biased",
    "levy": "Lévy flight", "sweep": "Edge-to-edge sweep", "homing": "Homing (d≈4)",
    "task": "A2C task (d≈4)", "ou_actions": "OU-noise actions",
    "ou_actions_lp_w0.1": "Low-pass OU actions", "ou_actions_hp_w0.3": "High-pass OU actions",
    "ou_actions_hp2_w3": "2nd-order high-pass OU actions",
}


# -------------------------------------------------------------- statistics
def bootstrap_ci(values, resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (resamples, len(values))).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_flip_p(differences):
    """Exact two-sided paired permutation test of the mean difference."""
    d = np.asarray(differences, float)
    if np.allclose(d, 0):
        return 1.0
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))), dtype=np.int8) \
        if len(d) <= 14 else None
    observed = abs(d.mean())
    if signs is None:
        # Enumerate in chunks over the leading bits to bound memory.
        tail = np.array(list(itertools.product([1, -1], repeat=14)), dtype=np.int8) @ d[-14:]
        count = 0
        for head in itertools.product([1, -1], repeat=len(d) - 14):
            totals = np.dot(head, d[:-14]) + tail
            count += int((np.abs(totals) / len(d) >= observed - 1e-12).sum())
        return count / 2 ** len(d)
    return float((np.abs(signs @ d) / len(d) >= observed - 1e-12).mean())


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired binary outcomes."""
    b10 = int(np.sum(a & ~b))
    b01 = int(np.sum(~a & b))
    n = b10 + b01
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b10, b01) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def ranks(x):
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for value in np.unique(x):
        tied = x == value
        r[tied] = r[tied].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


# ------------------------------------------------------------------ loading
def load(out):
    runs = {}
    for path in (out / "runs").glob("*.json"):
        r = json.loads(path.read_text())
        runs.setdefault(r["condition"], {})[r["seed"]] = r
    return runs


def check_reproduction(runs):
    checked = 0
    for label, (folder, condition) in REPRODUCE.items():
        saved = json.loads((RELIABILITY / folder / "metrics.json").read_text())["runs"]
        for old in saved:
            if old["condition"] != condition or old["seed"] not in runs.get(label, {}):
                continue
            if old["init_seed"] != old["seed"]:
                continue
            new = runs[label][old["seed"]]["metrics"]
            for key, value in old["metrics"].items():
                if not np.isclose(new[key], value, atol=1e-7, rtol=0):
                    raise RuntimeError(f"Not reproduced: {label} seed {old['seed']} {key}")
            checked += 1
    return checked


def values(runs, label, seeds, key=METRIC):
    return np.array([runs[label][s]["metrics"][key] for s in seeds])


def folds(runs, label, seeds):
    return np.array([runs[label][s]["metrics"]["reversed_grid_fraction"] > 0 for s in seeds])


def describe(runs, label, seeds):
    v = values(runs, label, seeds)
    f = folds(runs, label, seeds)
    stats = {k: float(np.mean([runs[label][s]["source_stats"][k] for s in seeds]))
             for k in runs[label][seeds[0]]["source_stats"] if k != "sha256"}
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "median": float(np.median(v)), "ci95": bootstrap_ci(v),
            "broader_ou_r2": float(values(runs, label, seeds, "reference_ou_r2").mean()),
            "first_update_r2": float(np.mean([runs[label][x]["history"][0][METRIC] for x in seeds])),
            "folds": int(f.sum()), "fold_ci95": wilson(int(f.sum()), len(f)),
            "source": stats}


def paired(runs, labels, reference, seeds):
    rows = []
    for label in labels:
        if label == reference:
            continue
        d = values(runs, label, seeds) - values(runs, reference, seeds)
        rows.append({"condition": label, "reference": reference,
                     "mean_difference": float(d.mean()), "ci95": bootstrap_ci(d),
                     "wins": int((d > 0).sum()), "p": sign_flip_p(d)})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


def fold_tests(runs, labels, reference, seeds):
    rows = [{"condition": label, "reference": reference,
             "p": mcnemar_p(folds(runs, label, seeds), folds(runs, reference, seeds))}
            for label in labels if label != reference]
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


# ----------------------------------------------------------------- report
def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def star(p):
    return " **" if p < 0.05 else ""


def build(out):
    runs = load(out)
    seeds = sorted(set.intersection(*(set(v) for v in runs.values())))
    quick = json.loads((out / "config.json").read_text())["quick"]
    reproduced = 0 if quick else check_reproduction(runs)
    summary = {"seeds": seeds, "reproduced_reliability_runs": reproduced,
               "conditions": {k: describe(runs, k, seeds) for k in sorted(runs)},
               "tests": {}}
    physical = {p: [f"{p}-{n}" for n in zoo.PHYSICAL] for p in ["handover", "pure"]}
    tests = summary["tests"]
    for protocol, labels in physical.items():
        for ref in ["random", "icm"]:
            tests[f"{protocol}_vs_{ref}"] = paired(runs, labels, f"{protocol}-{ref}", seeds)
        tests[f"{protocol}_vs_fixed_ou"] = paired(runs, labels, "pure-ou", seeds)
        tests[f"{protocol}_folds_vs_icm"] = fold_tests(runs, labels, f"{protocol}-icm", seeds)
    rows = []
    for name in zoo.PHYSICAL:
        d = values(runs, f"handover-{name}", seeds) - values(runs, f"pure-{name}", seeds)
        rows.append({"condition": name, "mean_difference": float(d.mean()),
                     "ci95": bootstrap_ci(d), "p": sign_flip_p(d)})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    tests["handover_minus_pure"] = rows
    rows = []
    for name in zoo.LATENT_FILTERS:
        if name == "ou_traj":
            continue
        d = values(runs, f"pure-{name}", seeds) - values(runs, f"pure-iid_{name}", seeds)
        rows.append({"condition": name, "reference": f"iid_{name}",
                     "mean_difference": float(d.mean()), "ci95": bootstrap_ci(d),
                     "p": sign_flip_p(d)})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    tests["filtered_vs_matched_iid"] = rows
    long = [k[len("pure_long-"):] for k in runs if k.startswith("pure_long-")]
    rows = []
    for name in long:
        d = values(runs, f"pure_long-{name}", seeds) - values(runs, f"pure-{name}", seeds)
        rows.append({"condition": name, "mean_difference": float(d.mean()),
                     "ci95": bootstrap_ci(d), "p": sign_flip_p(d)})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    tests["long_minus_ordinary"] = rows
    tests["latent_vs_fixed_ou"] = paired(
        runs, [f"pure-{n}" for n in zoo.LATENT], "pure-ou", seeds)
    pure = [summary["conditions"][f"pure-{n}"] for n in zoo.PHYSICAL]
    summary["spearman_pure_r2"] = {
        k: spearman([c["source"][k] for c in pure], [c["mean"] for c in pure])
        for k in ["size_bin_entropy", "size_bin_coverage", "max_size_bin_mass",
                  "gaussian_cdf_distance", "lag_correlation", "stay_fraction", "mean_abs_step"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(out, summary, runs, seeds)
    report(out, summary)
    print(f"{len(runs)} conditions x {len(seeds)} seeds; reproduced {reproduced} saved runs")


def report(out, s):
    C, T = s["conditions"], s["tests"]
    n = len(s["seeds"])
    by = lambda rows: {r["condition"]: r for r in rows}
    lines = ["# Policy zoo results", "",
             f"Data seeds {s['seeds'][0]}–{s['seeds'][-1]} ({n} per condition), paired by seed. "
             f"Mean shared-range $R^2$ with 95% bootstrap CI; folds = runs with any reversed grid step. "
             f"p-values: exact paired sign-flip tests, Holm-corrected within each column (see [PROTOCOL.md](../PROTOCOL.md)). "
             f"**Bold** marks Holm p < 0.05.", "",
             f"Reproduction: {s['reproduced_reliability_runs']} earlier reliability runs (seeds 8–12) were regenerated and match every saved metric to 1e-7.", "",
             f"Reference floor: every condition shares the seed's encoder initialization, and after a single update fixed-OU runs already have mean $R^2$ = {C['pure-ou']['first_update_r2']:.4f}. The tanh encoder starts close to affine in log-distance for most seeds, so scores largely measure whether training on a source keeps, improves or destroys that starting map.", ""]
    for protocol, title in [("pure", "Policy data only (pure)"),
                            ("handover", "Policy data, then OU handover (original protocol)")]:
        vr, vi, vo = (by(T[f"{protocol}_vs_random"]), by(T[f"{protocol}_vs_icm"]),
                      by(T[f"{protocol}_vs_fixed_ou"]))
        fi = by(T[f"{protocol}_folds_vs_icm"])
        lines += [f"## {title}", "",
                  "| Policy | Mean $R^2$ [95% CI] | Median | Folds | Δ vs random (p) | Δ vs ICM (p) | Δ vs fixed OU (p) | Folds vs ICM p |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        order = sorted(zoo.PHYSICAL, key=lambda k: -C[f"{protocol}-{k}"]["mean"])
        for name in order:
            label = f"{protocol}-{name}"
            c = C[label]

            def cell(table):
                if label not in table:
                    return "—"
                r = table[label]
                text = f"{r['mean_difference']:+.4f} ({fmt_p(r['p_holm'])})"
                return f"**{text}**" if r["p_holm"] < 0.05 else text
            fold_p = fmt_p(fi[label]["p_holm"]) if label in fi else "—"
            lines.append(f"| {NAMES[name]} | {c['mean']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] | "
                         f"{c['median']:.4f} | {c['folds']}/{c['n']} | {cell(vr)} | {cell(vi)} | {cell(vo)} | {fold_p} |")
        lines.append("")
    lines += ["## Does the OU handover help?", "",
              "| Policy | Handover − pure [95% CI] | Holm p |", "| --- | ---: | ---: |"]
    for r in sorted(T["handover_minus_pure"], key=lambda r: -r["mean_difference"]):
        lines.append(f"| {NAMES[r['condition']]} | {r['mean_difference']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {fmt_p(r['p_holm'])}{star(r['p_holm'])} |")
    lines += ["", "## Low-pass and high-pass OU (exact Gaussian, pure)", "",
              "Filtered trajectories vs independent pairs with the same lag-1 correlation ρ.", "",
              "| Source | ρ | Filtered $R^2$ [95% CI] | Folds | Matched i.i.d. $R^2$ | Folds | Filtered − i.i.d. (Holm p) | Δ filtered vs fixed OU (Holm p) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fm, fo = by(T["filtered_vs_matched_iid"]), by(T["latent_vs_fixed_ou"])
    for name in zoo.LATENT_FILTERS:
        c = C[f"pure-{name}"]
        rho = zoo.source_rho(name)
        if name == "ou_traj":
            matched, diff = C["pure-ou"], "—"
        else:
            matched, r = C[f"pure-iid_{name}"], fm[name]
            diff = f"{r['mean_difference']:+.4f} ({fmt_p(r['p_holm'])})"
        o = fo[f"pure-{name}"]
        lines.append(f"| {name} | {rho:+.3f} | {c['mean']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] | {c['folds']}/{c['n']} | "
                     f"{matched['mean']:.4f} | {matched['folds']}/{matched['n']} | {diff} | {o['mean_difference']:+.4f} ({fmt_p(o['p_holm'])}) |")
    c = C["pure-ou"]
    lines += [f"| fixed OU (i.i.d. pairs) | +0.900 | {c['mean']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] | {c['folds']}/{c['n']} | — | — | — | — |", "",
              "## Exploratory: twice the budget (added after the main results)", "",
              "For ρ < 0 the implemented loss is lower for a Gaussianized fold than for any rescaled linear map (see README). Do 8,000 updates find it?", "",
              "| Source | 4,000 updates | 8,000 updates | Folds (8,000) | Long − ordinary (Holm p) |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in T["long_minus_ordinary"]:
        a, b = C[f"pure-{r['condition']}"], C[f"pure_long-{r['condition']}"]
        lines.append(f"| {r['condition']} | {a['mean']:.4f} | {b['mean']:.4f} | {b['folds']}/{b['n']} | {r['mean_difference']:+.4f} ({fmt_p(r['p_holm'])}) |")
    lines += ["", "## Source statistics (pure protocol, physical policies)", "",
              "Descriptive only; computed with simulator labels that fitting never sees.", "",
              "| Policy | Size-bin entropy | Max bin mass | In sensor range | Gaussian-CDF dist. | Lag-1 corr. | Stay fraction |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in sorted(zoo.PHYSICAL, key=lambda k: -C[f"pure-{k}"]["mean"]):
        x = C[f"pure-{name}"]["source"]
        lines.append(f"| {NAMES[name]} | {x['size_bin_entropy']:.3f} | {x['max_size_bin_mass']:.3f} | {x['fraction_in_sensor_range']:.3f} | "
                     f"{x['gaussian_cdf_distance']:.3f} | {x['lag_correlation']:.4f} | {x['stay_fraction']:.3f} |")
    lines += ["", "Spearman correlation of a policy's mean pure $R^2$ with its source statistic (17 policies, descriptive):", "",
              "| Statistic | Spearman ρ |", "| --- | ---: |"]
    lines += [f"| {k} | {v:+.3f} |" for k, v in s["spearman_pure_r2"].items()]
    lines += ["", "![Policies](policies.png)", "", "![Filtered OU](filtered_ou.png)", "",
              "Full per-run metrics are in `runs/`; aggregated numbers and every test are in `summary.json`.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


# ------------------------------------------------------------------ figures
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"


def plot(out, s, runs, seeds):
    C = s["conditions"]
    order = sorted(zoo.PHYSICAL, key=lambda k: C[f"pure-{k}"]["mean"])
    fig, ax = plt.subplots(figsize=(9, 7.5), layout="constrained")
    for offset, protocol, color in [(-0.17, "pure", ORANGE), (0.17, "handover", BLUE)]:
        for i, name in enumerate(order):
            label = f"{protocol}-{name}"
            v = values(runs, label, seeds)
            jitter = np.linspace(-0.06, 0.06, len(v))
            ax.scatter(v, i + offset + jitter, s=9, color=color, alpha=0.35, linewidths=0)
            c = C[label]
            ax.errorbar(c["mean"], i + offset, xerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]],
                        fmt="o", ms=6, color=color, capsize=0, lw=2,
                        label={"pure": "Policy data only", "handover": "Policy → OU handover"}[protocol] if i == 0 else None)
    ou = C["pure-ou"]["mean"]
    ax.axvline(ou, color=GRAY, lw=1, ls="--")
    ax.text(ou, len(order) - 0.4, " fixed OU", color="#555", fontsize=9, va="bottom", ha="right")
    ax.set_yticks(range(len(order)), [NAMES[k] for k in order])
    ax.set_xlabel("Shared-range log-distance $R^2$ (dots: seeds; marker: mean, 95% bootstrap CI)")
    ax.set_title(f"Recovery by behavior policy ({len(seeds)} paired seeds)", loc="left")
    ax.grid(axis="x", color="#e5e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out / "policies.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name in zoo.LATENT_FILTERS:
        rho = zoo.source_rho(name)
        for label, color, dx in [(f"pure-{name}", BLUE, 0.004),
                                 ("pure-ou" if name == "ou_traj" else f"pure-iid_{name}", GRAY, -0.004)]:
            c = C[label]
            ax.errorbar(rho + dx, c["mean"], yerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]],
                        fmt="o", ms=6, color=color, lw=2, capsize=0)
        # Label the crowded rho ~ 1 cluster from below with leader lines.
        crowded = {"ou_traj": (-75, -62), "ou_lp_w1": (-45, -80), "ou_lp_w0.3": (-15, -98),
                   "ou_lp_w0.1": (15, -116)}
        dx, dy = crowded.get(name, (5, 6))
        ax.annotate(name.replace("ou_", ""), (rho, C[f"pure-{name}"]["mean"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, color="#444",
                    va="center", arrowprops={"arrowstyle": "-", "color": "#bbb", "lw": 0.8}
                    if name in crowded else None)
    ax.errorbar([], [], fmt="o", color=BLUE, label="Filtered OU trajectories")
    ax.errorbar([], [], fmt="o", color=GRAY, label="Independent pairs, same ρ")
    ax.axvline(0, color=GRAY, lw=1, ls=":")
    ax.set_xlabel("Lag-1 correlation ρ of the training pairs")
    ax.set_ylabel("Shared-range $R^2$ (mean, 95% CI)")
    ax.set_title("Low-pass raises ρ toward 1, high-pass lowers it (below 0 for 2nd order)", loc="left")
    ax.grid(color="#e5e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "filtered_ou.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()

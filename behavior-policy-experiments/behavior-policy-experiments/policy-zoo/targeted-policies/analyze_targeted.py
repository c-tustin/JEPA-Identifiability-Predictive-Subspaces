"""Matched-seed comparisons and a concise report of every targeted policy."""

import argparse
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from run_targeted import OUT, OLD, DISCOVERY, CONFIRMATION, targeted_policies, zoo


METRIC = "common_grid_r2"


def read(stage, protocol, name, seed):
    path = OUT / "runs" / stage / f"{protocol}-{name}_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    if stage == "discovery":
        return json.loads((OLD / "runs" / f"{protocol}-{name}_seed{seed}.json").read_text())
    raise FileNotFoundError(path)


def values(stage, protocol, name, seeds):
    return np.array([read(stage, protocol, name, seed)["metrics"][METRIC] for seed in seeds])


def bootstrap95(values):
    rng = np.random.default_rng(20261105)
    means = rng.choice(values, (20000, len(values))).mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def holm(rows):
    order = np.argsort([row["p"] for row in rows])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p"]))
        rows[index]["p_holm"] = running
    return rows


def comparison(stage, protocol, name, reference_protocol, reference_name, seeds):
    difference = values(stage, protocol, name, seeds) - values(stage, reference_protocol, reference_name, seeds)
    wins = int((difference > 1e-10).sum())
    losses = int((difference < -1e-10).sum())
    p = 1.0 if wins + losses == 0 else float(stats.binomtest(wins, wins + losses, .5).pvalue)
    return dict(
        protocol=protocol, policy=name, reference_protocol=reference_protocol, reference=reference_name,
        mean_difference=float(difference.mean()), paired_bootstrap95=bootstrap95(difference),
        wins=wins, losses=losses, ties=len(seeds) - wins - losses, p=p,
        seeds=list(seeds), differences=difference.tolist(),
    )


def describe(stage, protocol, name, seeds):
    records = [read(stage, protocol, name, seed) for seed in seeds]
    score = values(stage, protocol, name, seeds)
    source = [record.get("source_statistics", record.get("source_stats")) for record in records]
    return dict(
        protocol=protocol, policy=name, n=len(seeds), new=name in targeted_policies.POLICIES,
        mean=float(score.mean()), sd=float(score.std(ddof=1)), median=float(np.median(score)),
        minimum=float(score.min()), ci95=bootstrap95(score),
        folds=sum(record["metrics"]["reversed_grid_fraction"] > 0 for record in records),
        broader_ou_r2=float(np.mean([record["metrics"]["reference_ou_r2"] for record in records])),
        mean_gaussian_cdf_distance=float(np.mean([s["gaussian_cdf_distance"] for s in source])),
        mean_stay_fraction=float(np.mean([s["stay_fraction"] for s in source])),
        mean_size_bin_entropy=float(np.mean([s["size_bin_entropy"] for s in source])),
        mean_absolute_log_step=float(np.mean([s["mean_abs_step"] for s in source])),
        mean_log_distance_sd=0.7 * float(np.mean([s["z_std"] for s in source])),
    )


def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def pformat(p):
    return "<0.0001" if p < .0001 else f"{p:.4f}"


def build(stage):
    seeds = DISCOVERY if stage == "discovery" else CONFIRMATION
    if stage == "discovery":
        candidates = [(p, n) for p in ("pure", "handover") for n in targeted_policies.POLICIES]
        combinations = candidates + [(p, n) for p in ("pure", "handover") for n in zoo.PHYSICAL]
        combinations += [("pure", "ou")]
    else:
        selected = json.loads((OUT / "selection.json").read_text())["selected"]
        candidates = [(p, n) for p, names in selected.items() for n in names]
        combinations = candidates + [("pure", "ou"), ("pure", "count"), ("handover", "biased"),
                                     ("pure", "random"), ("handover", "random")]
    descriptions = [describe(stage, p, n, seeds) for p, n in combinations]
    tests = {}
    for family in ("fixed_ou", "random", "best_old"):
        rows = []
        for protocol, name in candidates:
            if family == "fixed_ou":
                ref_protocol, ref_name = "pure", "ou"
            elif family == "random":
                ref_protocol, ref_name = protocol, "random"
            else:
                ref_protocol = protocol
                ref_name = "count" if protocol == "pure" else "biased"
            rows.append(comparison(stage, protocol, name, ref_protocol, ref_name, seeds))
        tests[family] = holm(rows)
    for row in tests["fixed_ou"]:
        # A two-sided rejection can favor either policy. Require the direction
        # of the wins as well as a positive observed mean before calling a win.
        row["beats_ou_by_fixed_rule"] = (
            row["mean_difference"] > 0
            and row["wins"] > row["losses"]
            and row["p_holm"] < .05
        )
    result = dict(stage=stage, seeds=list(seeds), conditions=descriptions, tests=tests,
                  inference="Discovery is exploratory. Confirmation uses fresh seeds and frozen selection. Sign tests target non-tie win probabilities; intervals summarize mean differences.")
    (OUT / f"{stage}-summary.json").write_text(json.dumps(result, indent=2) + "\n")

    rows_per_panel = max(sum(r["protocol"] == p for r in tests["fixed_ou"]) for p in ("pure", "handover"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 1.5 + .55 * rows_per_panel), layout="constrained")
    for ax, protocol in zip(axes, ("pure", "handover")):
        chosen = sorted([r for r in tests["fixed_ou"] if r["protocol"] == protocol], key=lambda r: r["mean_difference"])
        for i, row in enumerate(chosen):
            low, high = row["paired_bootstrap95"]
            color = "#0072B2" if row["beats_ou_by_fixed_rule"] else ".5"
            ax.plot([low, high], [i, i], color=color, lw=2)
            ax.scatter(row["mean_difference"], i, color=color, s=35)
        ax.set_yticks(range(len(chosen)), [r["policy"].replace("gaussian_", "") for r in chosen])
        ax.set_ylim(-.6, len(chosen) - .4)
        ax.axvline(0, color=".3", ls="--")
        ax.set(xlabel="Paired mean R² difference from fixed OU", title=protocol.title())
        ax.grid(alpha=.15)
    fig.suptitle(f"{stage.title()}: same-seed comparisons with OU\nLines: unadjusted paired-bootstrap 95% intervals. Blue: positive mean, majority wins and Holm sign-test p < 0.05.")
    save(fig, f"{stage}_vs_ou")

    if stage == "discovery":
        fig, axes = plt.subplots(1, 2, figsize=(13, 10), layout="constrained")
        ou_mean = next(r["mean"] for r in descriptions if r["policy"] == "ou")
        for ax, protocol in zip(axes, ("pure", "handover")):
            chosen = sorted([r for r in descriptions if r["protocol"] == protocol and r["policy"] != "ou"], key=lambda r: r["mean"])
            for i, row in enumerate(chosen):
                color = "#0072B2" if row["new"] else ".6"
                ax.plot(row["ci95"], [i, i], color=color, lw=2)
                ax.scatter(row["mean"], i, color=color, s=22)
            ax.set_yticks(range(len(chosen)), [r["policy"].replace("gaussian_", "G: ") for r in chosen], fontsize=8)
            ax.axvline(ou_mean, color="#D55E00", ls="--", label="Fixed OU mean")
            ax.set(xlabel="Mean shared-grid R²", title=protocol.title(), xlim=(-.03, 1.01))
            ax.grid(alpha=.15)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("New controllers compared with the entire physical-policy zoo\nBlue = new Gaussian-targeted policies. Gray = original policies. Each row uses the same 20 seeds.")
        save(fig, "all_policy_comparison")

    lines = [f"# {stage.title()} results", "",
             f"Seeds {seeds[0]}–{seeds[-1]}, paired by seed. R² measures representation recovery, not goal-reaching. Every candidate is retained.", "",
             "## Matched comparisons with fixed OU", "",
             "| Protocol | Policy | Mean Δ R² [paired 95% CI] | Wins / losses | Holm sign p | Pass fixed rule? |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in sorted(tests["fixed_ou"], key=lambda r: (r["protocol"], -r["mean_difference"])):
        low, high = row["paired_bootstrap95"]
        lines.append(f"| {row['protocol']} | {row['policy']} | {row['mean_difference']:+.6f} [{low:+.6f}, {high:+.6f}] | {row['wins']} / {row['losses']} | {pformat(row['p_holm'])} | {'Yes' if row['beats_ou_by_fixed_rule'] else 'No'} |")
    for protocol in ("pure", "handover"):
        lines += ["", f"## {protocol.title()} rankings", "",
                  "| Policy | Mean R² [95% CI] | SD | Folds | Broader OU R² | Stay fraction | Gaussian discrepancy |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in sorted([r for r in descriptions if r["protocol"] == protocol], key=lambda r: -r["mean"]):
            low, high = row["ci95"]
            lines.append(f"| {row['policy']}{' (new)' if row['new'] else ''} | {row['mean']:.6f} [{low:.6f}, {high:.6f}] | {row['sd']:.6f} | {row['folds']}/{len(seeds)} | {row['broader_ou_r2']:.6f} | {row['mean_stay_fraction']:.3f} | {row['mean_gaussian_cdf_distance']:.4f} |")
    lines += ["", "All per-seed differences, comparisons against random and the best old source, and source diagnostics are in the stage summary JSON.", "",
              "Confidence intervals are unadjusted mean summaries. Sign tests concern win probability; they are a different null. Discovery uses old seeds. Confirmation uses fresh seeds and four selection-frozen comparisons.", "",
              f"![Paired comparisons](figures/{stage}_vs_ou.png)"]
    (OUT / f"{stage.upper()}_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(stage, [(r["protocol"], r["policy"], r["mean_difference"], r["p_holm"]) for r in tests["fixed_ou"]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "discovery", "confirmation"), default="all")
    args = parser.parse_args()
    for stage in ("discovery", "confirmation"):
        if args.stage in ("all", stage):
            assert (OUT / f"{stage}-complete.json").exists(), f"{stage} is not complete"
            build(stage)


if __name__ == "__main__":
    main()

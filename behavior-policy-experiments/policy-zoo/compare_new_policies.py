"""Overview of all 13 additions and campaign-wide fresh-seed comparisons."""

import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import zoo
import targeted_policies
import broad_policies

HERE = Path(__file__).resolve().parent
OUT = HERE / "new-policy-results"
STUDIES = ("targeted-policies", "broad-policies")
NAMES = {
    "gaussian_walk": "Gaussian walk",
    "gaussian_log_diffusion": "Log-scaled Gaussian walk",
    "gaussian_wide_diffusion": "Wider Gaussian walk",
    "gaussian_target_fast": "Fast size targets",
    "gaussian_target_slow": "Slow size targets",
    "gaussian_deficit": "Gaussian-weighted coverage",
    "gaussian_start_random": "Gaussian start, then random",
    "guarded_bias": "Guarded biased walk",
    "gentle_bias": "Gentle biased walk",
    "adaptive_bias": "Size-feedback bias",
    "guarded_sweep": "Guarded sweeps",
    "bias_sweep_mix": "Sweep/bias mixture",
    "no_escort_log_diffusion": "Gaussian walk without initial steering",
}


def holm_campaign(rows):
    running = 0.0
    for rank, index in enumerate(np.argsort([r["p"] for r in rows])):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p"]))
        rows[index]["p_holm_campaign"] = running
    for row in rows:
        higher = row["mean_difference"] > 0 and row["wins"] > row["losses"]
        lower = row["mean_difference"] < 0 and row["losses"] > row["wins"]
        row["conclusion"] = (
            "higher" if higher and row["p_holm_campaign"] < .05 else
            "lower" if lower and row["p_holm_campaign"] < .05 else
            "inconclusive"
        )
    return rows


def pformat(value):
    return "<0.0001" if value < .0001 else f"{value:.4f}"


def main():
    OUT.mkdir(exist_ok=True)
    loaded = {}
    for study in STUDIES:
        folder = HERE / study / "output"
        loaded[study] = {stage: json.loads((folder / f"{stage}-summary.json").read_text()) for stage in ("discovery", "confirmation")}
    combined = {}
    for family in ("fixed_ou", "best_old", "random"):
        rows = []
        for study in STUDIES:
            summary = loaded[study]["confirmation"]
            for row in summary["tests"][family]:
                result = dict(row, study=study)
                condition = next(c for c in summary["conditions"] if c["protocol"] == row["protocol"] and c["policy"] == row["policy"])
                result.update(mean_r2=condition["mean"], folds=condition["folds"], minimum_r2=condition["minimum"])
                rows.append(result)
        combined[family] = holm_campaign(rows)
    winning_ou = [r for r in combined["fixed_ou"] if r["conclusion"] == "higher"]
    winning_old = [r for r in combined["best_old"] if r["conclusion"] == "higher"]
    discovery = {}
    for study in STUDIES:
        for row in loaded[study]["discovery"]["conditions"]:
            if row["new"]:
                discovery[row["protocol"], row["policy"]] = row
    result = dict(
        studies=STUDIES, confirmation_tests=combined,
        confirmed_higher_than_ou=[dict(protocol=r["protocol"], policy=r["policy"], study=r["study"]) for r in winning_ou],
        confirmed_higher_than_best_old=[dict(protocol=r["protocol"], policy=r["policy"], study=r["study"]) for r in winning_old],
        inference="Holm across all eight fresh-seed comparisons per reference family. No pooling discovery into confirmation; positive mean and majority wins required. Sign tests target win probability, not mean difference.",
    )
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    colors = {"higher": "#0072B2", "lower": "#D55E00", "inconclusive": ".5"}
    for ax, protocol in zip(axes, ("pure", "handover")):
        rows = sorted([r for r in combined["fixed_ou"] if r["protocol"] == protocol], key=lambda r: r["mean_difference"])
        for index, row in enumerate(rows):
            lo, hi = row["paired_bootstrap95"]
            color = colors[row["conclusion"]]
            ax.plot([lo, hi], [index, index], color=color, lw=2)
            ax.scatter(row["mean_difference"], index, color=color, s=35)
        ax.set_yticks(range(len(rows)), [NAMES[r["policy"]] for r in rows], fontsize=9)
        ax.set_ylim(-.6, len(rows) - .4)
        ax.axvline(0, color=".3", ls="--")
        ax.set(xlabel="Paired mean R² difference from that block's fixed OU", title=protocol.title())
        ax.grid(alpha=.15)
    fig.suptitle("Fresh-seed results across both new-policy studies\nBlue = consistently higher; orange = lower; gray = inconclusive. Holm across all 8 comparisons. Lines: unadjusted mean 95% CIs.")
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"fresh_vs_ou.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines = ["# New behavior policies: short results", "",
             "**Scope:** these physical-policy extensions go beyond Klindt's exact Gaussian-OU assumptions. Gains below are finite-training representation results, not new identifiability guarantees. [Assumption boundary](../ASSUMPTION_BOUNDARY.md).", "",
             "Added **13 controllers** to the distance-only zoo. Each uses the same physical actions and collection budget. We ran **520 discovery fits** and **400 fresh-seed fits**, plus four original-path reproduction gates. The original 17-policy catalog remains reproducible; the expanded catalog has 30 physical policies.", ""]
    if winning_ou:
        names = ", ".join(NAMES[r["policy"]] + (" → OU" if r["protocol"] == "handover" else " (pure)") for r in winning_ou)
        lines += [f"**Confirmed improvements over fixed OU:** {names}.", ""]
    else:
        lines += ["**No new candidate consistently improves on fixed OU in confirmation.**", ""]
    if not any(r["protocol"] == "pure" for r in winning_ou):
        lines += ["No selected policy beats fixed OU using its own data alone. A handover win includes the original final OU training phase.", ""]
    if not winning_old:
        lines += ["**None shows a consistent improvement over the best old same-protocol reference** after the combined correction. Winning against OU alone does not establish a new best zoo policy.", ""]
    lines += ["## Fresh comparisons", "",
              "| Policy | Training | Fresh seeds | Mean R² | Folds | Δ vs OU | Wins / 20 | Combined Holm p | Result |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in sorted(combined["fixed_ou"], key=lambda r: (r["protocol"], -r["mean_difference"])):
        lines.append(f"| {NAMES[row['policy']]} | {row['protocol']} | {row['seeds'][0]}–{row['seeds'][-1]} | {row['mean_r2']:.6f} | {row['folds']}/20 | {row['mean_difference']:+.6f} | {row['wins']} | {pformat(row['p_holm_campaign'])} | {row['conclusion']} |")
    lines += ["", "![Fresh comparisons with OU](fresh_vs_ou.png)", ""]
    if winning_ou:
        lines += ["### Do the OU winners beat the old best?", "",
                  "These p-values are corrected across all eight old-reference tests; the table highlights the OU winners.", "",
                  "| Candidate | Old reference | Paired mean difference | Combined Holm p | Result |",
                  "| --- | --- | ---: | ---: | --- |"]
        for winner in winning_ou:
            row = next(r for r in combined["best_old"] if (r["study"], r["protocol"], r["policy"]) == (winner["study"], winner["protocol"], winner["policy"]))
            reference_name = "Count coverage" if row["protocol"] == "pure" else "Approach-biased → OU"
            lines.append(f"| {NAMES[row['policy']]} ({row['protocol']}) | {reference_name} | {row['mean_difference']:+.6f} | {pformat(row['p_holm_campaign'])} | {row['conclusion']} |")
        lines.append("")
    lines += [
              "The two-sided sign tests concern consistent seed-wise wins, not a zero-mean null. Each confidence interval summarizes the paired mean and is unadjusted. Every fresh seed is included; there is no successful-run subset.", "",
              "## All 13 additions", "",
              "Discovery uses the same seeds 8–27 as the old zoo. These means are exploratory; selected candidates are tested above on fresh blocks.", "",
              "| New policy | Pure mean R² | Handover mean R² |",
              "| --- | ---: | ---: |"]
    for name in NAMES:
        lines.append(f"| {NAMES[name]} | {discovery['pure',name]['mean']:.6f} | {discovery['handover',name]['mean']:.6f} |")
    pure_change = discovery["pure", "no_escort_log_diffusion"]["mean"] - discovery["pure", "gaussian_wide_diffusion"]["mean"]
    handover_change = discovery["handover", "no_escort_log_diffusion"]["mean"] - discovery["handover", "gaussian_wide_diffusion"]["mean"]
    lines += ["", "## What this comparison suggests", "",
              f"Removing the initial size-target steering from the same log-diffusion controller changes discovery R² by **{pure_change:+.4f} with pure data** but **{handover_change:+.4f} with OU handover**. The collection phase changes which views and transitions are sampled. A source's usefulness therefore depends on the later training protocol; matching a Gaussian-looking histogram alone is insufficient. This is an exploratory ablation, not a proof that width alone causes the difference.", ""]
    lines += ["", "Gaussian-targeted controllers use the known height=1/d relationship and Gaussian size goals. Broader controllers use guarded bias, sweeps, feedback, or remove the initial Gaussian steering. These are hand-coded priors; the encoder still receives only observation pairs.", "",
              "The main OU reference generates Gaussian latent pairs directly. The old physically acting **OU-noise actions** controller is a separate baseline. Both appear in the full comparison. R² is representation recovery, not a planning score.", "",
              "- [All old/new physical and Gaussian-source rankings](FULL_COMPARISON.md)",
              "- [Gaussian-targeted study: methods, failures and verification](../targeted-policies/README.md)",
              "- [Broader-coverage follow-up](../broad-policies/README.md)",
              "- [Combined test results, including best-old comparisons](summary.json)",
              "- [Inference rules across the two studies](../NEW_POLICY_INFERENCE.md)"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    full = ["# Every comparable discovery condition", "",
            "Same seeds 8–27, 4,000 encoder updates. Rankings are descriptive; fresh tests are in the short results. Physical policies and abstract Gaussian pair sources are labelled separately. The 8,000-update exploratory runs are excluded from this equal-budget table.", ""]
    for protocol in ("pure", "handover"):
        names = list(zoo.PHYSICAL) + list(NAMES)
        if protocol == "pure":
            names += list(zoo.LATENT)
        rows = []
        for name in names:
            if (protocol, name) in discovery:
                row = discovery[protocol, name]
                rows.append(dict(policy=name, mean=row["mean"], sd=row["sd"], folds=row["folds"], kind="new physical"))
            else:
                records = [json.loads((HERE / "output" / "runs" / f"{protocol}-{name}_seed{s}.json").read_text()) for s in range(8, 28)]
                scores = np.array([r["metrics"]["common_grid_r2"] for r in records])
                rows.append(dict(policy=name, mean=float(scores.mean()), sd=float(scores.std(ddof=1)),
                                 folds=sum(r["metrics"]["reversed_grid_fraction"] > 0 for r in records),
                                 kind="old physical" if name in zoo.PHYSICAL else "Gaussian pair source"))
        full += [f"## {protocol.title()}", "", "| Policy/source | Kind | Mean R² | SD | Folds |", "| --- | --- | ---: | ---: | ---: |"]
        for row in sorted(rows, key=lambda r: -r["mean"]):
            full.append(f"| {row['policy']} | {row['kind']} | {row['mean']:.6f} | {row['sd']:.6f} | {row['folds']}/20 |")
        full.append("")
    (OUT / "FULL_COMPARISON.md").write_text("\n".join(full).rstrip() + "\n")
    print("Confirmed higher than OU:", result["confirmed_higher_than_ou"])
    print("Confirmed higher than best old:", result["confirmed_higher_than_best_old"])


if __name__ == "__main__":
    main()

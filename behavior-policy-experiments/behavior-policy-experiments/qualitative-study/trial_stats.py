"""Seed-level comparisons; no goal-level pseudoreplication or test-winner filtering."""

from __future__ import annotations

import numpy as np
from scipy import stats

TIE = 1e-6


def holm_adjust(pvalues):
    """Holm family-wise adjustment, preserving input order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(order) - rank) * pvalues[index])
        adjusted[index] = min(previous, 1.0)
    return adjusted.tolist()


def paired_summary(candidate, baseline, seed=4200):
    """A negative difference is better. Sign test targets run-level win probability."""
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("Need matched one-dimensional seed-level arrays")
    difference = candidate - baseline
    if not np.isfinite(difference).all():
        raise ValueError(
            "Nonfinite trial outcomes must be audited, not silently dropped"
        )
    n = len(difference)
    wins, losses = int((difference < -TIE).sum()), int((difference > TIE).sum())
    non_ties = wins + losses
    p = float(stats.binomtest(wins, non_ties, 0.5).pvalue) if non_ties else 1.0
    if n:
        rng = np.random.default_rng(seed)
        bootstrap = difference[rng.integers(n, size=(20000, n))].mean(1)
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        mean = float(difference.mean())
    else:
        mean, interval = None, None
    t_p = None
    if n > 1 and np.std(difference, ddof=1) > 1e-12:
        t_p = float(stats.ttest_rel(candidate, baseline).pvalue)
    return dict(
        n=n,
        wins=wins,
        losses=losses,
        ties=n - non_ties,
        candidate_mean=float(candidate.mean()) if n else None,
        baseline_mean=float(baseline.mean()) if n else None,
        mean_difference=mean,
        bootstrap_95=interval,
        sign_test_p=p,
        supplementary_paired_t_p=t_p,
        differences=difference.tolist(),
    )


def curriculum_comparisons(rows):
    """Screen membership uses validation errors and grid folds, never test errors."""
    lookup = {(row["seed"], row["arm"]): row for row in rows}
    seeds = sorted({r["seed"] for r in rows})
    selected = [
        s
        for s in seeds
        if not lookup[s, "icm_to_ou"]["folded"]
        and lookup[s, "icm_to_ou"]["screen_error"]
        < lookup[s, "ou"]["screen_error"] - TIE
    ]
    results = []
    for name, cohort in (("all_trials", seeds), ("independent_screen", selected)):
        result = paired_summary(
            [lookup[s, "icm_to_ou"]["test_error"] for s in cohort],
            [lookup[s, "ou"]["test_error"] for s in cohort],
        )
        result.update(name=name, seeds=cohort)
        results.append(result)
    adjusted = holm_adjust([r["sign_test_p"] for r in results])
    for result, p in zip(results, adjusted):
        result["sign_test_holm_p"] = p
        result["significantly_better"] = bool(
            result["n"] and p < 0.05 and result["wins"] > result["losses"]
        )
        result["significantly_worse"] = bool(
            result["n"] and p < 0.05 and result["losses"] > result["wins"]
        )
    return dict(
        comparisons=results,
        screen_selected_seeds=selected,
        unit="one independently trained seed; goals are averaged within seed",
        sign_test_estimand="P(candidate has lower held-out seed-mean physical error), among non-ties",
        subgroup_estimand="Runs passing this fixed validation screen; not all runs or held-out winners",
        t_test_assumption="Normal paired differences; supplementary with small n",
        bootstrap_note="Percentile paired-seed bootstrap, 20,000 resamples; descriptive with small n",
    )

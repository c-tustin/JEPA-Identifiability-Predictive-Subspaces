"""Focused exploratory equivalence tests for recovery and grid-detected folds.

Continuous recovery: paired mean TOST, requiring all policy pairs to pass.
Fold risk: conservative exact paired test via the probability of discordance.
No representations are retrained and no grid points count as replicates.
"""
import argparse
import itertools
import json

import numpy as np
from scipy import stats

import matched_equivalence as tests


def fold_risk_equivalence(matrix, names, margin):
    """Bound |P(A folds)-P(B folds)| by P(A and B disagree).

    Test the latter probability against the margin with an exact lower-tail
    binomial test. This is sufficient, but not necessary, for risk equivalence.
    """
    n = len(matrix)
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        discordant = int(np.count_nonzero(matrix[:, i] != matrix[:, j]))
        p = float(stats.binom.cdf(discordant, n, margin))
        upper = 1. if discordant == n else float(stats.beta.ppf(.95, discordant + 1, n - discordant))
        pairs.append({'policy_a': names[i], 'policy_b': names[j],
                      'discordant_seeds': discordant, 'seeds': n,
                      'risk_difference': float(matrix[:, i].mean() - matrix[:, j].mean()),
                      'discordance_upper95': upper, 'p': p})
    return {'margin': margin, 'fold_counts': dict(zip(names, matrix.sum(0).astype(int).tolist())),
            'global_p': max(pair['p'] for pair in pairs),
            'equivalence_established': all(pair['p'] < .05 for pair in pairs),
            'pairs': pairs, 'intervals_are_pointwise': True}


def run(recovery_margin=.001, fold_margin=.05):
    results = []
    for study in tests.load_studies():
        assert study['complete']
        original = study['name'] == 'original_policies_common_reset'
        root = tests.OUT if original else tests.HERE.parent / 'matched-coverage/output'
        keys = ['affine_log_r2', 'near_r2', 'far_r2'] if original else ['log_distance_r2', 'uniform_log_grid_r2']
        blocks = []
        for seed in study['seeds']:
            blocks.append([json.loads((root / (f'zoo_{name}_reset_seed{seed}.json' if original else f'{name}_seed{seed}.json')).read_text())
                           ['metrics' if original else 'representation'] for name in study['names']])
        recovery = {}
        for key in keys:
            matrix = np.array([[row[key] for row in block] for block in blocks])
            result = tests.global_equivalence(matrix, study['names'], recovery_margin)
            # Independently check the boundary tests with SciPy's one-sample t test.
            for pair in result['pairs']:
                i, j = study['names'].index(pair['policy_a']), study['names'].index(pair['policy_b'])
                difference = matrix[:, i] - matrix[:, j]
                if np.std(difference, ddof=1) > 0:
                    p = max(stats.ttest_1samp(difference, -recovery_margin, alternative='greater').pvalue,
                            stats.ttest_1samp(difference, recovery_margin, alternative='less').pvalue)
                    assert np.isclose(p, pair['mean_tost_p'], atol=1e-14, rtol=1e-12)
            result['policy_means'] = dict(zip(study['names'], matrix.mean(0).tolist()))
            recovery[key] = result
        folds = np.array([[row['folded'] for row in block] for block in blocks], dtype=bool)
        risk = fold_risk_equivalence(folds, study['names'], fold_margin)
        for pair in risk['pairs']:
            reference = stats.binomtest(pair['discordant_seeds'], len(blocks), fold_margin, alternative='less').pvalue
            assert np.isclose(pair['p'], reference, atol=1e-14)
        results.append({'study': study['name'], 'policies': study['names'], 'seeds': study['seeds'],
                        'recovery': recovery, 'joint_recovery_p': max(r['mean_global_p'] for r in recovery.values()),
                        'fold_risk': risk})
    return {'post_hoc': True, 'recovery_margin': recovery_margin, 'fold_risk_margin': fold_margin,
            'scope': 'Finite-model empirical recovery and grid-detected folding; not a population identifiability theorem.',
            'studies': results, 'independent_boundary_checks_passed': True}


def write_report(result):
    lines = ['# Identifiability: equivalence of recovery, and uncertainty about folds', '',
             '**Recovery is practically equivalent across matched policies at the stated tolerance. Fold-risk equivalence must be evaluated separately from recovery-score equivalence.**', '',
             'Recovery R² was already included in the previous TOST analysis. This follow-up adds regional recovery checks and a paired exact test for fold risk. It reuses completed runs; no new training is claimed.', '',
             f"Exploratory recovery tolerance: **±{result['recovery_margin']:g} R²**. Every policy pair must pass paired TOST at alpha 0.05. Seeds are independent units; distances and goals are not. Paired-t assumptions apply. These tolerances were set after the experiment, not prespecified.", '',
             '| Study | Recovery measure | Range of policy means | Global TOST p | Equivalent? |',
             '| --- | --- | ---: | ---: | --- |']
    labels = {'affine_log_r2': 'Whole range, uniform log-distance grid', 'near_r2': 'Near object: d < 4',
              'far_r2': 'Far from object: d ≥ 9', 'log_distance_r2': 'Stationary-weighted log-distance recovery',
              'uniform_log_grid_r2': 'Uniform log-distance grid recovery'}
    for study in result['studies']:
        name = 'Seven original policies, 20 seeds' if len(study['policies']) == 7 else 'Four surrogates, 10 seeds'
        for key, recovery in study['recovery'].items():
            means = list(recovery['policy_means'].values())
            lines.append(f"| {name} | {labels[key]} | {min(means):.6f}–{max(means):.6f} | {recovery['mean_global_p']:.3g} | {'Yes' if recovery['mean_equivalent'] else 'Not established'} |")
    new = result['studies'][-1]
    risk = new['fold_risk']
    required_seeds = int(np.floor(np.log(.05) / np.log1p(-result['fold_risk_margin']))) + 1
    assert all(count == 0 for count in risk['fold_counts'].values())
    lines += ['', f"For the seven original policies, the conjunction of all three recovery measures passes: **p = {new['joint_recovery_p']:.3g}**. Each endpoint requires all 21 policy pairs to pass. This intersection-union claim needs no additional pair-count correction; individual intervals remain pointwise, not simultaneous.", '',
              '## What about folded representations?', '',
              'All seven matched policies have **0 grid-detected folds in 20 runs**. Identical observations do not establish identical population fold probabilities. The grid check uses 1,025 log-spaced distances and its existing numerical threshold; it cannot rule out every between-grid fold.', '',
              'For each policy pair, bound the absolute fold-probability difference by the probability that their paired runs disagree about folding. An exact binomial test asks whether that disagreement probability is below the tolerance. This is a conservative sufficient test of fold-risk equivalence.', '',
              f"At **±{100 * result['fold_risk_margin']:g} percentage points** of fold probability, the seven-policy global p is **{risk['global_p']:.3f}**: equivalence is **not established**. With zero disagreements in 20 seeds, the pointwise one-sided 95% upper bound is **{100 * risk['pairs'][0]['discordance_upper95']:.1f}%**. The earlier four-policy study has only 10 seeds and is also inconclusive at the same margin.", '',
              f"If we fix this {100 * result['fold_risk_margin']:g}-percentage-point tolerance and collect a fresh cohort, this conservative rule needs at least **{required_seeds} paired seeds with zero disagreements** to pass. That is a best-case requirement, not a general power calculation or a reason to stop sampling once significance appears.", '',
              '## What this does—and does not—say', '',
              '- Similar R² scores mean similar empirical linear recovery of the chosen log-distance coordinate. They do not prove the functions themselves are identical, or that either is exactly identifiable up to rotation.',
              '- In one dimension an orthogonal rotation is only a sign. R² allows fitting scale and offset as well; it is an empirical recovery diagnostic, not a direct test of the paper’s full theorem.',
              '- Both policies could perform equally poorly, so the table includes absolute recovery levels. Here these levels are high but below one.',
              '- The seven-policy study matches starting observations and SIGReg exposure, not successor distributions or stationarity. No conclusion extends to the original native-visitation zoo.', '',
              '[Full pairwise tests and exact fold-risk calculations](IDENTIFIABILITY_EQUIVALENCE.json) · [Earlier recovery/planning equivalence](EQUIVALENCE_SUMMARY.md)', '']
    (tests.OUT / 'IDENTIFIABILITY_EQUIVALENCE.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-margin', type=float, default=.001)
    parser.add_argument('--fold-margin', type=float, default=.05)
    args = parser.parse_args()
    if args.recovery_margin <= 0 or not 0 < args.fold_margin < 1:
        parser.error('Recovery margin must be positive and fold margin must lie between zero and one.')
    result = run(args.recovery_margin, args.fold_margin)
    (tests.OUT / 'IDENTIFIABILITY_EQUIVALENCE.json').write_text(json.dumps(result, indent=2) + '\n')
    write_report(result)
    print(json.dumps([{'study': s['study'], 'joint_recovery_p': s['joint_recovery_p'],
                       'fold_risk_p': s['fold_risk']['global_p']} for s in result['studies']]))


if __name__ == '__main__':
    main()

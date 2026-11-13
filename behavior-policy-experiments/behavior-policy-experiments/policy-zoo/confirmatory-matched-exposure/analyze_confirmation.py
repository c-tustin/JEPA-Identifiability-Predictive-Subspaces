"""Fixed-cohort inference. Refuses to substitute a partial-cohort analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import run_confirmation as run
import matched_equivalence as equivalence
from identifiability_equivalence import fold_risk_equivalence

ENDPOINTS = ('affine_log_r2', 'near_r2', 'far_r2', 'shared_goal_error')


def endpoint(row, key):
    return row['planning']['shared']['goal_error'] if key == 'shared_goal_error' else row['metrics'][key]


def paired_direction(difference):
    values = np.asarray(difference)
    wins, losses = int(np.sum(values > 1e-10)), int(np.sum(values < -1e-10))
    p = float(stats.binom.sf(wins - 1, wins + losses, .5)) if wins + losses else 1.
    rng = np.random.default_rng(20261107)
    samples = values[rng.integers(0, len(values), (20000, len(values)))].mean(1)
    return {'wins': wins, 'losses': losses, 'ties': len(values) - wins - losses,
            'p': p, 'mean_benefit': float(values.mean()),
            'bootstrap_ci95_unadjusted': np.quantile(samples, [.025, .975]).tolist(),
            'seed_benefits': values.tolist(), 'estimand': 'Directional non-tie win probability; mean/interval descriptive.'}


def collect():
    lock = run.check_lock()
    rows = {}
    missing = []
    for seed in run.SEEDS:
        for policy in run.POLICIES:
            cells = ('matched', 'native_exposure', 'common_exposure') if policy == 'icm' else ('matched',)
            for cell in cells:
                path = run.OUT / f'{policy}_{cell}_seed{seed}.json'
                if not path.exists():
                    missing.append(path.name)
                    continue
                row = json.loads(path.read_text())
                assert row['signature'] == lock['signature']
                assert row['seed'] == seed and row['policy'] == policy and row['cell'] == cell
                rows[policy, cell, seed] = row
    return rows, missing


def analyze(rows):
    primary = {}
    for key in ENDPOINTS:
        # Compare all seven matched-reset cells within each of the 60 seeds.
        # Recovery uses no action predictor; shared_goal_error uses separately
        # fitted predictors trained on the same physical transitions per seed.
        matrix = np.array([[endpoint(rows[p, 'matched', seed], key) for p in run.POLICIES] for seed in run.SEEDS])
        # Tolerances were fixed before the fresh cohort: .001 R² and .005
        # physical distance units (2% of a .25 action step), not exact equality.
        margin = .005 if key == 'shared_goal_error' else .001
        result = equivalence.global_equivalence(matrix, run.POLICIES, margin)
        result['policy_means'] = dict(zip(run.POLICIES, matrix.mean(0).tolist()))
        primary[key] = result
    # All 21 pairs must pass on all four endpoints: whole/near/far recovery
    # and shared-data reaching. This conjunction is one confirmatory claim.
    joint_p = max(r['mean_global_p'] for r in primary.values())
    joint_median_p = max(r['median_global_p'] for r in primary.values())
    mechanism = {}
    for key in ('affine_log_r2', 'folded', 'shared_goal_error'):
        values = []
        for seed in run.SEEDS:
            # Mechanism comparison is ICM versus itself with identical native
            # alignment pairs; only the independently sampled SIGReg pool differs.
            native = rows['icm', 'native_exposure', seed]
            common = rows['icm', 'common_exposure', seed]
            if key == 'folded':
                benefit = int(native['metrics']['folded']) - int(common['metrics']['folded'])
            else:
                benefit = endpoint(common, key) - endpoint(native, key)
                if key == 'shared_goal_error':
                    benefit *= -1
            values.append(benefit)
        mechanism[key] = paired_direction(values)
    # Correct four planned claims together: the equivalence conjunction plus
    # three directional ICM exposure effects. The latter are sign-based tests,
    # not additional TOST comparisons of the seven policies.
    family = [{'claim': 'joint_recovery_and_shared_planning_equivalence', 'p': joint_p}]
    family += [{'claim': 'exposure_only_icm_benefit_' + key, 'p': value['p']} for key, value in mechanism.items()]
    for item, adjusted in zip(family, equivalence.holm([r['p'] for r in family])):
        item['holm_p'] = adjusted
        item['significant'] = adjusted < .05
    # Secondary question: each matched cell supplies its own predictor data.
    # These are its policy-specific reset transitions, not native rollout data.
    # This equivalence claim need not pass even when shared-data planning does.
    native_matrix = np.array([[rows[p, 'matched', seed]['planning']['native']['goal_error'] for p in run.POLICIES] for seed in run.SEEDS])
    folds = np.array([[rows[p, 'matched', seed]['metrics']['folded'] for p in run.POLICIES] for seed in run.SEEDS])
    native_result = equivalence.global_equivalence(native_matrix, run.POLICIES, .005)
    fold_result = fold_risk_equivalence(folds, run.POLICIES, .05)
    risk_bounds = {}
    for index, policy in enumerate(run.POLICIES):
        count = int(folds[:, index].sum())
        risk_bounds[policy] = {'folds': count, 'seeds': len(run.SEEDS),
                               'upper95_pointwise': 1. if count == len(run.SEEDS) else float(stats.beta.ppf(.95, count + 1, len(run.SEEDS) - count))}
    return {'signature': run.signature(), 'complete': True, 'seeds': list(run.SEEDS),
            'primary_endpoints': primary, 'primary_joint_p': joint_p,
            'robustness_joint_typical_difference_p': joint_median_p,
            'confirmatory_holm_family': family, 'mechanism': mechanism,
            'secondary_unadjusted': {'policy_specific_planning': native_result, 'fold_risk': fold_result,
                                     'pointwise_policy_fold_bounds': risk_bounds},
            'no_old_seeds_pooled': True}


def report(result, rows):
    equivalence_passed = result['confirmatory_holm_family'][0]['significant']
    lines = ['# Fresh-seed confirmation: what carries over?', '',
             '**Fixed before the cohort:** 60 new actor/training seeds; ±0.001 R² and ±0.005 physical goal error. The latter is 2% of an action step. No old outcomes are pooled into the tests.', '',
             f"**Primary conjunction {'passes' if equivalence_passed else 'does not pass'} after the four-claim Holm correction.** All policy pairs must be equivalent on all three recovery regions and shared-data reaching. This is an empirical statement about matched exposure, not a theorem for arbitrary behavior policies.", '',
             '| Outcome | Range of policy means | Tolerance | Global TOST p |',
             '| --- | ---: | ---: | ---: |']
    for key, value in result['primary_endpoints'].items():
        means = list(value['policy_means'].values())
        lines.append(f"| {key} | {min(means):.6f}–{max(means):.6f} | ±{value['margin']:g} | {value['mean_global_p']:.3g} |")
    lines += ['', '| Confirmatory claim | p | Holm p | Pass? |', '| --- | ---: | ---: | --- |']
    for row in result['confirmatory_holm_family']:
        lines.append(f"| {row['claim']} | {row['p']:.3g} | {row['holm_p']:.3g} | {'Yes' if row['significant'] else 'No'} |")
    lines += ['', '## Why the ICM exposure control matters', '',
              'The same freshly trained ICM actor supplies identical native transition pairs to both control encoders. Only the observation pool used by SIGReg changes. Its formula and weight, optimizer budget and initialization stay fixed. Both cells use independent SIGReg sampling; coupled-versus-independent batching is not changed.', '',
              '| ICM cell | Mean recovery R² | Folds / 60 | Mean shared-data goal error |', '| --- | ---: | ---: | ---: |']
    for cell in ('native_exposure', 'common_exposure', 'matched'):
        group = [rows['icm', cell, seed] for seed in run.SEEDS]
        lines.append(f"| {cell} | {np.mean([r['metrics']['affine_log_r2'] for r in group]):.4f} | {sum(r['metrics']['folded'] for r in group)} | {np.mean([r['planning']['shared']['goal_error'] for r in group]):.4f} |")
    lines += ['', 'The three mechanism tests concern directional seed-wise improvement. Mean improvements and unadjusted bootstrap intervals are supplied in the JSON. Common exposure adds broad real observations; this is not a cost-free reweighting of ICM trajectories.', '',
              '## Significance and usefulness', '',
              'A significant TOST rejects differences as large as the fixed tolerance; it does not prove exact equality. The prospective cohort strengthens the earlier exploratory finding. The sign-based robustness check concerns typical differences, not means. Similar scores do not prove that learned functions are the same or linearly recover the true latent exactly.', '',
              'The scientifically useful question is whether exposure-only changes rescue geometry and actual reaching while keeping action transitions fixed. A confirmed effect would distinguish one finite-training failure mechanism from an intrinsic disadvantage of the ICM action rule. The paper already discusses sampling and finite-optimization limitations: this experiment offers a controlled mechanism test in one environment, not a new universal guarantee or an established novelty claim.', '',
              '## Secondary diagnostics, not additional confirmatory claims', '',
              f"Policy-specific predictor-data equivalence at ±0.005: unadjusted p = {result['secondary_unadjusted']['policy_specific_planning']['mean_global_p']:.3g}.", '',
              f"Paired fold-risk equivalence at ±5 percentage points: unadjusted p = {result['secondary_unadjusted']['fold_risk']['global_p']:.3g}. Rare-failure inference is limited even with 60 seeds; individual-policy upper bounds are pointwise. These diagnostics are outside the four-claim confirmatory family.", '',
              '## Scope and next question', '',
              'Starting observations and SIGReg exposure match; successor distributions and stationarity do not. These are physical OU action policies, not Gaussian latent OU dynamics. Shared predictor data retain the inherited endpoint-action omission. Results concern this architecture, data budget and goal distribution.', '',
              'A next extension would test whether a common frozen representation removes remaining policy-specific planning differences when only predictor action data change. That separates representation quality from action-conditioned transition coverage. It is not claimed to be answered by this cohort.', '',
              '[Frozen protocol](../PROTOCOL.md) · [Time/code lock](LOCK.json) · [Exact tests and effect sizes](INFERENCE.json) · [Verification](VERIFICATION.json) · [Paper §§7–8](https://arxiv.org/html/2605.26379v1#S7)', '']
    (run.OUT / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    rows, missing = collect()
    if missing:
        print(json.dumps({'complete': False, 'completed': len(rows), 'expected': 540, 'missing': len(missing), 'no_partial_inference': True}))
        if args.require_complete:
            raise SystemExit(1)
        return
    result = analyze(rows)
    run.save_json(run.OUT / 'INFERENCE.json', result)
    report(result, rows)
    print(json.dumps(result['confirmatory_holm_family']))


if __name__ == '__main__':
    main()

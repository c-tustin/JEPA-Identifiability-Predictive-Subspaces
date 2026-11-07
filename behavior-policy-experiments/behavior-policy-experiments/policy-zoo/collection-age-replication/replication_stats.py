"""The 19 comparisons fixed before the fresh-seed experiment began."""
import math

import numpy as np
from scipy.stats import spearmanr


TEST_NAMES = (
    'early_vs_late_recovery', 'early_vs_late_folds', 'early_vs_late_planning',
    'early_vs_random_recovery', 'early_vs_random_planning',
    'early_vs_count_recovery', 'early_vs_count_planning',
    'early_vs_ou_recovery', 'early_vs_ou_planning',
    'early_vs_initial_recovery', 'age_trend_recovery', 'age_trend_planning',
    'rescue_vs_random_recovery', 'rescue_vs_random_planning',
    'rescue_vs_count_recovery', 'rescue_vs_count_planning',
    'continuation_recovery', 'continuation_folds', 'continuation_planning',
)


def sign_test(values):
    """Exact two-sided signs; zero effects do not become artificial wins."""
    values = np.asarray(values)
    wins = int(np.sum(values > 1e-10))
    losses = int(np.sum(values < -1e-10))
    n = wins + losses
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    p = min(1., 2 * tail / 2**n) if n else 1.
    return wins, losses, p


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.zeros(len(order))
    largest = 0.
    for rank, index in enumerate(order):
        largest = max(largest, (len(order) - rank) * pvalues[index])
        result[index] = min(1., largest)
    return result


def bootstrap(values):
    values = np.asarray(values)
    rng = np.random.default_rng(20261105)
    sampled = values[rng.integers(0, len(values), (20000, len(values)))].mean(1)
    return np.quantile(sampled, [.025, .975]).tolist()


def age_correlation(ages, values):
    if np.ptp(values) < 1e-12:
        return 0.
    return float(spearmanr(ages, values).statistic)


def comparisons(rows, seeds, ages):
    """Only complete seed blocks enter the family; unfinished blocks stay pending."""
    lookup = {(row['condition'], row['seed']): row for row in rows}
    required = [f'{policy}_age{age:03d}' for policy in ('icm', 'random', 'count') for age in ages]
    required += ['ou', 'icm_early_to_early', 'icm_early_to_late']
    complete = [seed for seed in seeds if all((name, seed) in lookup for name in required)]
    effects = {name: [] for name in TEST_NAMES}
    for seed in complete:
        def row(condition):
            return lookup[condition, seed]
        def recovery(condition):
            return row(condition)['metrics']['affine_log_r2']
        def error(condition):
            return row(condition)['planning']['shared']['goal_error']
        def fold(condition):
            return int(row(condition)['metrics']['folded'])
        early, late = 'icm_age000', 'icm_age128'
        rescue_recovery = recovery(early) - recovery(late)
        rescue_planning = error(late) - error(early)
        effects['early_vs_late_recovery'].append(rescue_recovery)
        effects['early_vs_late_folds'].append(fold(late) - fold(early))
        effects['early_vs_late_planning'].append(rescue_planning)
        for policy in ('random', 'count', 'ou'):
            alternative = policy if policy == 'ou' else policy + '_age000'
            effects[f'early_vs_{policy}_recovery'].append(recovery(early) - recovery(alternative))
            effects[f'early_vs_{policy}_planning'].append(error(alternative) - error(early))
        effects['early_vs_initial_recovery'].append(recovery(early) - row(early)['initial_metrics']['affine_log_r2'])
        effects['age_trend_recovery'].append(-age_correlation(ages, [recovery(f'icm_age{age:03d}') for age in ages]))
        effects['age_trend_planning'].append(age_correlation(ages, [error(f'icm_age{age:03d}') for age in ages]))
        for policy in ('random', 'count'):
            control_early, control_late = policy + '_age000', policy + '_age128'
            control_recovery = recovery(control_early) - recovery(control_late)
            control_planning = error(control_late) - error(control_early)
            effects[f'rescue_vs_{policy}_recovery'].append(rescue_recovery - control_recovery)
            effects[f'rescue_vs_{policy}_planning'].append(rescue_planning - control_planning)
        a, b = 'icm_early_to_early', 'icm_early_to_late'
        effects['continuation_recovery'].append(recovery(a) - recovery(b))
        effects['continuation_folds'].append(fold(b) - fold(a))
        effects['continuation_planning'].append(error(b) - error(a))
    result = []
    if complete:
        for name in TEST_NAMES:
            delta = effects[name]
            wins, losses, p = sign_test(delta)
            result.append({'test': name, 'n': len(delta), 'wins': wins, 'losses': losses,
                           'mean_oriented_effect': float(np.mean(delta)),
                           'mean_ci95': bootstrap(delta), 'p': p,
                           'seed_effects': dict(zip(map(str, complete), map(float, delta)))})
        for row, adjusted in zip(result, holm([row['p'] for row in result])):
            row['holm_p'] = float(adjusted)
    return {'planned_family_size': 19, 'complete_seeds': complete,
            'pending_seeds': [s for s in seeds if s not in complete], 'tests': result}

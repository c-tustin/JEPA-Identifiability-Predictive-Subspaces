"""Design check from old data only; never reads a new cohort outcome."""
import json
import numpy as np
from scipy import stats
import run_confirmation as run


def main():
    metrics = ('affine_log_r2', 'near_r2', 'far_r2', 'shared_goal_error')
    blocks = []
    for seed in range(8, 28):
        row = []
        for policy in run.POLICIES:
            saved = json.loads((run.previous.OUT / f'zoo_{policy}_reset_seed{seed}.json').read_text())
            row.append([saved['planning']['shared']['goal_error'] if key == 'shared_goal_error' else saved['metrics'][key] for key in metrics])
        blocks.append(row)
    values = np.array(blocks)
    import itertools
    differences = np.stack([values[:, i] - values[:, j] for i, j in itertools.combinations(range(7), 2)], axis=1)
    margins = np.array([.001, .001, .001, .005])
    rng = np.random.default_rng(20261107)
    passed = 0
    draws = 10000
    # Conservative for the Holm family: use alpha .05/4 for each component.
    critical = stats.t.ppf(1 - .05 / 4, 59)
    for _ in range(draws // 250):
        samples = differences[rng.integers(0, 20, (250, 60))]
        means = samples.mean(1)
        widths = critical * samples.std(1, ddof=1) / np.sqrt(60)
        passed += int(np.sum(np.all(np.abs(means) + widths < margins, axis=(1, 2))))
    result = {'old_seeds': list(range(8, 28)), 'new_seeds_examined': 0,
              'fixed_new_cohort_size': 60, 'pilot_bootstrap_draws': draws,
              'pilot_joint_pass_fraction_at_bonferroni_level': passed / draws,
              'zero_discordance_probability_at_rates': {str(q): (1-q)**60 for q in (0, .005, .01, .02, .05)},
              'caution': 'Pilot resampling can miss rare failures and fresh-actor variability. This is design sensitivity, not guaranteed power.'}
    run.save_json(run.OUT / 'PILOT_PRECISION.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

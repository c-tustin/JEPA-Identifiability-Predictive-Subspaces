"""Verify collection windows, original objective and planned statistical contrasts."""
import unittest

import numpy as np
from scipy.stats import binomtest
import torch
import common as study
import replication_stats as stats


class ReplicationControls(unittest.TestCase):
    def test_windows_preserve_real_trajectory_steps(self):
        torch.set_num_threads(1)
        c = study.original.config()
        c.burnin, c.horizon = 0, 192
        full = study.behavior.collect(c, 'random', c.train_episodes, 10200)
        for age in study.AGES:
            window = study.extract_window(full, age)
            self.assertEqual(len(window['x']), 16384)
            first = age * c.train_episodes
            torch.testing.assert_close(window['x'][0], full['x'][first], atol=0, rtol=0)
            # Apart from float sensor inversion, every next view follows its action.
            d = 1 / window['x'][:, 1]
            successor = study.behavior.move(d, window['action'], c)
            torch.testing.assert_close(study.behavior.sensor(successor, c), window['xp'], atol=2e-6, rtol=2e-6)
        late_c = study.original.config()
        direct = study.behavior.collect(late_c, 'random', late_c.train_episodes, 10200)
        self.assertEqual(study.data_hash(direct), study.data_hash(study.extract_window(full, 128)))

    def test_exact_signs_and_holm(self):
        for wins, losses in [(20,0), (10,0), (19,1), (7,3), (0,0)]:
            values = [1]*wins + [-1]*losses + [0]*3
            _, _, p = stats.sign_test(values)
            expected = binomtest(wins, wins+losses, .5).pvalue if wins+losses else 1.
            self.assertAlmostEqual(p, expected)
        np.testing.assert_allclose(stats.holm([.01, .04, .03]), [.03, .06, .06])
        self.assertEqual(len(stats.TEST_NAMES), 19)

    def test_planned_contrasts_use_complete_seed_blocks(self):
        rows = []
        for policy in study.POLICIES:
            for age in study.AGES:
                recovery = .9 - age/256 if policy == 'icm' else (.8 if policy == 'random' else .85)
                error = .1 + age/256 if policy == 'icm' else .2
                rows.append({'condition': study.label(policy, age), 'seed': 1,
                             'metrics': {'affine_log_r2': recovery, 'folded': policy == 'icm' and age >= 64},
                             'initial_metrics': {'affine_log_r2': .7},
                             'planning': {'shared': {'goal_error': error}}})
        for name,recovery,error,fold in [('ou',.95,.08,False),
                                         ('icm_early_to_early',.92,.05,False),
                                         ('icm_early_to_late',.3,.8,True)]:
            rows.append({'condition':name,'seed':1,
                         'metrics':{'affine_log_r2':recovery,'folded':fold},
                         'planning':{'shared':{'goal_error':error}}})
        result = stats.comparisons(rows, [1], study.AGES)
        self.assertEqual(result['complete_seeds'], [1])
        self.assertEqual(len(result['tests']), 19)
        lookup = {r['test']:r['mean_oriented_effect'] for r in result['tests']}
        self.assertAlmostEqual(lookup['early_vs_late_recovery'], .5)
        self.assertAlmostEqual(lookup['early_vs_ou_recovery'], -.05)
        self.assertAlmostEqual(lookup['rescue_vs_random_recovery'], .5)
        self.assertAlmostEqual(lookup['continuation_folds'], 1.)
        self.assertAlmostEqual(lookup['continuation_planning'], .75)
        self.assertAlmostEqual(lookup['age_trend_recovery'], 1.)
        incomplete = stats.comparisons(rows[:-1], [1], study.AGES)
        self.assertEqual(incomplete['complete_seeds'], [])
        self.assertEqual(incomplete['tests'], [])

    def test_continuation_starts_with_same_model_and_adam(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = study.behavior.Encoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        c = study.original.config()
        d = torch.linspace(1, 16, 64)
        x = study.behavior.sensor(d, c)
        xp = study.behavior.sensor((d+.1).clamp(max=16), c)
        regularizer = study.behavior.SIGReg1D()
        study.controls.update(model, optimizer, regularizer, x, xp, c, .001)
        import copy
        state = copy.deepcopy(optimizer.state_dict())
        branches = []
        for _ in range(2):
            branch = copy.deepcopy(model)
            adam = torch.optim.Adam(branch.parameters(), lr=.001)
            adam.load_state_dict(copy.deepcopy(state))
            study.controls.update(branch, adam, regularizer, x, xp, c, .001)
            branches.append(branch)
        for a,b in zip(branches[0].parameters(), branches[1].parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()

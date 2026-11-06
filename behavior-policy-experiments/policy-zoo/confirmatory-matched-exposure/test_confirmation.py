"""Synthetic protocol checks; these never train or inspect a fresh seed."""
import unittest
from copy import deepcopy
import analyze_confirmation as analysis
import run_confirmation as run


class FixedInference(unittest.TestCase):
    def synthetic(self):
        rows = {}
        for seed in run.SEEDS:
            for index, policy in enumerate(run.POLICIES):
                cells = ('matched', 'native_exposure', 'common_exposure') if policy == 'icm' else ('matched',)
                for cell in cells:
                    bad = cell == 'native_exposure'
                    metric = .80 if bad else .98 + index * .00001 + (seed % 3) * .0001
                    error = .5 if bad else .05 + index * .00001
                    rows[policy, cell, seed] = {'metrics': {'affine_log_r2': metric, 'near_r2': metric, 'far_r2': metric, 'folded': bad},
                                               'planning': {'shared': {'goal_error': error}, 'native': {'goal_error': error}}}
        return rows

    def test_global_equivalence_requires_every_pair_and_endpoint(self):
        rows = self.synthetic()
        result = analysis.analyze(rows)
        self.assertTrue(result['confirmatory_holm_family'][0]['significant'])
        for seed in run.SEEDS:
            rows['random', 'matched', seed]['metrics']['near_r2'] -= .01
        failed = analysis.analyze(rows)
        self.assertFalse(failed['confirmatory_holm_family'][0]['significant'])
        self.assertEqual(len(failed['confirmatory_holm_family']), 4)

    def test_direction_and_ties_are_explicit(self):
        self.assertEqual(analysis.paired_direction([0] * 60)['p'], 1.)
        all_benefit = analysis.paired_direction([1] * 60)
        self.assertAlmostEqual(all_benefit['p'], .5 ** 60)
        self.assertEqual(all_benefit['wins'], 60)
        self.assertEqual(analysis.paired_direction([-1] * 60)['p'], 1.)


if __name__ == '__main__':
    unittest.main()

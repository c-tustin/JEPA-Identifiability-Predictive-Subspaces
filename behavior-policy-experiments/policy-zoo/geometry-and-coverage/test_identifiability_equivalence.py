"""Exact fold-risk inference must not mistake observed equality for equivalence."""
import unittest
import numpy as np
from identifiability_equivalence import fold_risk_equivalence


class FoldRiskInference(unittest.TestCase):
    def test_zero_disagreements_requires_enough_independent_seeds(self):
        small = fold_risk_equivalence(np.zeros((20, 2), dtype=bool), ['a', 'b'], .05)
        self.assertAlmostEqual(small['global_p'], .95 ** 20)
        self.assertFalse(small['equivalence_established'])
        self.assertAlmostEqual(small['pairs'][0]['discordance_upper95'], 1 - .05 ** (1 / 20))
        larger = fold_risk_equivalence(np.zeros((59, 2), dtype=bool), ['a', 'b'], .05)
        self.assertTrue(larger['equivalence_established'])

    def test_global_claim_fails_if_any_pair_disagrees_every_time(self):
        outcomes = np.zeros((59, 3), dtype=bool)
        outcomes[:, 2] = True
        result = fold_risk_equivalence(outcomes, ['a', 'b', 'c'], .05)
        self.assertEqual(result['global_p'], 1.)
        self.assertFalse(result['equivalence_established'])
        self.assertEqual(len(result['pairs']), 3)


if __name__ == '__main__':
    unittest.main()

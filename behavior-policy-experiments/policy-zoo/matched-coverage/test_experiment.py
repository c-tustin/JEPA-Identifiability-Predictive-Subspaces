"""Checks the actual controls that make the policy comparison interpretable."""
import unittest
import numpy as np
import torch
import matched as exp


class ControlledComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.design = exp.make_design()

    def test_stationarity_reversibility_and_matched_movement(self):
        d = self.design
        for name in exp.NAMES:
            counts, kernel = d['flows'][name], d['kernels'][name]
            np.testing.assert_array_equal(counts.sum(0), d['counts'])
            np.testing.assert_array_equal(counts.sum(1), d['counts'])
            np.testing.assert_array_equal(counts, counts.T)
            np.testing.assert_allclose(d['pi'] @ kernel, d['pi'], atol=1e-15)
            self.assertEqual(int(np.trace(counts, offset=1)), d['total_flux'])
            self.assertTrue(np.all(np.diag(kernel) >= .5))
            self.assertTrue(np.all(np.diag(kernel, 1) > 0))

    def test_actual_datasets_share_both_marginals_and_actions(self):
        d = self.design
        reference = exp.source(d, 'reference', 100)
        for name in exp.NAMES:
            data = exp.source(d, name, 100)
            np.testing.assert_array_equal(np.bincount(data['i']), d['counts'])
            np.testing.assert_array_equal(np.bincount(data['j']), d['counts'])
            torch.testing.assert_close(data['x'], reference['x'], rtol=0, atol=0)
            torch.testing.assert_close(data['action'].bincount(), reference['action'].bincount())
            distance = torch.tensor(d['distance'], dtype=torch.float32)
            successor = exp.behavior.move(distance[data['i']], data['action'], exp.behavior.Config())
            torch.testing.assert_close(successor, distance[data['j']], rtol=0, atol=0)
            replay = exp.source(d, name, 100)
            torch.testing.assert_close(data['xp'], replay['xp'], rtol=0, atol=0)

    def test_spectral_features_satisfy_the_alignment_identity(self):
        d = self.design
        for name in exp.NAMES:
            eigenvalues, features = exp.slow_features(d['kernels'][name], d['pi'], d['distance'])
            np.testing.assert_allclose(d['pi'] @ features, 0, atol=1e-11)
            np.testing.assert_allclose(features.T @ (d['pi'][:, None] * features), np.eye(3), atol=1e-11)
            np.testing.assert_allclose(d['kernels'][name] @ features, features * eigenvalues, atol=1e-11)
            feature = features[:, 0]
            loss = np.sum(d['pi'][:, None] * d['kernels'][name] * (feature[:, None] - feature[None, :])**2)
            self.assertAlmostEqual(loss, 2 * (1 - eigenvalues[0]), places=11)
            # Connected reversible nearest-neighbor chains have a monotone first mode.
            self.assertTrue(np.all(np.diff(feature) > 0))


if __name__ == '__main__':
    unittest.main()

"""Check the discrete conditional-action excitation bounds independently."""

import unittest
import torch


class ExcitationChecks(unittest.TestCase):
    def test_uniform_floor_bounds_variance_and_counterfactual_error(self):
        torch.manual_seed(70)
        eta = 0.2
        base = torch.softmax(torch.randn(1000, 3) * 10, -1)
        probabilities = (1 - eta) * base + eta / 3
        actions = torch.tensor([-1.0, 0.0, 1.0])
        means = (probabilities * actions).sum(-1)
        variance = (probabilities * actions.square()).sum(-1) - means.square()
        self.assertTrue(bool((variance >= 2 * eta / 3 - 1e-6).all()))
        error = torch.rand(1000, 3)
        uniform = error.mean(-1)
        behavior = (probabilities * error).sum(-1)
        self.assertTrue(bool((uniform <= behavior / eta + 1e-6).all()))
        # The counterfactual factor is attained at a least-covered action.
        p = torch.tensor([1 - 2 * eta / 3, eta / 3, eta / 3])
        e = torch.tensor([0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(e.mean() / (p * e).sum()), 1 / eta, places=5)


if __name__ == "__main__":
    unittest.main()

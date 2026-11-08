import unittest

import torch

from check_policy_effects import radial_twist, simulate


class PolicyEffectChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_twist_preserves_radius_and_has_explicit_inverse(self):
        z = torch.randn(10000, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(19))
        u = radial_twist(z, .7)
        torch.testing.assert_close(u.square().sum(-1), z.square().sum(-1), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(radial_twist(u, -.7), z, rtol=1e-12, atol=1e-12)

    def test_nonlinear_gaussian_preserving_map_beats_linear_maps(self):
        result, _, _ = simulate(samples=30000)
        self.assertGreater(result["empirical_loss_advantage"], 10 * result["monte_carlo_standard_error_of_advantage"])
        self.assertAlmostEqual(result["empirical_nonlinear_alignment"], result["population_nonlinear_optimum"], delta=.015)

    def test_zero_twist_reduces_to_ordinary_ou(self):
        result, _, _ = simulate(samples=1000, strength=0.)
        self.assertAlmostEqual(result["empirical_linear_alignment"], result["empirical_nonlinear_alignment"], places=12)


if __name__ == "__main__":
    unittest.main()

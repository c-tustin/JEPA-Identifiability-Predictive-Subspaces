"""Check the action-coordinate calculation and the approximate bound."""

import unittest
import torch
from check_actuation import analytic_bound, intervention_views, AnchoredEncoder


class ActuationChecks(unittest.TestCase):
    def test_common_noise_cancels_for_affine_coordinate(self):
        d = torch.linspace(1.2, 15.8, 300, dtype=torch.float64)
        minus, plus = intervention_views(d, torch.randn_like(d))
        effect = (1 / plus[:, 1] - 1 / minus[:, 1]) / (2 * 0.25 * 0.02)
        torch.testing.assert_close(effect, torch.ones_like(effect), atol=1e-10, rtol=0)

    def test_ito_action_coefficient_detects_nonlinear_warp(self):
        d = torch.linspace(1.2, 15.8, 300, dtype=torch.float64, requires_grad=True)
        y = d + 0.05 * torch.sin(d)
        first = torch.autograd.grad(y.sum(), d, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), d)[0]
        sigma = 0.05 * (1 + 0.1 * torch.cos(d))
        state_drift = first * 0.3 * torch.sin(d) + 0.5 * second * sigma.square()
        plus = state_drift + 0.25 * first
        minus = state_drift - 0.25 * first
        torch.testing.assert_close((plus - minus) / 2, 0.25 * first, atol=1e-15, rtol=0)
        self.assertGreater(float(first.max() - first.min()), 0.09)

    def test_proved_bound_contains_smooth_example_error(self):
        result = analytic_bound()
        self.assertTrue(result["bound_holds_on_grid"])
        self.assertGreater(result["conditional_drift_risk"], 0)
        self.assertLess(
            result["actual_sup_affine_error_squared"],
            result["proved_sup_error_squared_bound"],
        )

    def test_encoder_is_smooth_and_strictly_ordered(self):
        encoder = AnchoredEncoder()
        x = torch.ones(100, 6, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            x[:, 1] = torch.linspace(0.06, 1, 100, dtype=torch.float64)
        y = encoder(x)
        first = torch.autograd.grad(y.sum(), x, create_graph=True)[0][:, 1]
        second = torch.autograd.grad(first.sum(), x)[0][:, 1]
        self.assertTrue(bool((first < 0).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))


if __name__ == "__main__":
    unittest.main()

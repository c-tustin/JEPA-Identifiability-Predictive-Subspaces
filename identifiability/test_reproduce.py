"""Numerical checks for the scalar SIGReg adaptation and inverse-distance sensor."""
import unittest

import torch
import torch.nn.functional as F

from reproduce import Encoder, SIGReg1D, observe


class ReproductionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_sigreg_matches_upstream_sliced_expression(self):
        # Reference: klindtlab/lejepa-identifiability, losses.py (MIT).
        h = torch.randn(2, 512, 1, requires_grad=True)
        actual = SIGReg1D()(h)
        flat = h.flatten(0, 1)
        directions = F.normalize(torch.randn(1, 256), dim=0)
        frequency = torch.linspace(0., 3., 17)
        phase = (flat @ directions).unsqueeze(-1) * frequency
        target = torch.exp(-frequency.square() / 2)
        error = (phase.cos().mean(0) - target).square() + phase.sin().mean(0).square()
        weights = torch.full((17,), 2 * 3 / 16)
        weights[[0, -1]] = 3 / 16
        expected = (error @ (weights * target)).mean() * flat.size(0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(SIGReg1D()(-h), actual)
        actual.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertGreater(float(h.grad.abs().max()), 0.)

    def test_sigreg_penalizes_collapse_and_mean_shift(self):
        criterion = SIGReg1D()
        normal = torch.randn(12000)
        reference = float(criterion(normal))
        self.assertLess(reference, float(criterion(torch.zeros_like(normal))))
        self.assertLess(reference, float(criterion(normal + 1.)))

    def test_inverse_distance_sensor(self):
        sigma = .7
        z = torch.tensor([-1., 0., 1.])
        x = observe(z, sigma)
        torch.testing.assert_close(x[0], x[1] * torch.exp(torch.tensor(sigma)))
        torch.testing.assert_close(x[2], x[1] * torch.exp(torch.tensor(-sigma)))
        self.assertEqual(Encoder()(x).shape, (3,))

    def test_stationary_pair_moments(self):
        z = torch.randn(200000)
        rho = .9
        zp = rho * z + (1 - rho**2)**.5 * torch.randn_like(z)
        self.assertAlmostEqual(float(zp.mean()), 0., delta=.015)
        self.assertAlmostEqual(float(zp.var()), 1., delta=.015)
        self.assertAlmostEqual(float((zp - z).square().mean()), 2 * (1 - rho), delta=.003)


if __name__ == "__main__":
    unittest.main()

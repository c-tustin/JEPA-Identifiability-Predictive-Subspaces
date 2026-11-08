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

    def test_readable_sigreg_preserves_previous_values_and_gradients(self):
        for dtype in [torch.float32, torch.float64]:
            for shape in [(512,), (512, 1), (2, 256, 1)]:
                old_input = torch.randn(shape, dtype=dtype, requires_grad=True)
                new_input = old_input.detach().clone().requires_grad_(True)
                loss = SIGReg1D().to(dtype=dtype)
                # Exact pre-refactor expression, retained as an independent check.
                phase = old_input.reshape(-1, 1) * loss.frequency
                real_error = phase.cos().mean(0) - loss.target
                imaginary_error = phase.sin().mean(0)
                previous = old_input.numel() * ((real_error.square() + imaginary_error.square()) * loss.weight).sum()
                readable = loss(new_input)
                torch.testing.assert_close(readable, previous, rtol=0, atol=0)
                old_gradient = torch.autograd.grad(previous, old_input)[0]
                new_gradient = torch.autograd.grad(readable, new_input)[0]
                torch.testing.assert_close(new_gradient, old_gradient, rtol=0, atol=0)

    def test_sigreg_buffer_compatibility_and_invalid_input(self):
        source, restored = SIGReg1D(), SIGReg1D()
        self.assertEqual(set(source.state_dict()), {"frequency", "target", "weight"})
        restored.load_state_dict(source.state_dict())
        data = torch.randn(32)
        torch.testing.assert_close(source(data), restored(data), rtol=0, atol=0)
        for data in [torch.empty(0), torch.randn(8, 2), torch.tensor(1.)]:
            with self.assertRaises(ValueError):
                source(data)
        with self.assertRaises(ValueError):
            SIGReg1D(knots=1)

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

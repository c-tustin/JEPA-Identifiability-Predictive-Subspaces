"""Scientific assumption checks, not only implementation snapshots."""

import numpy as np
import pytest
from scipy.integrate import quad
from closure_controls import (
    R,
    observe,
    calibrated_coordinate,
    signed_logdensity,
    pairs,
    lognormal,
)
from run_drift import features, simulate, basis


def test_same_triangle_roundtrip():
    z = np.linspace(-6, 6, 101)
    np.testing.assert_allclose(calibrated_coordinate(observe(z)), z, atol=1e-14)
    assert observe(z).shape == (101, 6)


@pytest.mark.parametrize("z", [0.0, 0.5, 2.0])
def test_mixture_normalization_and_conditional_moments(z):
    density = lambda y: np.exp(signed_logdensity(z, y))
    assert quad(density, -np.inf, np.inf)[0] == pytest.approx(1, abs=1e-9)
    assert quad(lambda y: y * density(y), -np.inf, np.inf)[0] == pytest.approx(
        0, abs=1e-9
    )
    assert quad(lambda y: y * y * density(y), -np.inf, np.inf)[0] == pytest.approx(
        1 - R * R + R * R * z * z, abs=1e-8
    )


def test_information_bound_and_marginal():
    lower = -0.5 * np.log(1 - R * R) - np.log(2)
    assert lower > -0.5 * np.log(0.75)
    z, y, _ = pairs(500000, 9981)
    info = signed_logdensity(z, y) - lognormal(y, 0, 1)
    assert info.mean() > lower
    assert abs(y.mean()) < 0.01
    assert abs(y.var() - 1) < 0.015
    assert abs(np.corrcoef(z * z, y * y)[0, 1] - R * R) < 0.015
    assert abs(np.mean(y**4) - 3) < 0.07


def test_non_gaussian_conditionals_are_not_theorem_counterexample():
    # At z=0 variance=.0975, at z=2 variance=3.7075. A.2 fails.
    assert 1 - R * R != 1 - R * R + 4 * R * R
    # Away from zero the fourth conditional cumulant is -2(r*z)^4.
    z = 1.0
    fourth = quad(lambda y: y**4 * np.exp(signed_logdensity(z, y)), -np.inf, np.inf)[0]
    variance = 1 - R * R + R * R * z * z
    assert fourth - 3 * variance**2 == pytest.approx(-2 * (R * z) ** 4, abs=1e-8)


def test_density_stability_in_tails():
    assert np.isfinite(
        signed_logdensity(np.array([1e3, -1e3]), np.array([1e3, 1e3]))
    ).all()


def test_multi_interval_design_has_instantaneous_limit():
    d = np.linspace(2, 15, 20)
    a = np.resize([-1.0, 1.0], 20)
    f = features(d, a, np.zeros(20), True)
    np.testing.assert_allclose(f[:, 5:], 0)
    np.testing.assert_allclose(f[:, :5], features(d, a, np.ones(20), False))


def test_simulator_independent_noise_and_deterministic_convergence():
    d = np.full(10000, 5.0)
    a = np.ones_like(d)
    y = simulate(d, a, 0.02, np.random.default_rng(12))
    sigma = 0.05 * (1 + 0.1 * np.cos(5.0))
    assert np.var(y) / (0.02 * sigma**2) == pytest.approx(1, rel=0.04)
    x = np.linspace(2, 15, 100)
    p = simulate(x, np.ones(100), 0.1, np.random.default_rng(1), False, steps=64)
    q = simulate(x, np.ones(100), 0.1, np.random.default_rng(1), False, steps=128)
    assert np.max(abs(p - q)) < 2e-5


def test_matched_learned_parameter_budget_and_gradients():
    import torch
    from run_learned import KINDS, network, losses, parameters

    counts = []
    for kind in KINDS:
        p = network()
        counts.append(sum(v.numel() for v in p.parameters()))
        raw = torch.randn(16, 3, requires_grad=True)
        loss = losses(raw, torch.randn(16), kind).mean()
        loss.backward()
        assert torch.isfinite(raw.grad).all()
        assert (raw.grad.abs().sum(0) > 0).all()
    assert len(set(counts)) == 1

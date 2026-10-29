"""Readable, one-dimensional SIGReg loss used by the triangle experiments.

Adapted from klindtlab/lejepa-identifiability, experiments/lejepa_id/losses.py.
See THIRD_PARTY_NOTICES.md for the upstream MIT notice.
"""
import math

import torch
from torch import nn


class SIGReg1D(nn.Module):
    """Encourage scalar outputs to follow a standard Gaussian distribution.

    Compare the batch's average cosine and sine at several frequencies with
    their known averages for a standard Gaussian. Integrate the squared error,
    using the authors' Gaussian weighting and batch-size scaling.

    Args:
        knots: Number of evenly spaced integration frequencies, including zero.
        t_max: Largest integration frequency.

    Inputs may be [samples], [samples, 1], or [views, samples, 1]. Each view's
    scalar output counts as one sample. This class does not handle vectors of
    several features: full SIGReg would first project those vectors to scalars.

    In one dimension, normalized projection directions are only +1 and -1.
    Both give the same squared error, so we need no random projections here.
    Finite batches and a finite frequency grid provide an approximation, not a
    certificate that the entire output distribution is exactly Gaussian.
    """

    def __init__(self, knots=17, t_max=3.):
        super().__init__()
        if not isinstance(knots, int) or knots < 2:
            raise ValueError("knots must be an integer of at least 2")
        if not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("t_max must be finite and positive")

        # 1. Choose the frequencies where we compare the two distributions.
        frequencies = torch.linspace(0., t_max, knots)

        # A standard Gaussian has E[cos(t*y)] = exp(-t²/2), E[sin(t*y)] = 0.
        gaussian_cosine = torch.exp(-frequencies.square() / 2)

        # 2. Build twice the trapezoidal integration weights. The factor of two
        # includes negative frequencies: their squared errors are identical.
        trapezoid_weights = torch.ones(knots) * (2 * t_max / (knots - 1))
        trapezoid_weights[[0, -1]] *= .5
        integration_weights = trapezoid_weights * gaussian_cosine

        # Keep the original buffer names for state-dictionary compatibility.
        self.register_buffer("frequency", frequencies)
        self.register_buffer("target", gaussian_cosine)
        self.register_buffer("weight", integration_weights)

    def forward(self, embedding):
        if embedding.numel() == 0:
            raise ValueError("SIGReg needs at least one scalar sample")
        if embedding.ndim == 0 or (embedding.ndim > 1 and embedding.shape[-1] != 1):
            raise ValueError("Expected scalar outputs: [B], [B,1], or [V,B,1]")

        samples = embedding.reshape(-1, 1)
        sample_count = embedding.numel()

        # 3. Average sine and cosine over samples at every frequency.
        # Together these are the real and imaginary parts of the empirical
        # characteristic function, a way of describing a distribution.
        phase = samples * self.frequency
        mean_cosine = phase.cos().mean(dim=0)
        mean_sine = phase.sin().mean(dim=0)

        # 4. Compare with the standard-Gaussian averages.
        cosine_error = mean_cosine - self.target
        sine_error = mean_sine  # The Gaussian sine average is zero.
        squared_error = cosine_error.square() + sine_error.square()

        # 5. Approximate the integral, then use the authors' batch-size scaling.
        integrated_error = (squared_error * self.weight).sum()
        return sample_count * integrated_error

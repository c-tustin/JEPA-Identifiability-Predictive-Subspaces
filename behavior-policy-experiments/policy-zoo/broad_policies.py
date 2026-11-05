"""Broad-coverage controllers following the Gaussian-targeted discovery."""

import math
import torch

from zoo import Policy
import targeted_policies
from targeted_policies import GaussianMetropolis, ESCORT_STEPS, observed_distance


def guard(view, action):
    distance = observed_distance(view)
    proposed = distance + (action.float() - 1) * 0.25
    valid = (proposed >= 1.0) & (proposed <= 16.0)
    return torch.where(valid, action, torch.ones_like(action))


class GuardedBias(Policy):
    def __init__(self, probabilities=(.36, .33, .31)):
        self.probabilities = torch.tensor(probabilities)

    def act(self, view):
        action = torch.multinomial(self.probabilities.expand(len(view), 3), 1).flatten()
        return guard(view, action)


class AdaptiveBias(Policy):
    def act(self, view):
        error = observed_distance(view).log() - math.log(4)
        inward = .07 * torch.tanh(error / .8)
        probabilities = torch.stack([(.67 + inward) / 2, torch.full_like(error, .33), (.67 - inward) / 2], dim=1)
        action = torch.multinomial(probabilities, 1).flatten()
        return guard(view, action)


class GuardedSweep(Policy):
    def reset(self, n):
        self.direction = torch.where(torch.rand(n) < .5, -1, 1)

    def act(self, view):
        distance = observed_distance(view)
        self.direction = torch.where(distance <= 1.25, 1, self.direction)
        self.direction = torch.where(distance >= 15.75, -1, self.direction)
        noise = torch.rand(len(view)) < .15
        action = torch.where(noise, torch.randint(0, 3, (len(view),)), self.direction + 1)
        return guard(view, action)


class BiasSweepMixture(GuardedSweep):
    def __init__(self):
        self.bias = GuardedBias()

    def act(self, view):
        sweep = super().act(view)
        biased = self.bias.act(view)
        return torch.where(torch.rand(len(view)) < .5, sweep, biased)


class NoEscortLogDiffusion(GaussianMetropolis):
    def __init__(self):
        super().__init__(standard_deviation=.65, log_diffusion=True)

    def reset(self, n):
        # Keep target RNG draws matched to the earlier controller; only skip
        # the actions toward that target. All later actions use the same rule.
        super().reset(n)
        self.steps = ESCORT_STEPS


POLICIES = {
    "guarded_bias": GuardedBias,
    "gentle_bias": lambda: GuardedBias((.35, .33, .32)),
    "adaptive_bias": AdaptiveBias,
    "guarded_sweep": GuardedSweep,
    "bias_sweep_mix": BiasSweepMixture,
    "no_escort_log_diffusion": NoEscortLogDiffusion,
}

# The earlier additions can be evaluated as frozen references without making
# them eligible candidates in this follow-up's selection.
COLLECTION_POLICIES = {**targeted_policies.POLICIES, **POLICIES}

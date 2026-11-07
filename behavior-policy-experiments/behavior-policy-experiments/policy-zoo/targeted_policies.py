"""View-based controllers designed to preserve an ordered distance code.

The world still has one distance and three actions. These controllers use the
known height = 1/d geometry, Gaussian random targets, and private policy memory.
They do not give simulator labels to the encoder or change the reset rule.
"""

import math

import torch

from zoo import Policy


MIN_DISTANCE = 1.0
MAX_DISTANCE = 16.0
MOVE = 0.25
LOG_CENTER = math.log(4.0)
ESCORT_STEPS = 64


def observed_distance(view):
    """Infer distance from the visible triangle height, not simulator state."""
    return view[:, 1].clamp_min(1e-8).reciprocal()


def candidates(view):
    distance = observed_distance(view)
    proposed = distance[:, None] + torch.tensor([-MOVE, 0.0, MOVE])
    feasible = (proposed >= MIN_DISTANCE) & (proposed <= MAX_DISTANCE)
    return proposed, feasible


def truncated_log_targets(n, standard_deviation):
    """Sample log-distance targets from the specified bounded Gaussian."""
    normal = torch.distributions.Normal(0.0, 1.0)
    low = normal.cdf(torch.tensor(-LOG_CENTER / standard_deviation))
    high = normal.cdf(torch.tensor((math.log(MAX_DISTANCE) - LOG_CENTER) / standard_deviation))
    probability = low + (high - low) * torch.rand(n)
    return LOG_CENTER + standard_deviation * normal.icdf(probability)


def move_toward(view, target_log_distance):
    proposed, feasible = candidates(view)
    error = (proposed.clamp_min(1e-8).log() - target_log_distance[:, None]).abs()
    error[~feasible] = float("inf")
    return error.argmin(dim=1)


class GaussianStart(Policy):
    """Use 64 of the existing burn-in actions to approach sampled size goals."""

    def __init__(self, standard_deviation=0.55):
        self.standard_deviation = standard_deviation

    def reset(self, n):
        self.steps = 0
        self.target = truncated_log_targets(n, self.standard_deviation)

    def remember(self, view):
        pass

    def act(self, view):
        self.remember(view)
        if self.steps < ESCORT_STEPS:
            action = move_toward(view, self.target)
        else:
            action = self.after_escort(view)
        self.steps += 1
        return action

    def after_escort(self, view):
        _, feasible = candidates(view)
        return torch.multinomial(feasible.float(), 1).flatten()


class GaussianMetropolis(GaussianStart):
    """A reversible physical-step walk targeting a Gaussian in log distance.

    A distance-dependent proposal activity requires the proposal ratio in the
    acceptance probability. Omitting it would change the stationary density.
    """

    def __init__(self, standard_deviation=0.50, log_diffusion=False):
        super().__init__(standard_deviation)
        self.log_diffusion = log_diffusion

    def activity(self, distance):
        if self.log_diffusion:
            return (distance / 8.0).square().clamp(max=1.0)
        return torch.ones_like(distance)

    def log_density(self, distance):
        q = distance.log()
        # The -log(d) term is the Jacobian from log distance to physical distance.
        return -0.5 * ((q - LOG_CENTER) / self.standard_deviation).square() - q

    def log_acceptance(self, distance, proposed):
        density_ratio = self.log_density(proposed) - self.log_density(distance)
        proposal_ratio = self.activity(proposed).log() - self.activity(distance).log()
        return density_ratio + proposal_ratio

    def after_escort(self, view):
        distance = observed_distance(view)
        direction = 2 * torch.randint(0, 2, distance.shape) - 1
        proposed = distance + direction * MOVE
        feasible = (proposed >= MIN_DISTANCE) & (proposed <= MAX_DISTANCE)
        active = torch.rand(len(distance)) < self.activity(distance)
        accepted = torch.rand(len(distance)).log() < self.log_acceptance(distance, proposed)
        move = feasible & active & accepted
        return torch.where(move, direction + 1, torch.ones_like(direction))


class GaussianTargetTracking(GaussianStart):
    """Follow a smoothly varying random apparent-size target."""

    def __init__(self, correlation):
        super().__init__(standard_deviation=0.55)
        self.correlation = correlation

    def after_escort(self, view):
        innovation = self.standard_deviation * math.sqrt(1 - self.correlation**2)
        self.target = (
            LOG_CENTER
            + self.correlation * (self.target - LOG_CENTER)
            + innovation * torch.randn_like(self.target)
        )
        return move_toward(view, self.target)


class GaussianDeficit(GaussianStart):
    """Visit log-size bins in Gaussian proportions using shared visit counts."""

    def reset(self, n):
        super().reset(n)
        self.counts = torch.zeros(32)
        edges = torch.linspace(0.0, math.log(MAX_DISTANCE), 33)
        normal = torch.distributions.Normal(LOG_CENTER, self.standard_deviation)
        self.target_mass = torch.diff(normal.cdf(edges))
        self.target_mass /= self.target_mass.sum()

    @staticmethod
    def bins(distance):
        fraction = distance.clamp_min(1e-8).log() / math.log(MAX_DISTANCE)
        return (32 * fraction).long().clamp(0, 31)

    def remember(self, view):
        current = self.bins(observed_distance(view))
        self.counts += torch.bincount(current, minlength=32)

    def after_escort(self, view):
        proposed, feasible = candidates(view)
        bins = self.bins(proposed)
        score = self.target_mass[bins].clamp_min(1e-12).log() - (1 + self.counts[bins]).log()
        score[~feasible] = -float("inf")
        probabilities = torch.softmax(score / 0.25, dim=1)
        uniform_feasible = feasible.float() / feasible.sum(dim=1, keepdim=True)
        probabilities = 0.9 * probabilities + 0.1 * uniform_feasible
        return torch.multinomial(probabilities, 1).flatten()


POLICIES = {
    "gaussian_walk": lambda: GaussianMetropolis(0.50, log_diffusion=False),
    "gaussian_log_diffusion": lambda: GaussianMetropolis(0.50, log_diffusion=True),
    "gaussian_wide_diffusion": lambda: GaussianMetropolis(0.65, log_diffusion=True),
    "gaussian_target_fast": lambda: GaussianTargetTracking(0.98),
    "gaussian_target_slow": lambda: GaussianTargetTracking(0.995),
    "gaussian_deficit": GaussianDeficit,
    "gaussian_start_random": GaussianStart,
}

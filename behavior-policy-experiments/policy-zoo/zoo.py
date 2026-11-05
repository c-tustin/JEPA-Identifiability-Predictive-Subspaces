"""Behavior policies and Gaussian sources for the policy-zoo comparison.

Physical policies act in the unchanged three-action distance world (approach,
stay, retreat by 0.25). Latent sources are exact stationary Gaussian processes
in z = log(d/4)/0.7: OU trajectories passed through linear low/high-pass
filters. No policy or source sees true distance; closed-loop rules read only
the rendered triangle height x[:, 1] = 1/d.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reliability"))
import policies as learned
import run_study as reliability

behavior = reliability.behavior
controls = reliability.controls

STAY_THRESHOLD = 0.4307272992954576  # Phi^{-1}(2/3): uniform action marginal
WARMUP = 1000  # filter warm-up, so latent/action noise starts stationary


# ---------------------------------------------------------------- filters
def impulse_response(stages, rho=0.9, taps=20000):
    """Response of OU(rho) followed by first-order filter stages to one shock."""
    x = rho ** np.arange(taps)
    for kind, w in stages:
        a, y = np.exp(-w), np.zeros(taps)
        for t in range(taps):
            previous = y[t - 1] if t else 0.0
            if kind == "lp":
                y[t] = a * previous + (1 - a) * x[t]
            else:
                y[t] = a * (previous + x[t] - (x[t - 1] if t else 0.0))
        x = y
    return x


def filter_stats(stages, rho=0.9):
    """Stationary standard deviation and lag-one correlation (unit shocks)."""
    h = impulse_response(stages, rho)
    return float(np.sqrt(h @ h)), float((h[:-1] @ h[1:]) / (h @ h))


class FilteredOU:
    """Vectorized OU(rho) trajectories passed through causal filter stages.

    lp: y_t = a y_{t-1} + (1-a) x_t;  hp: y_t = a (y_{t-1} + x_t - x_{t-1}),
    with a = exp(-w). Output is standardized to unit stationary variance, so
    every source is exactly Gaussian with N(0,1) marginals after warm-up.
    """

    def __init__(self, n, stages, rho=0.9, generator=None):
        self.stages, self.rho, self.generator = stages, rho, generator
        self.scale = filter_stats(stages, rho)[0]
        self.z = torch.randn(n, generator=generator)  # stationary OU start
        self.state = [(torch.zeros(n), torch.zeros(n)) for _ in stages]
        self.value = None
        for _ in range(WARMUP):
            self.step()

    def step(self):
        noise = torch.randn(len(self.z), generator=self.generator)
        self.z = self.rho * self.z + (1 - self.rho**2) ** 0.5 * noise
        # The OU state has unit variance; impulse_response assumes unit shocks.
        x = self.z / (1 - self.rho**2) ** 0.5
        for i, (kind, w) in enumerate(self.stages):
            a = float(np.exp(-w))
            y_prev, x_prev = self.state[i]
            y = a * y_prev + (1 - a) * x if kind == "lp" else a * (y_prev + x - x_prev)
            self.state[i] = (y, x)
            x = y
        self.value = x / self.scale
        return self.value


# ------------------------------------------------------- physical policies
class Policy:
    """Interface: reset for a batch of trajectories, then act on views."""

    def reset(self, n):
        pass

    def act(self, x):
        raise NotImplementedError


class Sticky(Policy):
    """Repeat the previous action with probability p; else uniform."""

    def __init__(self, p=0.9):
        self.p = p

    def reset(self, n):
        self.action = torch.randint(0, 3, (n,))

    def act(self, x):
        fresh = torch.randint(0, 3, (len(x),))
        keep = torch.rand(len(x)) < self.p
        self.action = torch.where(keep, self.action, fresh)
        return self.action


class Lazy(Policy):
    """Stay with probability 0.8; approach/retreat 0.1 each."""

    def __init__(self, probs=(0.1, 0.8, 0.1)):
        self.probs = torch.tensor(probs)

    def act(self, x):
        return torch.multinomial(self.probs.expand(len(x), 3), 1).flatten()


class Biased(Policy):
    """Approach-biased i.i.d. actions; net drift toward the object."""

    def __init__(self, probs=(0.36, 0.33, 0.31)):
        self.probs = torch.tensor(probs)

    def act(self, x):
        return torch.multinomial(self.probs.expand(len(x), 3), 1).flatten()


class Levy(Policy):
    """Uniform actions held for Pareto(alpha) run lengths (heavy tails)."""

    def __init__(self, alpha=1.5, cap=1000):
        self.alpha, self.cap = alpha, cap

    def _runs(self, n):
        u = torch.rand(n).clamp_min(1e-12)
        return u.pow(-1 / self.alpha).floor().clamp(1, self.cap).long()

    def reset(self, n):
        self.action, self.left = torch.randint(0, 3, (n,)), self._runs(n)

    def act(self, x):
        done = self.left <= 0
        self.action = torch.where(done, torch.randint(0, 3, (len(x),)), self.action)
        self.left = torch.where(done, self._runs(len(x)), self.left) - 1
        return self.action


class Sweep(Policy):
    """Move one way until the view nears a sensor-range edge, then reverse.

    Reads only apparent height. 10% of actions are uniform random.
    """

    def __init__(self, epsilon=0.1, near=0.9, far=1 / 15):
        self.epsilon, self.near, self.far = epsilon, near, far

    def reset(self, n):
        self.direction = torch.where(torch.rand(n) < 0.5, -1, 1)

    def act(self, x):
        height = x[:, 1]
        self.direction = torch.where(height >= self.near, 1, self.direction)
        self.direction = torch.where(height <= self.far, -1, self.direction)
        action = 1 + self.direction
        random = torch.rand(len(x)) < self.epsilon
        return torch.where(random, torch.randint(0, 3, (len(x),)), action)


class Homing(Policy):
    """Hold the view near apparent height 1/4 (d=4); 30% uniform actions."""

    def __init__(self, target=0.25, band=0.1, epsilon=0.3):
        self.target, self.band, self.epsilon = target, band, epsilon

    def act(self, x):
        error = torch.log(x[:, 1] / self.target)
        action = torch.ones(len(x), dtype=torch.long)
        action = torch.where(error < -self.band, 0, action)  # too small: approach
        action = torch.where(error > self.band, 2, action)  # too large: retreat
        random = torch.rand(len(x)) < self.epsilon
        return torch.where(random, torch.randint(0, 3, (len(x),)), action)


class ColoredActions(Policy):
    """Quantize a (filtered) OU signal into approach/stay/retreat.

    Thresholds +-Phi^{-1}(2/3) give each action marginal probability 1/3,
    matching uniform random actions; only temporal correlation changes.
    """

    def __init__(self, stages):
        self.stages = stages

    def reset(self, n):
        self.noise = FilteredOU(n, self.stages)

    def act(self, x):
        u = self.noise.step()
        action = torch.ones(len(x), dtype=torch.long)
        action = torch.where(u < -STAY_THRESHOLD, 0, action)
        return torch.where(u > STAY_THRESHOLD, 2, action)


@torch.no_grad()
def collect_physical(c, policy, episodes, seed):
    """Same trajectory structure as behavior.collect: burn-in then horizon."""
    torch.manual_seed(seed)
    distance = behavior.initial(episodes, c)
    policy.reset(episodes)
    data = {k: [] for k in ["x", "xp", "z", "zp", "action"]}
    for t in range(c.burnin + c.horizon):
        x = behavior.sensor(distance, c)
        action = policy.act(x)
        successor = behavior.move(distance, action, c)
        if t >= c.burnin:
            for k, v in {
                "x": x,
                "xp": behavior.sensor(successor, c),
                "z": torch.log(distance / 4) / c.sigma,
                "zp": torch.log(successor / 4) / c.sigma,
                "action": action,
            }.items():
                data[k].append(v)
        distance = successor
    return {k: torch.cat(v) for k, v in data.items()}


# ---------------------------------------------------------- latent sources
@torch.no_grad()
def collect_latent(c, stages, seed):
    """Consecutive pairs from filtered-OU trajectories, rendered as views."""
    generator = torch.Generator().manual_seed(10000 + seed)
    process = FilteredOU(c.train_episodes, stages, generator=generator)
    z = [process.step().clone() for _ in range(c.burnin + c.horizon + 1)]
    z = torch.stack(z[c.burnin :])  # (horizon + 1, episodes)
    zt, zp = z[:-1].flatten(), z[1:].flatten()
    return {
        "x": behavior.observe(zt, c.sigma),
        "xp": behavior.observe(zp, c.sigma),
        "z": zt,
        "zp": zp,
        "action": torch.full((len(zt),), -1, dtype=torch.long),
    }


@torch.no_grad()
def collect_pairs(c, rho, seed):
    """Independent Gaussian pairs with correlation rho (ou_data RNG stream).

    With rho = 0.9 this reproduces the fixed-OU training pairs exactly.
    """
    generator = torch.Generator().manual_seed(10000 + seed)
    n = c.train_episodes * c.horizon
    z, noise = torch.randn(n, generator=generator), torch.randn(n, generator=generator)
    zp = rho * z + (1 - rho**2) ** 0.5 * noise
    return {
        "x": behavior.observe(z, c.sigma),
        "xp": behavior.observe(zp, c.sigma),
        "z": z,
        "zp": zp,
        "action": torch.full((n,), -1, dtype=torch.long),
    }


# ------------------------------------------------------------- learned task
def task_reward(xp, target=0.25, width=0.35):
    """Extrinsic reward for holding apparent height near 1/4; in [0, 1]."""
    return torch.exp(-(torch.log(xp[:, 1].clamp_min(1e-8) / target) / width) ** 2)


# -------------------------------------------------------------- registry
LATENT_FILTERS = {
    "ou_traj": [],
    "ou_lp_w1": [("lp", 1.0)],
    "ou_lp_w0.3": [("lp", 0.3)],
    "ou_lp_w0.1": [("lp", 0.1)],
    "ou_hp_w0.1": [("hp", 0.1)],
    "ou_hp_w0.3": [("hp", 0.3)],
    "ou_hp_w1": [("hp", 1.0)],
    "ou_hp2_w1": [("hp", 1.0), ("hp", 1.0)],
    "ou_hp2_w3": [("hp", 3.0), ("hp", 3.0)],
}

PHYSICAL_RULES = {
    "sticky": lambda: Sticky(),
    "lazy": lambda: Lazy(),
    "biased": lambda: Biased(),
    "levy": lambda: Levy(),
    "sweep": lambda: Sweep(),
    "homing": lambda: Homing(),
    "ou_actions": lambda: ColoredActions([]),
    "ou_actions_lp_w0.1": lambda: ColoredActions([("lp", 0.1)]),
    "ou_actions_hp_w0.3": lambda: ColoredActions([("hp", 0.3)]),
    "ou_actions_hp2_w3": lambda: ColoredActions([("hp", 3.0), ("hp", 3.0)]),
}

# Previously tested sources, regenerated with the reliability-study code.
EXISTING = ["random", "icm", "rnd", "disagreement", "count", "action_balanced"]
LEARNED = ["icm", "rnd", "disagreement", "task"]
PHYSICAL = EXISTING + list(PHYSICAL_RULES) + ["task"]


def matched_name(name):
    return f"iid_{name}"


LATENT = (
    ["ou"]
    + list(LATENT_FILTERS)
    + [matched_name(k) for k in LATENT_FILTERS if k != "ou_traj"]
)


def source_rho(name):
    if name == "ou":
        return 0.9
    base = name[4:] if name.startswith("iid_") else name
    return filter_stats(LATENT_FILTERS[base])[1]

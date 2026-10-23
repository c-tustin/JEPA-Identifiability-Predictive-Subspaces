"""Observation-only exploration alternatives in the unchanged distance world.

Actor/ICM equations follow our source-derived Pathak port; see notices in the
repository root. RND and ensemble bonuses are small toy implementations of the
published ideas, not ports of their reference software.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "curiosity"))
import experiment as behavior
import torch
from torch import nn
from torch.nn import functional as F


class ExploratoryPolicy(behavior.Policy):
    def __init__(self, epsilon=0.):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        distribution, value = super().forward(x)
        if self.epsilon:
            distribution = torch.distributions.Categorical(probs=(1 - self.epsilon) * distribution.probs + self.epsilon / 3)
        return distribution, value


def size_bin(x, bins=32):
    """Known apparent-size ordering: log triangle height, no distance labels.

    Fixed sensor range is [1/16,1]. This deliberately uses scene geometry; it is
    a coverage heuristic, not evidence the policy inferred a latent coordinate.
    """
    fraction = (x[:, 1].clamp_min(1e-8).log() + torch.log(torch.tensor(16.))) / torch.log(torch.tensor(16.))
    return (fraction * bins).long().clamp(0, bins - 1)


class CountBonus:
    def __init__(self):
        self.counts = torch.zeros(32)

    def reward(self, x):
        ix = size_bin(x)
        # Give a simultaneous rollout batch the same pre-update count.
        result = .1 / (1 + self.counts[ix]).sqrt()
        self.counts += torch.bincount(ix, minlength=32)
        return result


class ActionBalancedPolicy:
    """Adaptive action-count heuristic; its memory is reset per collection."""
    def __init__(self):
        self.counts = torch.zeros(32, 3)

    def sample(self, x):
        ix = size_bin(x)
        probabilities = torch.softmax(-self.counts[ix], -1)
        probabilities = .9 * probabilities + .1 / 3
        action = torch.distributions.Categorical(probs=probabilities).sample()
        self.counts += torch.bincount(ix * 3 + action, minlength=96).reshape(32, 3)
        return action


class RND(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(6, 64), nn.ELU(), nn.Linear(64, 32))
        self.predictor = nn.Sequential(nn.Linear(6, 64), nn.ELU(), nn.Linear(64, 32))
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def errors(self, x, xp, a):
        view = torch.tanh(4 * xp)
        return (self.predictor(view) - self.target(view)).square().mean(-1)


class Disagreement(nn.Module):
    def __init__(self, members=5):
        super().__init__()
        self.models = nn.ModuleList([nn.Sequential(nn.Linear(9, 64), nn.ELU(), nn.Linear(64, 6)) for _ in range(members)])

    def predictions(self, x, a):
        inputs = torch.cat([torch.tanh(4 * x), F.one_hot(a, 3).float()], -1)
        return torch.stack([model(inputs) for model in self.models])

    def errors(self, x, xp, a):
        return self.predictions(x, a).var(0, unbiased=False).mean(-1)


def train(c, seed, kind, policy_factory=None):
    """Same A2C interaction budget as ICM; all rewards are detached for actor."""
    torch.manual_seed(seed)
    epsilon = .2 if kind == "icm_epsilon" else 0.
    policy = ExploratoryPolicy(epsilon) if policy_factory is None else policy_factory()
    # Instantiate ICM even for alternative bonuses to preserve actor RNG setup.
    icm = behavior.ICM()
    bonus = RND() if kind == "rnd" else Disagreement() if kind == "disagreement" else None
    actor_groups = policy.optimizer_groups() if hasattr(policy, "optimizer_groups") else policy.parameters()
    actor_opt = torch.optim.Adam(actor_groups, lr=1e-4)
    intrinsic_model = icm if bonus is None else bonus
    intrinsic_opt = torch.optim.Adam([p for p in intrinsic_model.parameters() if p.requires_grad], lr=1e-3)
    counts = CountBonus()
    distance, age = behavior.initial(c.nenv, c), torch.zeros(c.nenv, dtype=torch.long)
    reward_second_moment = 1e-4
    history = []
    for iteration in range(c.policy_updates):
        observations, successors, actions, rewards, terminals = [], [], [], [], []
        for _ in range(c.rollout):
            x = behavior.sensor(distance, c)
            with torch.no_grad():
                distribution, _ = policy(x)
                a = distribution.sample()
                successor = behavior.move(distance, a, c)
                xp = behavior.sensor(successor, c)
                if kind == "count":
                    reward = counts.reward(xp)
                elif bonus is not None:
                    error = bonus.errors(x, xp, a)
                    reward_second_moment = .99 * reward_second_moment + .01 * float(error.square().mean())
                    reward = (.1 * error / max(reward_second_moment**.5, 1e-8)).clamp(max=1)
                else:
                    reward = (c.reward_scale * icm(x, xp, a)[0]).clamp(max=1)
                    if kind == "icm_coverage":
                        reward = (reward + counts.reward(xp)).clamp(max=1)
            age += 1
            done = age >= 256
            observations.append(x); successors.append(xp); actions.append(a)
            rewards.append(reward); terminals.append(done)
            distance = torch.where(done, behavior.initial(c.nenv, c), successor)
            age = torch.where(done, torch.zeros_like(age), age)
        x, xp, a = torch.cat(observations), torch.cat(successors), torch.cat(actions)
        with torch.no_grad():
            bootstrap = policy(behavior.sensor(distance, c))[1]
            returns = behavior.discounted_returns(torch.stack(rewards), torch.stack(terminals), bootstrap, c.gamma).flatten()
        distribution, value = policy(x)
        advantage = returns - value.detach()
        actor_loss = -(distribution.log_prob(a) * advantage).mean()
        actor_loss += .25 * (value - returns).square().mean() - c.entropy_weight * distribution.entropy().mean()
        actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 40.); actor_opt.step()
        if kind != "count":
            if bonus is None:
                error, inverse = icm(x, xp, a)
                intrinsic_loss = (1 - c.forward_weight) * F.cross_entropy(inverse, a) + c.forward_weight * error.mean()
            elif kind == "rnd":
                intrinsic_loss = bonus.errors(x, xp, a).mean()
            else:
                predictions = bonus.predictions(x, a)
                # Independently bootstrap transitions for each ensemble member.
                member_losses = []
                for member in range(len(bonus.models)):
                    sampled = torch.randint(len(x), (len(x),))
                    member_losses.append((predictions[member, sampled] - torch.tanh(4 * xp[sampled])).square().mean())
                intrinsic_loss = torch.stack(member_losses).mean()
            intrinsic_opt.zero_grad(); intrinsic_loss.backward()
            nn.utils.clip_grad_norm_(intrinsic_model.parameters(), 40.); intrinsic_opt.step()
        if iteration % 100 == 0 or iteration == c.policy_updates - 1:
            history.append({"update": iteration, "mean_reward": float(torch.stack(rewards).mean()),
                            "entropy": float(distribution.entropy().detach().mean())})
    return policy.eval(), {"policy": policy.state_dict(), "epsilon": epsilon, "kind": kind,
                           "intrinsic_model": intrinsic_model.state_dict(), "history": history,
                           "config": vars(c)}


def load(saved):
    if saved["kind"] == "action_balanced":
        return ActionBalancedPolicy()
    policy = ExploratoryPolicy(saved.get("epsilon", 0.))
    policy.load_state_dict(saved["policy"])
    return policy.eval()


@torch.no_grad()
def collect(c, kind, episodes, seed, policy=None):
    if kind != "action_balanced":
        return behavior.collect(c, "icm", episodes, seed, policy)
    torch.manual_seed(seed)
    policy = ActionBalancedPolicy()
    distance = behavior.initial(episodes, c)
    data = {k: [] for k in ["x", "xp", "z", "zp", "action"]}
    for step in range(c.burnin + c.horizon):
        x = behavior.sensor(distance, c)
        a = policy.sample(x)
        successor = behavior.move(distance, a, c)
        if step >= c.burnin:
            for k, value in {"x": x, "xp": behavior.sensor(successor, c), "z": torch.log(distance / 4) / c.sigma,
                             "zp": torch.log(successor / 4) / c.sigma, "action": a}.items():
                data[k].append(value)
        distance = successor
    return {k: torch.cat(values) for k, values in data.items()}

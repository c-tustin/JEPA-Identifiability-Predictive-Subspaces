"""Compare SIGReg identifiability under OU, random actions, and learned ICM.

ICM and actor-critic ported from pathak22/noreward-rl at
3e220c2177fc253916f12d980957fc40579d577a; see upstream/ and README.md.
No TensorFlow/game packages are executed. Policy/ICM/JEPA train on observations
and actions only. Physical coordinates are restricted to simulation/evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identifiability"))
from reproduce import Encoder, SIGReg1D, observe

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    policy_updates: int = 600
    jepa_steps: int = 4000
    nenv: int = 16
    rollout: int = 20
    train_episodes: int = 256
    eval_episodes: int = 64
    horizon: int = 64
    burnin: int = 128
    batch: int = 512
    distance_min: float = 1.
    distance_max: float = 16.
    move: float = .25
    sigma: float = .7
    rho: float = .9
    sigreg_weight: float = .1
    forward_weight: float = .2
    reward_scale: float = .2
    entropy_weight: float = .0005
    gamma: float = .99


def sensor(distance, c):
    # Exactly the baseline renderer: V/d. The log is inside the simulator only.
    return observe(torch.log(distance / 4) / c.sigma, c.sigma)


def move(distance, action, c):
    proposed = distance + (action.float() - 1) * c.move
    # Reflect overshoot, rather than creating sticky clipped boundary states.
    proposed = torch.where(proposed < c.distance_min, 2 * c.distance_min - proposed, proposed)
    return torch.where(proposed > c.distance_max, 2 * c.distance_max - proposed, proposed)


def initial(n, c):
    return c.distance_min + (c.distance_max - c.distance_min) * torch.rand(n)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(6, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x):
        hidden = self.body(torch.tanh(4 * x))
        return torch.distributions.Categorical(logits=self.actor(hidden)), self.critic(hidden).flatten()


class ICM(nn.Module):
    """Port of StateActionPredictor: shared phi, inverse CE, forward L2 sum.

    Both phi branches receive forward-loss gradients, matching upstream (which
    does not stop-gradient phi2). Actor rewards are detached separately.
    Replaces game CNNs by an MLP; feature width 32 replaces upstream width 288.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(6, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU())
        self.inverse = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        self.forward_model = nn.Sequential(nn.Linear(35, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, x, xp, action):
        phi = self.features(torch.tanh(4 * x))
        phip = self.features(torch.tanh(4 * xp))
        logits = self.inverse(torch.cat([phi, phip], -1))
        prediction = self.forward_model(torch.cat([phi, F.one_hot(action, 3).float()], -1))
        error = .5 * (prediction - phip).square().sum(-1)
        return error, logits


def discounted_returns(rewards, done, bootstrap, gamma):
    result = torch.empty_like(rewards)
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * (1 - done[t].float()) * running
        result[t] = running
    return result


def train_policy(c, seed, entropy_schedule=None):
    torch.manual_seed(seed)
    policy, icm = Policy(), ICM()
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)
    distance, age = initial(c.nenv, c), torch.zeros(c.nenv, dtype=torch.long)
    history = []
    initial_weights = torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
    for update in range(c.policy_updates):
        observations, successors, actions, rewards, terminals = [], [], [], [], []
        for _ in range(c.rollout):
            x = sensor(distance, c)
            with torch.no_grad():
                distribution, _ = policy(x)
                action = distribution.sample()
                next_distance = move(distance, action, c)
                xp = sensor(next_distance, c)
                error, _ = icm(x, xp, action)
                # Intrinsic-only reward, collected before updating the predictor.
                reward = (c.reward_scale * error).clamp(max=1.)
            age += 1
            done = age >= 256
            observations.append(x)
            successors.append(xp)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)
            distance = torch.where(done, initial(c.nenv, c), next_distance)
            age = torch.where(done, torch.zeros_like(age), age)
        x, xp = torch.cat(observations), torch.cat(successors)
        action = torch.cat(actions)
        with torch.no_grad():
            _, bootstrap = policy(sensor(distance, c))
            returns = discounted_returns(torch.stack(rewards), torch.stack(terminals), bootstrap, c.gamma).flatten()
        distribution, value = policy(x)
        advantage = returns - value.detach()
        # Matches upstream pi_loss + .5*vf_loss - entropy_beta*entropy,
        # with vf_loss=.5*mean((V-return)^2), no advantage normalization.
        actor_loss = -(distribution.log_prob(action) * advantage).mean()
        entropy_weight = c.entropy_weight if entropy_schedule is None else entropy_schedule(update, c.policy_updates)
        actor_loss += .25 * (value - returns).square().mean() - entropy_weight * distribution.entropy().mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 40.)
        actor_optimizer.step()
        error, inverse_logits = icm(x, xp, action)
        inverse_loss = F.cross_entropy(inverse_logits, action)
        icm_loss = (1 - c.forward_weight) * inverse_loss + c.forward_weight * error.mean()
        icm_optimizer.zero_grad()
        icm_loss.backward()
        nn.utils.clip_grad_norm_(icm.parameters(), 40.)
        icm_optimizer.step()
        if update % 25 == 0 or update == c.policy_updates - 1:
            history.append({"update": update, "intrinsic_reward": float(torch.stack(rewards).mean()),
                            "forward_error": float(error.detach().mean()),
                            "inverse_accuracy": float((inverse_logits.argmax(-1) == action).float().mean()),
                            "action_majority_fraction": float(torch.bincount(action, minlength=3).max() / len(action)),
                            "entropy_weight": float(entropy_weight),
                            "entropy": float(distribution.entropy().detach().mean())})
        if update % 200 == 0:
            print(f"  policy update {update}/{c.policy_updates}, reward={history[-1]['intrinsic_reward']:.5f}", flush=True)
    final_weights = torch.cat([p.detach().flatten() for p in policy.parameters()])
    return policy.eval(), icm.eval(), history, float((final_weights - initial_weights).norm())


@torch.no_grad()
def collect(c, kind, episodes, seed, policy=None):
    torch.manual_seed(seed)
    n = episodes * c.horizon
    if kind == "ou":
        z = torch.randn(n)
        zp = c.rho * z + (1 - c.rho**2)**.5 * torch.randn(n)
        return {"x": observe(z, c.sigma), "xp": observe(zp, c.sigma), "z": z, "zp": zp,
                "action": torch.full((n,), -1, dtype=torch.long)}
    distance = initial(episodes, c)
    collected = {key: [] for key in ["x", "xp", "z", "zp", "action"]}
    # Every vector element is a separate trajectory. Burn-in is discarded.
    # Frozen policy and fixed episode length; resets are never positive pairs.
    for t in range(c.burnin + c.horizon):
        x = sensor(distance, c)
        action = policy(x)[0].sample() if kind == "icm" else torch.randint(0, 3, (episodes,))
        successor = move(distance, action, c)
        if t >= c.burnin:
            for key, value in {"x": x, "xp": sensor(successor, c),
                               "z": torch.log(distance / 4) / c.sigma,
                               "zp": torch.log(successor / 4) / c.sigma, "action": action}.items():
                collected[key].append(value)
        distance = successor
    return {key: torch.cat(value) for key, value in collected.items()}


def train_encoder(data, c, seed):
    # Representation fitting accesses only x/xp, never latent coordinates/actions.
    torch.manual_seed(seed)
    model, regularizer = Encoder(), SIGReg1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for step in range(c.jepa_steps):
        phase = max(0., (step - c.jepa_steps / 2) / max(1., c.jepa_steps / 2 - 1))
        optimizer.param_groups[0]["lr"] = .001 * .5 * (1 + np.cos(np.pi * phase))
        ix = torch.randint(len(data["x"]), (c.batch,))
        y = model(torch.cat([data["x"][ix], data["xp"][ix]]))
        alignment = (y[:c.batch] - y[c.batch:]).square().mean()
        sigreg = regularizer(y)
        loss = (1 - c.sigreg_weight) * alignment + c.sigreg_weight * sigreg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == c.jepa_steps - 1:
            history.append([step, float(alignment.detach()), float(sigreg.detach())])
    return model.eval(), history


def affine_score(x_cal, z_cal, x_test, z_test):
    coefficient = np.linalg.lstsq(np.column_stack([x_cal, np.ones(len(x_cal))]), z_cal, rcond=None)[0]
    prediction = np.column_stack([x_test, np.ones(len(x_test))]) @ coefficient
    return float(1 - np.mean((prediction - z_test)**2) / np.var(z_test))


def distribution_stats(z, zp):
    normalized = (z - z.mean()) / z.std(unbiased=False).clamp_min(1e-8)
    sorted_z = normalized.sort().values
    cdf = .5 * (1 + torch.erf(sorted_z / 2**.5))
    n = len(z)
    ks = max(float((torch.arange(1, n + 1) / n - cdf).max()),
             float((cdf - torch.arange(n) / n).max()))
    return {"mean": float(z.mean()), "variance": float(z.var(unbiased=False)),
            "skewness": float(normalized.pow(3).mean()),
            "excess_kurtosis": float(normalized.pow(4).mean() - 3),
            "standardized_gaussian_cdf_distance": ks,
            "lag_correlation": float(torch.corrcoef(torch.stack([z, zp]))[0, 1]),
            "paired_marginal_mean_shift": float(zp.mean() - z.mean())}


@torch.no_grad()
def evaluate_encoder(model, calibration, test, c, seed):
    yc, yt = model(calibration["x"]).numpy(), model(test["x"]).numpy()
    zc, zt = calibration["z"].numpy(), test["z"].numpy()
    stats = {"on_policy_log_distance_r2": affine_score(yc, zc, yt, zt),
             "on_policy_physical_distance_r2": affine_score(yc, np.exp(c.sigma * zc), yt, np.exp(c.sigma * zt)),
             "calibration_embedding_mean": float(yc.mean()),
             "calibration_embedding_variance": float(yc.var()),
             "raw_embedding_mean": float(yt.mean()), "raw_embedding_variance": float(yt.var()),
             "heldout_gaussian_cf_discrepancy": float(SIGReg1D()(torch.from_numpy(yt)) / len(yt)),
             "raw_alignment": float((model(test["xp"]) - torch.from_numpy(yt)).square().mean()),
             "latent_distribution": distribution_stats(test["z"], test["zp"])}
    # Shared evaluation support for every condition, independent of behavior data.
    # Fit the affine probe on a separate random calibration set; test on a grid.
    low = np.log(c.distance_min / 4) / c.sigma
    high = np.log(c.distance_max / 4) / c.sigma
    torch.manual_seed(seed + 800000)
    zg_cal = low + (high - low) * torch.rand(4096)
    zg = torch.linspace(low, high, 1024)
    yg_cal, yg = model(observe(zg_cal, c.sigma)).numpy(), model(observe(zg, c.sigma)).numpy()
    stats["common_log_distance_grid_r2"] = affine_score(yg_cal, zg_cal.numpy(), yg, zg.numpy())
    normalized_grid = (yg - yg_cal.mean()) / max(yg_cal.std(), 1e-8)
    sign = np.sign(np.cov(yg_cal, zg_cal.numpy())[0, 1])
    stats["grid_calibrated_variance"] = float(normalized_grid.var())
    distances = 4 * torch.exp(c.sigma * test["z"])
    inside = (distances >= c.distance_min) & (distances <= c.distance_max)
    counts = torch.histc(distances[inside], bins=32, min=c.distance_min, max=c.distance_max)
    stats["physical_range_bin_coverage"] = float((counts > 0).float().mean())
    stats["fraction_in_common_distance_range"] = float(inside.float().mean())
    return stats, {"z": zg.numpy(), "embedding": sign * normalized_grid,
                   "samples": test["z"].numpy(), "distance": distances.numpy()}


def plot(out, runs, examples, policy_histories):
    colors = {"ou": "#0e7490", "random": "#7c3aed", "icm": "#c2410c"}
    names = {"ou": "OU baseline", "random": "Random actions", "icm": "ICM behavior policy"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for kind in colors:
        example = examples[kind]
        axes[0, 0].hist(example["samples"], bins=45, density=True, histtype="step", lw=2,
                        color=colors[kind], label=names[kind])
        axes[0, 1].plot(example["z"], example["embedding"], color=colors[kind], label=names[kind])
    axes[0, 0].set(title="Collected log-distance distributions (seed 0)", xlabel="z = log(d/4) / 0.7", ylabel="Density")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Learned coordinates on the same distance range", xlabel="True log-distance coordinate z", ylabel="Embedding (affine calibration for display)")
    axes[0, 1].legend(fontsize=8)
    for i, kind in enumerate(colors):
        for j, metric in enumerate(["on_policy_log_distance_r2", "common_log_distance_grid_r2"]):
            values = [r["conditions"][kind][metric] for r in runs]
            axes[1, 0].bar(i + (j - .5) * .3, np.mean(values), width=.28,
                           color=colors[kind], alpha=1 if j == 0 else .4)
            axes[1, 0].scatter(np.full(len(values), i + (j - .5) * .3), values, color="black", s=10)
    axes[1, 0].set(xticks=range(3), xticklabels=[names[k] for k in colors], ylabel="Held-out linear-probe R²",
                   title="Recovery: solid = behavior data, pale = shared grid")
    for i, history in enumerate(policy_histories):
        axes[1, 1].plot([h["update"] for h in history],
                        [h["inverse_accuracy"] - h["action_majority_fraction"] for h in history],
                        alpha=.8, label=f"Seed {runs[i]['seed']}")
    axes[1, 1].axhline(0, color="black", linestyle="--", label="Majority-action reference")
    axes[1, 1].set(title="ICM action prediction beyond class imbalance", xlabel="Actor-critic update", ylabel="Accuracy minus majority-action fraction")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Behavior-policy experiment: identifiability beyond Gaussian OU pairs", fontsize=15)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def verify_upstream():
    root = Path(__file__).parent / "upstream" / "noreward-rl"
    record = json.loads((root / "PROVENANCE.json").read_text())
    for path, expected in record["sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Upstream source integrity failed: {path}")
    return record["commit"]


def write_report(out, result):
    runs = result["runs"]
    def summary(values):
        return f"{np.mean(values):.4f} ± {np.std(values, ddof=1) if len(values) > 1 else 0:.4f}"
    lines = ["# Behavior-policy results", "",
             "Generated from `metrics.json`; mean ± sample standard deviation across seeds. "
             "These are finite-sample recovery measurements, not an identifiability proof.", "",
             f"Seeds: {', '.join(str(r['seed']) for r in runs)}. "
             f"Policy updates per seed: {result['config']['policy_updates']}; "
             f"representation updates per condition: {result['config']['jepa_steps']}.", ""]
    if result["quick"]:
        lines += ["**SMOKE RUN ONLY: models are not trained to convergence. Do not interpret these scores as evidence of identifiability.**", ""]
    lines += ["| Condition | Behavior-distribution log-distance R² | Common-grid log-distance R² | Raw embedding variance |",
              "| --- | ---: | ---: | ---: |"]
    for kind in ["ou", "random", "icm"]:
        rows = [r["conditions"][kind] for r in runs]
        lines.append(f"| {kind} | " + " | ".join(summary([row[key] for row in rows]) for key in
                     ["on_policy_log_distance_r2", "common_log_distance_grid_r2", "raw_embedding_variance"]) + " |")
    lines += ["", "## Per-seed outcomes", "",
              "| Seed | Condition | Behavior R² | Common-grid R² | Raw variance | Gaussian CF discrepancy |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in runs:
        for kind, row in r["conditions"].items():
            lines.append(f"| {r['seed']} | {kind} | {row['on_policy_log_distance_r2']:.4f} | "
                         f"{row['common_log_distance_grid_r2']:.4f} | {row['raw_embedding_variance']:.4f} | "
                         f"{row['heldout_gaussian_cf_discrepancy']:.6f} |")
    lines += ["", "The common-grid probe tests the same physical distance range [1,16] for every condition. "
              "It is fitted on independent evaluation-only calibration samples. "
              "The behavior-distribution probe is fitted on separate calibration trajectories and scored on new episodes.", "",
              "| Condition | Standardized latent Gaussian-CDF distance | Distance-bin coverage |",
              "| --- | ---: | ---: |"]
    for kind in ["ou", "random", "icm"]:
        rows = [r["conditions"][kind] for r in runs]
        lines.append(f"| {kind} | {summary([x['latent_distribution']['standardized_gaussian_cdf_distance'] for x in rows])} | "
                     f"{summary([x['physical_range_bin_coverage'] for x in rows])} |")
    lines += ["", "CDF distances are descriptive; no independent-sample p-values are applied to correlated trajectories.", "",
              "| Seed | Policy weight change (L2) | Mean action-distribution L1 difference from uniform | Final inverse accuracy | Final majority-action fraction |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for r in runs:
        h = r["policy_history"][-1]
        lines.append(f"| {r['seed']} | {r['policy_weight_change_l2']:.4f} | {r['policy_mean_l1_deviation_from_uniform']:.4f} | "
                     f"{h['inverse_accuracy']:.4f} | {h['action_majority_fraction']:.4f} |")
    lines += ["", "## Interpretation", "",
              "The bounded random-action and ICM conditions violate the exact Gaussian-marginal and OU-transition assumptions. "
              "Consequently the Klindt-paper guarantee cannot be asserted for these policies. "
              "The tables quantify how close the trained coordinates remain to affine log-distance mappings in this toy experiment. "
              "A high R² is not a proof of universal identifiability; a lower R² does not refute the original theorem.", "",
              "Differences between random and ICM involve learned action choice and its resulting occupancy, whereas comparisons "
              "to OU also change state support and transition geometry. Coverage alone does not equal a matched occupancy distribution. "
              "Raw embedding variances show that Gaussian regularization is approximate; optimization may contribute to recovery error.", "",
              "For any non-degenerate bounded log-distance distribution, exact Gaussian output and exact affine recovery "
              "of that coordinate cannot both hold: every affine image of a bounded variable is bounded, whereas a standard "
              "Gaussian is not. This incompatibility already applies to the bounded random-action control; it does not "
              "single out curiosity as the cause. The numerical experiments measure approximate recovery instead.", "",
              "![Comparison](comparison.png)", "",
              "See the parent README for the pinned upstream implementation, source-to-port mapping, complete protocol and limitations.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def evaluate_saved(out):
    """Reconstruct held-out trajectories and regenerate reports without training."""
    result = json.loads((out / "metrics.json").read_text())
    c = Config(**result["config"])
    if result["upstream_commit"] != verify_upstream():
        raise RuntimeError("Checkpoint/source provenance mismatch")
    examples = {}
    for index, record in enumerate(result["runs"]):
        seed = record["seed"]
        behavior = torch.load(out / f"behavior_seed{seed}.pt", map_location="cpu")
        policy = Policy()
        policy.load_state_dict(behavior["policy"])
        policy.eval()
        for kind in ["ou", "random", "icm"]:
            calibration = collect(c, kind, c.eval_episodes, 20000 + seed, policy)
            test = collect(c, kind, c.eval_episodes, 30000 + seed, policy)
            model = Encoder()
            model.load_state_dict(torch.load(out / f"encoder_{kind}_seed{seed}.pt", map_location="cpu")["encoder"])
            model.eval()
            metrics, example = evaluate_encoder(model, calibration, test, c, seed)
            # Catch an accidental mismatch with the recorded collection/evaluation.
            for metric in ["on_policy_log_distance_r2", "common_log_distance_grid_r2", "raw_embedding_variance"]:
                if not np.isclose(metrics[metric], record["conditions"][kind][metric], atol=1e-6):
                    raise RuntimeError(f"Stored evaluation did not reproduce: seed={seed}, {kind}, {metric}")
            record["conditions"][kind] = metrics
            if index == 0:
                examples[kind] = example
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(out, result)
    plot(out, result["runs"], examples, [r["policy_history"] for r in result["runs"]])
    print("Reproduced every recorded recovery score and raw variance from saved checkpoints.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--policy-updates", type=int, default=600)
    parser.add_argument("--jepa-steps", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true", help="Tiny execution check; not an identifiability result")
    parser.add_argument("--evaluate-only", action="store_true", help="Reconstruct evaluation and reports from saved checkpoints")
    args = parser.parse_args()
    c = Config(policy_updates=args.policy_updates, jepa_steps=args.jepa_steps)
    if args.quick:
        c.policy_updates, c.jepa_steps = 3, 10
        c.train_episodes, c.eval_episodes, c.horizon, c.burnin = 8, 4, 8, 4
    if c.policy_updates < 1 or c.jepa_steps < 1:
        parser.error("Training step counts must be positive")
    torch.set_num_threads(1)
    if args.evaluate_only:
        evaluate_saved(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    upstream = verify_upstream()
    runs, examples, histories = [], {}, []
    for index, seed in enumerate(args.seeds):
        print(f"seed={seed}: training ICM behavior policy", flush=True)
        policy, icm, policy_history, weight_change = train_policy(c, seed)
        histories.append(policy_history)
        grid = torch.linspace(c.distance_min, c.distance_max, 1024)
        with torch.no_grad():
            probabilities = policy(sensor(grid, c))[0].probs
        record = {"seed": seed, "conditions": {}, "policy_weight_change_l2": weight_change,
                  "policy_mean_l1_deviation_from_uniform": float((probabilities - 1 / 3).abs().sum(-1).mean()),
                  "policy_mean_action_probabilities": probabilities.mean(0).tolist(),
                  "policy_history": policy_history}
        torch.save({"policy": policy.state_dict(), "icm": icm.state_dict(), "config": asdict(c),
                    "upstream_commit": upstream}, args.output / f"behavior_seed{seed}.pt")
        for kind in ["ou", "random", "icm"]:
            train = collect(c, kind, c.train_episodes, 10000 + seed, policy)
            calibration = collect(c, kind, c.eval_episodes, 20000 + seed, policy)
            test = collect(c, kind, c.eval_episodes, 30000 + seed, policy)
            print(f"seed={seed}: training SIGReg encoder, condition={kind}", flush=True)
            model, history = train_encoder(train, c, seed + 40000)
            metrics, example = evaluate_encoder(model, calibration, test, c, seed)
            record["conditions"][kind] = metrics
            if index == 0:
                examples[kind] = example
            torch.save({"encoder": model.state_dict(), "config": asdict(c), "condition": kind,
                        "history": history}, args.output / f"encoder_{kind}_seed{seed}.pt")
            np.savez_compressed(args.output / f"evaluation_{kind}_seed{seed}.npz", **example)
            print(json.dumps({"seed": seed, "condition": kind, **metrics}), flush=True)
        runs.append(record)
        result = {"config": asdict(c), "seeds": args.seeds, "quick": args.quick,
                  "upstream_commit": upstream, "runs": runs,
                  "interpretation": "Finite-sample empirical recovery only; non-OU conditions are outside the Gaussian OU theorem. No Theorem 3 bounds are applied to them."}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        plot(args.output, runs, examples, histories)
        write_report(args.output, result)
    print(f"Complete: {args.output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

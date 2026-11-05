"""Learn OU parameters with an observation-based objective, not distance R².

Two distinct experiments: exact OU simulator pairs, and a physical three-action
controller inspired by OU targets. The latter is explicitly not an exact OU law.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reliability"))
import run_study as study
import policies

behavior, controls = study.behavior, study.controls
import numpy as np
import torch
from torch import nn

CONDITIONS = [
    "fixed_ou",
    "fixed_gap_rho",
    "learn_rho",
    "learn_spread",
    "learn_both",
    "physical_only",
    "physical_to_ou",
]


def logit(value):
    return math.log(value / (1 - value))


class OUActionPolicy(behavior.Policy):
    """Discretize a Gaussian target change in observed log apparent height.

    A target AR(1) log-height change is converted to approach/stay/retreat
    probabilities using fixed thresholds ±0.02. Real physical moves are still
    ±0.25 with reflection. Their realized correlation is not the target rho.
    """

    def __init__(self):
        super().__init__()
        self.ou_parameters = nn.Parameter(
            torch.tensor(
                [
                    logit((0.9 - 0.1) / (0.995 - 0.1)),
                    logit((0.02 - 0.002) / (0.3 - 0.002)),
                    0.0,
                ]
            )
        )

    def values(self):
        values = self.ou_parameters.sigmoid()
        return (
            0.1 + 0.895 * values[0],
            0.002 + 0.298 * values[1],
            -math.log(16) + math.log(16) * values[2],
        )

    def optimizer_groups(self):
        other = [p for name, p in self.named_parameters() if name != "ou_parameters"]
        return [
            {"params": other, "lr": 1e-4},
            {"params": [self.ou_parameters], "lr": 0.01},
        ]

    def forward(self, x):
        _, value = super().forward(x)
        rho, noise, center = self.values()
        size = x[:, 1].clamp_min(1e-8).log()
        change = (1 - rho) * (center - size)
        normal_cdf = lambda t: 0.5 * (1 + torch.erf(t / math.sqrt(2)))
        approach = 1 - normal_cdf((0.02 - change) / noise)
        retreat = normal_cdf((-0.02 - change) / noise)
        stay = (1 - approach - retreat).clamp_min(1e-8)
        probabilities = torch.stack([approach, stay, retreat], -1)
        probabilities = 0.9 * probabilities + 0.1 / 3
        return torch.distributions.Categorical(probs=probabilities), value


def learn_generator(seed, mode, updates=600):
    raw = nn.Parameter(
        torch.tensor([logit((0.9 - 0.1) / 0.88), logit((0.7 - 0.3) / 0.9)])
    )
    optimizer = torch.optim.Adam([raw], lr=0.02)
    generator = torch.Generator().manual_seed(1600000 + seed)
    centers = torch.linspace(-math.log(16), 0, 32)
    history = []
    for step in range(updates):
        rho = (
            0.1 + 0.88 * raw[0].sigmoid()
            if mode != "learn_spread"
            else torch.tensor(0.9)
        )
        spread = (
            0.3 + 0.9 * raw[1].sigmoid() if mode != "learn_rho" else torch.tensor(0.7)
        )
        z = torch.randn(512, generator=generator)
        zp = rho * z + (1 - rho.square()).sqrt() * torch.randn(512, generator=generator)
        # The optimization objective sees only rendered observations.
        size = behavior.observe(z, spread)[:, 1].log()
        next_size = behavior.observe(zp, spread)[:, 1].log()
        first = size - size.mean()
        second = next_size - next_size.mean()
        correlation = (first * second).mean() / (
            first.square().mean() * second.square().mean()
        ).sqrt().clamp_min(1e-8)
        gap = correlation * (1 - correlation)
        probabilities = torch.softmax(
            -((size[:, None] - centers) / 0.15).square(), -1
        ).mean(0)
        coverage_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
        outside = (
            torch.relu(size).square() + torch.relu(-math.log(16) - size).square()
        ).mean()
        objective = gap + 0.1 * coverage_entropy - 0.5 * outside
        optimizer.zero_grad()
        (-objective).backward()
        optimizer.step()
        if step % 100 == 0 or step == updates - 1:
            history.append(
                {
                    "update": step,
                    "rho": float(rho.detach()),
                    "spread": float(spread.detach()),
                    "observed_correlation": float(correlation.detach()),
                    "objective": float(objective.detach()),
                }
            )
    # Record post-update parameters, not the pre-update last minibatch values.
    rho = float(0.1 + 0.88 * raw[0].sigmoid()) if mode != "learn_spread" else 0.9
    spread = float(0.3 + 0.9 * raw[1].sigmoid()) if mode != "learn_rho" else 0.7
    return {"rho": rho, "spread": spread, "history": history}


def fit_pairs(data, c, seed, reference):
    # Ordinary representation fitting: only the two observation tensors enter.
    model, history = behavior.train_encoder(
        {"x": data["x"], "xp": data["xp"]}, c, 40000 + seed
    )
    return model, history


def report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Learned OU parameters",
        "",
        "Five independent seeds 13–17; no distance-labelled tuning. Simulator OU and the physical controller are separate experiments.",
        "",
    ]
    if result["quick"]:
        lines += ["**Execution check only.**", ""]
    lines += [
        "| Condition | Runs | Shared-range $R^2$ | Broader fixed-OU $R^2$ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in CONDITIONS:
        rows = [r for r in result["runs"] if r["condition"] == name]
        if rows:
            lines.append(
                f"| {name} | {len(rows)} | "
                + " | ".join(
                    controls.summary([r["metrics"][k] for r in rows])
                    for k in ["common_grid_r2", "reference_ou_r2"]
                )
                + " |"
            )
    lines += [
        "",
        "| Seed | Generator mode | Learned correlation | Learned spread |",
        "| --- | --- | ---: | ---: |",
    ]
    for seed, values in result["generator_parameters"].items():
        for mode, setting in values.items():
            lines.append(
                f"| {seed} | {mode} | {setting['rho']:.5f} | {setting['spread']:.5f} |"
            )
    lines += [
        "",
        "The generator maximizes an observed-correlation spectral-gap surrogate plus observed-size coverage, with an outside-range penalty. "
        "It has simulator access for reparameterized sampling; it is not a physical action policy. The physical controller learns target correlation, noise and center through count-reward actor–critic and retains the three original actions. "
        "Its realized transition need not be OU. The known-Gaussian gap is maximized at correlation 0.5, so a fixed 0.5 control tests whether parameter learning adds anything beyond that analytic choice.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    c = behavior.Config()
    if args.quick:
        c.jepa_steps, c.policy_updates, c.train_episodes, c.eval_episodes = 40, 2, 8, 4
        c.nenv, c.rollout, c.horizon, c.burnin, c.batch = 4, 4, 8, 4, 16
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(c),
        "quick": args.quick,
        "generator_parameters": {},
        "physical_parameters": {},
        "runs": [],
    }
    if args.resume or args.evaluate_only:
        result = json.loads((args.output / "metrics.json").read_text())
        if result["config"] != vars(c) or result["quick"] != args.quick:
            raise RuntimeError("Configuration differs")
    seeds = [13] if args.quick else list(range(13, 18))
    for seed in seeds:
        reference = (
            behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
            behavior.collect(c, "ou", c.eval_episodes, 30000 + seed),
        )
        if args.evaluate_only:
            for row in [r for r in result["runs"] if r["seed"] == seed]:
                saved = torch.load(args.output / row["checkpoint"], map_location="cpu")
                model = behavior.Encoder()
                model.load_state_dict(saved["encoder"])
                metrics, _ = controls.evaluate(model, reference, c, seed)
                for key in metrics:
                    if not np.isclose(
                        metrics[key], row["metrics"][key], atol=1e-7, rtol=0
                    ):
                        raise RuntimeError("Evaluation changed")
            continue
        if str(seed) not in result["generator_parameters"]:
            result["generator_parameters"][str(seed)] = {
                mode: learn_generator(seed, mode, 5 if args.quick else 600)
                for mode in ["learn_rho", "learn_spread", "learn_both"]
            }
        policy_path = args.output / f"physical_policy_seed{seed}.pt"
        if policy_path.exists():
            saved = torch.load(policy_path, map_location="cpu")
            policy = OUActionPolicy()
            policy.load_state_dict(saved["policy"])
        else:
            policy, saved = policies.train(
                c, seed, "count", policy_factory=OUActionPolicy
            )
            saved["kind"] = "learned_ou_physical"
            torch.save(saved, policy_path)
        rho, noise, center = policy.values()
        result["physical_parameters"][str(seed)] = {
            "rho": float(rho),
            "noise": float(noise),
            "center_log_height": float(center),
        }
        physical = behavior.collect(c, "icm", c.train_episodes, 10000 + seed, policy)
        torch.save(physical, args.output / f"physical_data_seed{seed}.pt")
        for name in CONDITIONS:
            if any(
                r["seed"] == seed and r["condition"] == name for r in result["runs"]
            ):
                continue
            print(f"seed={seed}: {name}", flush=True)
            if name == "physical_to_ou":
                model, history, _ = study.fit(physical, "icm", c, seed, seed, reference)
            elif name == "physical_only":
                model, history = fit_pairs(physical, c, seed, reference)
            else:
                setting = result["generator_parameters"][str(seed)].get(
                    name,
                    {"rho": 0.5 if name == "fixed_gap_rho" else 0.9, "spread": 0.7},
                )
                generator = torch.Generator().manual_seed(10000 + seed)
                n = c.train_episodes * c.horizon
                z = torch.randn(n, generator=generator)
                zp = setting["rho"] * z + math.sqrt(
                    1 - setting["rho"] ** 2
                ) * torch.randn(n, generator=generator)
                data = {
                    "x": behavior.observe(z, setting["spread"]),
                    "xp": behavior.observe(zp, setting["spread"]),
                }
                model, history = fit_pairs(data, c, seed, reference)
            metrics, example = controls.evaluate(model, reference, c, seed)
            filename = f"encoder_{name}_seed{seed}.pt"
            torch.save(
                {"encoder": model.state_dict(), "condition": name, "config": vars(c)},
                args.output / filename,
            )
            np.savez_compressed(
                args.output / filename.replace(".pt", ".npz"), **example
            )
            result["runs"].append(
                {
                    "seed": seed,
                    "condition": name,
                    "metrics": metrics,
                    "checkpoint": filename,
                }
            )
            report(args.output, result)
    print(
        f"{'Reproduced' if args.evaluate_only else 'Completed'} {len(result['runs'])} learned-OU evaluations.",
        flush=True,
    )


if __name__ == "__main__":
    main()

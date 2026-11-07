"""Interior nonlinear drift: finite-time action bias and independent-noise cost.

This is a simulator/statistical diagnostic, not representation learning. No
shared-noise action interventions are used in the Monte Carlo experiment.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def deterministic_effect(distance, interval, substeps=64):
    action = torch.tensor([-1.0, 1.0], dtype=torch.float64)[:, None]
    state = distance[None].repeat(2, 1)
    dt = interval / substeps
    for _ in range(substeps):

        def drift(x):
            return 0.3 * torch.sin(x) + 0.25 * action

        k1 = drift(state)
        k2 = drift(state + 0.5 * dt * k1)
        k3 = drift(state + 0.5 * dt * k2)
        k4 = drift(state + dt * k3)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return (state[1] - state[0]) / (2 * interval)


def noisy_contrasts(distance, interval, groups=256, samples=128, substeps=64, seed=29):
    generator = torch.Generator().manual_seed(3600000 + seed)
    state = distance[None, :, None, None].expand(2, -1, groups, samples).clone()
    action = torch.tensor([-1.0, 1.0], dtype=torch.float64)[:, None, None, None]
    dt = interval / substeps
    for _ in range(substeps):
        noise = torch.randn(state.shape, generator=generator, dtype=torch.float64)
        drift = 0.3 * torch.sin(state) + 0.25 * action
        diffusion = 0.05 * (1 + 0.1 * torch.cos(state))
        state = state + dt * drift + math.sqrt(dt) * diffusion * noise
    means = state.mean(-1)
    return (means[1] - means[0]) / (2 * interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output-drift"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    distance = torch.tensor([2.0, 5.0, 9.0, 13.0], dtype=torch.float64)
    groups, samples, substeps = (16, 16, 8) if args.quick else (256, 128, 64)
    result = {
        "quick": args.quick,
        "groups": groups,
        "samples_per_action": samples,
        "microsteps": substeps,
        "distance": distance.tolist(),
        "runs": [],
    }
    for index, interval in enumerate([0.02, 0.05, 0.1, 0.2]):
        print(f"interval={interval}", flush=True)
        effects = noisy_contrasts(
            distance, interval, groups, samples, substeps, seed=29 + index
        )
        variance = effects.var(-1, unbiased=True)
        leading = (0.05 * (1 + 0.1 * torch.cos(distance))).square() / (
            2 * samples * interval
        )
        effect = deterministic_effect(distance, interval)
        first_order = 0.25 + interval * 0.25 * 0.3 * torch.cos(distance) / 2
        result["runs"].append(
            {
                "interval": interval,
                "contrast_means": effects.mean(-1).tolist(),
                "empirical_variance": variance.tolist(),
                "leading_variance": leading.tolist(),
                "mean_variance_ratio": float((variance / leading).mean()),
                "deterministic_effect": effect.tolist(),
                "leading_effect": first_order.tolist(),
                "max_deterministic_bias": float((effect - 0.25).abs().max()),
                "max_first_order_remainder": float((effect - first_order).abs().max()),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Finite-time action effects and independent-noise sampling",
        "",
        "Four interior source states. Each action has independently sampled noise. This is a simulator diagnostic, not an encoder-training result. [Derivation and limits](../DRIFT_ESTIMATION.md).",
        "",
        f"{groups} independent groups, {samples} endpoints per action/group, {substeps} Euler microsteps per interval.",
        "",
        "| Interval | Empirical / leading variance | Largest deterministic action-effect bias | First-order remainder |",
        "| ---: | ---: | ---: | ---: |",
    ]
    if args.quick:
        lines.insert(2, "**Execution check only.**")
    for row in result["runs"]:
        lines.append(
            f"| {row['interval']:.2f} | {row['mean_variance_ratio']:.3f} | {row['max_deterministic_bias']:.6g} | {row['max_first_order_remainder']:.6g} |"
        )
    lines += [
        "",
        "The leading variance is $\\sigma(d)^2/(2N\\delta)$. Deterministic bias is measured against the instantaneous coefficient 0.25. Its first-order approximation is $0.25+0.25\\delta(0.3\\cos d)/2$. The deterministic calculation does not include stochastic conditional-mean corrections. Full per-state values and settings are retained in metrics.json.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    print(
        f"Completed {len(result['runs'])} independent-noise interval checks.",
        flush=True,
    )


if __name__ == "__main__":
    main()

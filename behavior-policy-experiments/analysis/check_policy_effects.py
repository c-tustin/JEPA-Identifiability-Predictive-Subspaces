"""Check a controlled example: same Gaussian states, different transition rule.

This is a mathematical two-dimensional construction, not an ICM agent run.
It shows that changing transitions alone can change which coordinates alignment
learns, even when both endpoint distributions stay exactly standard Gaussian.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identifiability"))
import reproduce  # Reuse the plotting/cache setup; no training is run.
import matplotlib.pyplot as plt
import numpy as np
import torch


def radial_twist(z, strength):
    """Rotate by strength*radius²; preserves radius and Gaussian measure.

    Its inverse is radial_twist(z, -strength), since rotation preserves radius.
    In polar coordinates it only adds a radius-dependent angle.
    """
    angle = strength * z.square().sum(dim=-1)
    cosine, sine = angle.cos(), angle.sin()
    return torch.stack([cosine * z[..., 0] - sine * z[..., 1],
                        sine * z[..., 0] + cosine * z[..., 1]], dim=-1)


def simulate(samples=250000, rho=.9, strength=.7, seed=123):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(samples, 2, dtype=torch.float64, generator=generator)
    noise = torch.randn(samples, 2, dtype=torch.float64, generator=generator)
    u = radial_twist(z, strength)
    up = rho * u + (1 - rho**2)**.5 * noise
    zp = radial_twist(up, -strength)

    linear_pair_loss = (zp - z).square().sum(-1)
    nonlinear_pair_loss = (radial_twist(zp, strength) - u).square().sum(-1)
    difference = linear_pair_loss - nonlinear_pair_loss
    result = {
        "description": "Analytic Gaussian-preserving nonlinear transition example; not a trained behavior policy.",
        "samples": samples, "rho": rho, "strength": strength, "seed": seed,
        "population_nonlinear_optimum": 4 * (1 - rho),
        "empirical_linear_alignment": float(linear_pair_loss.mean()),
        "empirical_nonlinear_alignment": float(nonlinear_pair_loss.mean()),
        "empirical_loss_advantage": float(difference.mean()),
        "monte_carlo_standard_error_of_advantage": float(difference.std() / samples**.5),
        "initial_mean": z.mean(0).tolist(), "successor_mean": zp.mean(0).tolist(),
        "initial_covariance": torch.cov(z.T).tolist(), "successor_covariance": torch.cov(zp.T).tolist(),
        "roundtrip_max_absolute_error": float((radial_twist(u, -strength) - z).abs().max()),
        "interpretation": "Exact Gaussian marginals follow from the measure-preserving construction, not from these sample moment checks. The nonlinear transition lies outside the original independent additive-OU assumptions.",
    }
    return result, z, u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=250000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("At least two samples are needed")
    torch.set_num_threads(1)
    result, z, u = simulate(samples=args.samples)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy_counterexample.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    count = min(3000, len(z))
    for ax, values, title in [(axes[0], z, "Physical coordinates z"),
                               (axes[1], u, "Nonlinear coordinates F(z)")]:
        ax.scatter(values[:count, 0], values[:count, 1], c=z[:count, 0], cmap="coolwarm", s=3, alpha=.6)
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal", title=title)
    axes[2].bar(["Any Gaussian-compatible\nlinear map Qz", "Nonlinear\nmap F(z)"],
                [result["empirical_linear_alignment"], result["empirical_nonlinear_alignment"]],
                color=["#c2410c", "#0e7490"])
    axes[2].axhline(result["population_nonlinear_optimum"], color="black", linestyle="--", label="Population optimum")
    axes[2].set(title="Alignment loss: lower is better", ylabel="Mean squared pair difference")
    axes[2].legend(fontsize=8)
    fig.suptitle("Same Gaussian state distribution; a different transition rule changes the optimum", fontsize=13)
    fig.savefig(args.output / "policy_counterexample.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

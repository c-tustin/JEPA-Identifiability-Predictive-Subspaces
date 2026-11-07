"""Label-free recovery of log-distance from a triangle's inverse-distance projection.

Run: python3 identifiability/reproduce.py
Requires numpy, torch, matplotlib. Uses CPU; no downloads or external data.
This is a toy adaptation of Theorem 1, not the authors' implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kempner-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# Fixed, frontoparallel triangle, with unit circumradius. Image coordinates are
# continuous and unbounded: no crop, quantization, visibility loss, or noise.
TRIANGLE = torch.tensor([[0., 1.], [-.8660254, -.5], [.8660254, -.5]])


def observe(z, sigma=.7, d0=4.):
    """Only the renderer sees latent z. Returns 3 projected (x,y) vertices."""
    distance = d0 * torch.exp(sigma * z)
    return (TRIANGLE.to(z) / distance[:, None, None]).flatten(1)


def sample_pairs(n, rho, sigma):
    z = torch.randn(n)
    zp = rho * z + (1 - rho ** 2) ** .5 * torch.randn(n)
    return observe(z, sigma), observe(zp, sigma)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 48), nn.Tanh(),
                                 nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x):
        # Bounded sensor coordinates improve conditioning. This supplies neither
        # log-distance nor an inverse-distance formula to the network.
        return self.net(torch.tanh(4 * x)).flatten()


class SIGReg1D(nn.Module):
    """Scalar specialization of the authors' SIGReg (17 knots, t_max=3).

    Reference: https://github.com/klindtlab/lejepa-identifiability/blob/main/experiments/lejepa_id/losses.py
    In 1D every normalized projection is +1 or -1. The squared CF error is
    invariant to this sign, so averaging random slices gives this same loss.
    Matches their factor of two, Gaussian integration weight, and sample scaling.
    """
    def __init__(self, knots=17, t_max=3.):
        super().__init__()
        frequency = torch.linspace(0., t_max, knots)
        target = torch.exp(-frequency.square() / 2)
        quadrature = torch.ones(knots) * (2 * t_max / (knots - 1))
        quadrature[[0, -1]] *= .5
        self.register_buffer("frequency", frequency)
        self.register_buffer("target", target)
        self.register_buffer("weight", quadrature * target)

    def forward(self, embedding):
        phase = embedding.reshape(-1, 1) * self.frequency
        real_error = phase.cos().mean(0) - self.target
        imaginary_error = phase.sin().mean(0)
        return embedding.numel() * ((real_error.square() + imaginary_error.square()) * self.weight).sum()


def fit(steps, batch, rho, sigma, seed, sigreg_weight):
    torch.manual_seed(seed)
    model = Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    regularizer = SIGReg1D()
    history = []
    for step in range(steps):
        # Constant learning rate, then cosine decay as in the paper's schedule.
        fraction = max(0., (step - steps / 2) / max(1., steps / 2 - 1))
        optimizer.param_groups[0]["lr"] = .001 * .5 * (1 + np.cos(np.pi * fraction))
        x, xp = sample_pairs(batch, rho, sigma)
        y = model(torch.cat([x, xp]))
        alignment = (y[:batch] - y[batch:]).square().mean()
        gaussian = regularizer(y)
        loss = (1 - sigreg_weight) * alignment + sigreg_weight * gaussian
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append([step, float(alignment.detach()), float(gaussian.detach())])
    return model.eval(), history


def r2_linear(x, target):
    design = np.column_stack([np.asarray(x), np.ones(len(x))])
    # Fit the evaluation probe on one half and score on the other.
    split = len(target) // 2
    coefficient = np.linalg.lstsq(design[:split], target[:split], rcond=None)[0]
    residual = target[split:] - design[split:] @ coefficient
    return float(1 - np.mean(residual ** 2) / np.var(target[split:]))


@torch.no_grad()
def evaluate(model, rho, sigma, seed):
    # Calibration and test data are independent of optimization. Ground truth
    # enters only evaluation, including choosing the unidentifiable global sign.
    torch.manual_seed(100000 + seed)
    cal = model(observe(torch.randn(40000), sigma))
    mean, std = cal.mean(), cal.std(unbiased=False)
    z = torch.randn(100000)
    zp = rho * z + (1 - rho ** 2) ** .5 * torch.randn(len(z))
    y = (model(observe(z, sigma)) - mean) / std
    yp = (model(observe(zp, sigma)) - mean) / std
    sign = 1. if (y * z).mean() >= 0 else -1.
    y, yp = sign * y, sign * yp
    cov = float(y.var(unbiased=False))
    alignment = float((yp - y).square().mean())
    delta = alignment - 2 * (1 - rho) * cov
    D = max(0., delta) / (2 * rho * (1 - rho))
    eps = abs(cov - 1.)
    z_np, y_np = z.numpy(), y.numpy()
    size = (1 / (4 * torch.exp(sigma * z))).numpy()
    # Flipping signs outside a symmetric interval preserves N(0,1) exactly,
    # while scrambling the latent. This isolates alignment's role.
    scrambled = torch.where(z.abs() <= 1., z, -z)
    scrambled_p = torch.where(zp.abs() <= 1., zp, -zp)
    metrics = {
        "seed": seed,
        "raw_embedding_mean": float(y.mean() * std + sign * mean),
        "raw_embedding_variance": float(cov * std.square()),
        "raw_alignment": float(alignment * std.square()),
        "raw_sign_aligned_mse": float((y * std + sign * mean - z).square().mean()),
        "metric_note": "Other embedding metrics use independent calibration to zero mean/unit variance; raw metrics above do not.",
        "r2_linear_embedding_to_log_distance": r2_linear(y_np, z_np),
        "r2_linear_apparent_size_to_log_distance": r2_linear(size, z_np),
        "r2_linear_embedding_to_physical_distance": r2_linear(y_np, np.exp(sigma * z_np)),
        "sign_aligned_mse": float((y - z).square().mean()),
        "alignment": alignment,
        "theoretical_minimum_alignment": 2 * (1 - rho),
        "gaussian_preserving_scramble_alignment": float((scrambled_p - scrambled).square().mean()),
        "gaussian_preserving_scramble_mse": float((scrambled - z).square().mean()),
        "embedding_mean": float(y.mean()),
        "embedding_variance": cov,
        "transition_residual_variance": float((yp - rho * y).var(unbiased=False)),
        "theoretical_transition_residual_variance": 1 - rho ** 2,
        "estimated_alignment_gap": delta,
        "estimated_theorem3_bound": D + (eps + D) ** 2,
        "bound_note": "Monte Carlo diagnostic, not a certified population bound; mean only approximately zero.",
    }
    return metrics, (z_np, y_np, size), (float(mean), float(std), sign)


def plot_result(out, records, samples, history, sigma):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    ax = axes[0, 0]
    triangle = TRIANGLE.numpy()
    for center, d, color in [(0, 2, "#0e7490"), (1, 4, "#7c3aed"), (2, 8, "#c2410c")]:
        vertices = triangle / d
        ax.fill(vertices[:, 0] + center, vertices[:, 1], color=color, alpha=.8)
        ax.text(center, -.38, f"d = {d}\nsize = 1/{d}", ha="center", va="top")
    ax.set(xlim=(-.6, 2.6), ylim=(-.75, .65), aspect="equal",
           title="Observation: the same triangle scales as 1/d")
    ax.axis("off")
    z, y, size = samples
    keep = np.argsort(z)[::40]
    ax = axes[0, 1]
    ax.scatter(z[keep], size[keep], s=4, alpha=.5, color="#c2410c")
    ax.set(xlabel=f"True latent z = log(d/4) / {sigma}", ylabel="Apparent size",
           title="Sensor mapping is nonlinear", xlim=(-3.5, 3.5), ylim=(0, 3))
    ax = axes[1, 0]
    ax.scatter(z[keep], y[keep], s=4, alpha=.4, color="#0e7490")
    ax.plot([-4, 4], [-4, 4], "k--", lw=1)
    ax.set(xlabel=f"True latent z = log(d/4) / {sigma}", ylabel="Learned embedding (sign aligned)",
           title=f"Recovery from observations only: R² = {records[0]['r2_linear_embedding_to_log_distance']:.4f}",
           xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
    ax = axes[1, 1]
    h = np.array(history)
    ax.plot(h[:, 0], h[:, 1], color="#0e7490", label="Training pair alignment")
    ax.axhline(records[0]["theoretical_minimum_alignment"], color="k", linestyle="--",
               label="Unit-variance reference 2(1-rho)")
    ax.set(xlabel="Optimization step", ylabel="Mean squared pair difference (raw)",
           title="Alignment with SIGReg; no training whitening")
    ax.legend(fontsize=8)
    fig.suptitle("A one-dimensional identifiability experiment with SIGReg", fontsize=16)
    fig.savefig(out / "reproduction.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--rho", type=float, default=.9)
    parser.add_argument("--sigma", type=float, default=.7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sigreg-weight", type=float, default=.1,
                        help="lambda in (1-lambda)*alignment + lambda*SIGReg")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if not 0 < args.rho < 1 or args.sigma <= 0 or args.steps < 1 or args.batch < 2 or not 0 < args.sigreg_weight < 1:
        parser.error("Require 0 < rho, sigreg-weight < 1, sigma > 0, steps >= 1, batch >= 2")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, seed in enumerate(args.seeds):
        model, history = fit(args.steps, args.batch, args.rho, args.sigma, seed, args.sigreg_weight)
        metrics, samples, calibration = evaluate(model, args.rho, args.sigma, seed)
        records.append(metrics)
        torch.save({"state_dict": model.state_dict(), "calibration": calibration,
                    "sigma": args.sigma, "rho": args.rho, "history": history,
                    "sigreg_weight": args.sigreg_weight}, args.output / f"encoder_seed{seed}.pt")
        if i == 0:
            plot_result(args.output, records, samples, history, args.sigma)
        print(json.dumps(metrics, indent=2), flush=True)
    result = {"config": {**vars(args), "output": str(args.output)}, "runs": records,
              "method": "MLP with raw pair alignment + SIGReg1D; no training whitening",
              "sigreg_reference": "https://github.com/klindtlab/lejepa-identifiability/blob/main/experiments/lejepa_id/losses.py"}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

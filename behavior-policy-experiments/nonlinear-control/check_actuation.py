"""One-distance nonlinear drift: test the proved action-anchor mechanism.

Geometry-aware smooth monotone encoder, paired +/- action interventions, common noise.
No true distance targets enter training. This is not the passive SIGReg model.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reliability"))
from policies import behavior
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AnchoredEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "knots",
            torch.exp(
                torch.linspace(
                    math.log(1 / 18), math.log(1 / 0.8), 129, dtype=torch.float64
                )
            ),
        )
        self.increments = nn.Parameter(torch.full((129,), -3.0, dtype=torch.float64))

    def forward(self, x):
        # Positive combinations of smooth convex hinges preserve ordering and
        # are C-infinity, unlike a piecewise-linear spline at its knots.
        height = x[:, 1, None]
        bandwidth = 0.025 * self.knots
        hinges = F.softplus((self.knots - height) / bandwidth) * bandwidth
        return (hinges * F.softplus(self.increments)).sum(-1)


def views(distance):
    return behavior.sensor(distance, behavior.Config())


def intervention_views(distance, noise, dt=0.02, c=0.25):
    # Nonlinear drift and state-dependent noise, same scalar physical state.
    drift = 0.3 * torch.sin(distance)
    diffusion = 0.05 * (1 + 0.1 * torch.cos(distance))
    middle = distance + drift * dt + diffusion * math.sqrt(dt) * noise
    return views(middle - c * dt), views(middle + c * dt)


def analytic_bound():
    low, high = 1.2, 15.8
    length = high - low
    amplitude = 0.05
    c = 0.25
    v0 = 2 / 3
    average_cos = (math.sin(high) - math.sin(low)) / length
    variance_cos = (
        0.5 + (math.sin(2 * high) - math.sin(2 * low)) / (4 * length) - average_cos**2
    )
    risk = v0 * c * c * amplitude**2 * variance_cos
    d = np.linspace(low, high, 100000)
    residual = amplitude * (np.sin(d) - math.sin(low) - average_cos * (d - low))
    bound = length * length * risk / (c * c * v0)
    return {
        "conditional_drift_risk": risk,
        "actual_sup_affine_error_squared": float(np.max(residual**2)),
        "proved_sup_error_squared_bound": bound,
        "bound_holds_on_grid": bool(np.max(residual**2) <= bound),
    }


def run(out, quick=False):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "quick": quick,
        "assumptions": "Paired interventions, common noise, invertible ordered encoder, interior distances; no distance-labelled target.",
        "analytic_bound_check": analytic_bound(),
        "runs": [],
    }
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for seed in [0] if quick else [0, 1, 2]:
        torch.manual_seed(seed)
        model = AnchoredEncoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        generator = torch.Generator().manual_seed(1700000 + seed)
        for step in range(30 if quick else 4000):
            distance = 1.2 + 14.6 * torch.rand(
                512, dtype=torch.float64, generator=generator
            )
            noise = torch.randn(512, dtype=torch.float64, generator=generator)
            minus, plus = intervention_views(distance, noise)
            # Constant calibrated action effect; no true D or log(D) target.
            estimated_effect = (model(plus) - model(minus)) / (2 * 0.25 * 0.02)
            loss = (estimated_effect - 1).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            grid = torch.linspace(1.2, 15.8, 1024, dtype=torch.float64)
            encoded = model(views(grid)).numpy()
            # Independent calibration probes used only after all training.
            cal = 1.2 + 14.6 * torch.rand(
                4096, dtype=torch.float64, generator=generator
            )
            ycal = model(views(cal)).numpy()
            score = behavior.affine_score(ycal, cal.numpy(), encoded, grid.numpy())
            probe = np.linalg.lstsq(
                np.column_stack([ycal, np.ones(len(ycal))]), cal.numpy(), rcond=None
            )[0]
            prediction = np.column_stack([encoded, np.ones(len(encoded))]) @ probe
            row = {
                "seed": seed,
                "physical_distance_r2": score,
                "final_action_effect_loss": float(loss),
            }
            result["runs"].append(row)
            torch.save(
                {"encoder": model.state_dict(), "seed": seed},
                out / f"encoder_seed{seed}.pt",
            )
            np.savez_compressed(
                out / f"evaluation_seed{seed}.npz",
                distance=grid.numpy(),
                prediction=prediction,
            )
            ax.plot(grid, prediction, label=f"Seed {seed}")
    ax.plot([1.2, 15.8], [1.2, 15.8], "k--", label="Exact distance")
    ax.set(
        xlabel="True distance (evaluation only)",
        ylabel="Affine readout of learned coordinate",
        title="Nonlinear drift with calibrated action interventions",
    )
    ax.legend()
    fig.savefig(out / "actuation.png", dpi=160)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Nonlinear action-anchor check",
        "",
        (
            "**Execution check only.**"
            if quick
            else "Three fixed seeds; one physical distance; no distance-labelled training targets."
        ),
        "",
        "| Seed | Physical-distance $R^2$ | Action-effect loss |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {r['seed']} | {r['physical_distance_r2']:.6f} | {r['final_action_effect_loss']:.6g} |"
        for r in result["runs"]
    ]
    lines += [
        "",
        "The analytic perturbation satisfies the proved bound in the independent check. The encoder uses known apparent-size ordering and paired interventions with common simulator noise. These are stronger assumptions than the passive experiment.",
        "",
        "![Action anchor](actuation.png)",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def verify_saved(out):
    result = json.loads((out / "metrics.json").read_text())
    for record in result["runs"]:
        seed = record["seed"]
        generator = torch.Generator().manual_seed(1700000 + seed)
        # Recreate only the independent evaluation RNG position, not training.
        for _ in range(30 if result["quick"] else 4000):
            torch.rand(512, dtype=torch.float64, generator=generator)
            torch.randn(512, dtype=torch.float64, generator=generator)
        cal = 1.2 + 14.6 * torch.rand(4096, dtype=torch.float64, generator=generator)
        grid = torch.linspace(1.2, 15.8, 1024, dtype=torch.float64)
        model = AnchoredEncoder()
        model.load_state_dict(
            torch.load(out / f"encoder_seed{seed}.pt", map_location="cpu")["encoder"]
        )
        with torch.no_grad():
            score = behavior.affine_score(
                model(views(cal)).numpy(),
                cal.numpy(),
                model(views(grid)).numpy(),
                grid.numpy(),
            )
        if not np.isclose(score, record["physical_distance_r2"], rtol=0, atol=1e-10):
            raise RuntimeError("Action-anchor evaluation changed")
    print(f"Reproduced {len(result['runs'])} action-anchor evaluations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    verify_saved(args.output) if args.evaluate_only else run(args.output, args.quick)

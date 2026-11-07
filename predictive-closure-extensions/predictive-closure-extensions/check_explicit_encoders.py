"""Prospective same-world explicit encoder check, including finite resolution."""

import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import ndtr, ndtri
from closure_controls import (
    R,
    pairs,
    signed_logdensity,
    lognormal,
    observe,
    calibrated_coordinate,
)

ROOT = Path(__file__).parent
EPSILONS = [1e-4, 1e-6, 1e-8]
STEPS = [None, 0.5, 0.25, 0.125]


def encode(z, kind, epsilon, step):
    probability = ndtr(z) if kind == "full" else 2 * ndtr(abs(z)) - 1
    clipped = (probability < epsilon) | (probability > 1 - epsilon)
    h = ndtri(np.clip(probability, epsilon, 1 - epsilon))
    if step is not None:
        h = np.round(h / step) * step
    return h, float(clipped.mean())


def design(h):
    knots = np.r_[np.full(4, -6.0), np.linspace(-3, 3, 19), np.full(4, 6.0)]
    return BSpline.design_matrix(h, knots, 3).toarray()


def quantized_joint(kind, epsilon, step, nodes):
    k = ndtri(1 - epsilon)
    levels = np.arange(round(-k / step), round(k / step) + 1) * step
    thresholds = (levels[:-1] + levels[1:]) / 2
    if kind == "full":
        edges = np.r_[-np.inf, thresholds, np.inf]
        marginal = np.diff(ndtr(edges))
        finite = np.clip(edges, -10, 10)
    else:
        edges = np.r_[0.0, ndtri((ndtr(thresholds) + 1) / 2), np.inf]
        marginal = np.diff(2 * ndtr(edges) - 1)
        finite = np.clip(edges, 0, 10)
    points, weights = np.polynomial.legendre.leggauss(nodes)
    half = np.diff(finite) / 2
    source = (finite[1:] + finite[:-1])[:, None] / 2 + half[:, None] * points
    integration = half[:, None] * weights * np.exp(lognormal(source, 0, 1))
    if kind == "magnitude":
        integration *= 2
    sd = np.sqrt(1 - R * R)
    if kind == "full":
        cdf = 0.5 * (
            ndtr((edges[None, None, :] - R * source[..., None]) / sd)
            + ndtr((edges[None, None, :] + R * source[..., None]) / sd)
        )
    else:
        cdf = ndtr((edges[None, None, :] - R * source[..., None]) / sd) - ndtr(
            (-edges[None, None, :] - R * source[..., None]) / sd
        )
    joint = np.einsum("ik,ikj->ij", integration, np.diff(cdf, axis=-1))
    joint = np.maximum(joint, 0)
    product = marginal[:, None] * marginal[None, :]
    positive = joint > 0
    information = np.sum(joint[positive] * np.log(joint[positive] / product[positive]))
    return dict(
        mi=float(information),
        mass=float(joint.sum()),
        levels=len(levels),
        marginal_max_error=float(
            max(
                np.max(abs(joint.sum(0) - marginal)),
                np.max(abs(joint.sum(1) - marginal)),
            )
        ),
    )


def main():
    result = dict(
        settings=dict(
            seeds=list(range(401, 406)),
            train=32768,
            calibration=16384,
            test=65536,
            epsilons=EPSILONS,
            rounding_steps=STEPS,
            ridge=1e-6,
            spline_interior_knots=19,
        ),
        fits=[],
        quantized=[],
    )
    for epsilon in EPSILONS:
        for step in STEPS[1:]:
            for kind in ["full", "magnitude"]:
                low = quantized_joint(kind, epsilon, step, 64)
                high = quantized_joint(kind, epsilon, step, 128)
                difference = abs(high["mi"] - low["mi"])
                assert abs(high["mass"] - 1) < 1e-10
                assert high["marginal_max_error"] < 1e-9
                assert difference < 1e-8
                result["quantized"].append(
                    dict(
                        kind=kind,
                        epsilon=epsilon,
                        step=step,
                        quadrature_difference=difference,
                        **high
                    )
                )
    for seed in range(401, 406):
        splits = [
            pairs(n, seed + offset)
            for n, offset in [(32768, 0), (16384, 10000), (65536, 20000)]
        ]
        # Exact observation inversion is an explicit calibration privilege.
        splits = [
            (calibrated_coordinate(observe(z)), calibrated_coordinate(observe(y)), s)
            for z, y, s in splits
        ]
        for epsilon in EPSILONS:
            for step in STEPS:
                for kind in ["full", "magnitude"]:
                    transformed = [
                        (encode(z, kind, epsilon, step), encode(y, kind, epsilon, step))
                        for z, y, _ in splits
                    ]
                    (h, clip_source), (target, clip_target) = transformed[0]
                    x = design(h)
                    coef = np.linalg.solve(
                        x.T @ x + 1e-6 * np.eye(x.shape[1]), x.T @ target
                    )
                    row = dict(
                        seed=seed,
                        kind=kind,
                        epsilon=epsilon,
                        step=step,
                        parameters=len(coef),
                        coefficients=coef.tolist(),
                    )
                    for name, ((h, cs), (target, ct)) in zip(
                        ["train", "calibration", "test"], transformed
                    ):
                        mse = np.mean((target - design(h) @ coef) ** 2)
                        variance = np.var(target)
                        row.update(
                            {
                                name + "_mse": float(mse),
                                name + "_variance": float(variance),
                                name + "_normalized_mse": float(mse / variance),
                                name + "_source_clipping_fraction": cs,
                                name + "_target_clipping_fraction": ct,
                            }
                        )
                    result["fits"].append(row)
        z, y, _ = splits[-1]
        full_ratio = signed_logdensity(z, y) - lognormal(y, 0, 1)
        # Folded conditional/marginal densities both double, cancelling exactly.
        folded_ratio = (
            signed_logdensity(abs(z), abs(y))
            + np.log(2)
            - (lognormal(abs(y), 0, 1) + np.log(2))
        )
        result.setdefault("continuous_identity", []).append(
            dict(
                seed=seed,
                maximum_logratio_error=float(np.max(abs(full_ratio - folded_ratio))),
                mi=float(full_ratio.mean()),
                irreducible_distance_mse_from_lost_sign=float(
                    np.mean((4 * np.sinh(0.7 * abs(z))) ** 2)
                ),
            )
        )
        print("explicit encoders seed", seed, flush=True)
    z = np.linspace(-5, 5, 1001)
    h1, _ = encode(z, "magnitude", 1e-8, None)
    h2, _ = encode(-z, "magnitude", 1e-8, None)
    assert np.array_equal(h1, h2)
    d = 4 * np.exp(0.7 * z)
    assert np.max(abs(d * (4 * np.exp(-0.7 * z)) - 16)) < 1e-12
    result["ambiguity"] = "magnitude maps distance d and 16/d to the same value"
    (ROOT / "output" / "explicit-encoders.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

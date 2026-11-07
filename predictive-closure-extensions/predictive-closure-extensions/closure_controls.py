"""CPU reference controls; no upstream implementation copied."""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

TRIANGLE = np.array([[0.0, 1.0], [-0.8660254, -0.5], [0.8660254, -0.5]])
R = 0.95


def observe(z):
    return (TRIANGLE / (4 * np.exp(0.7 * np.asarray(z)))[..., None, None]).reshape(
        -1, 6
    )


def calibrated_coordinate(x):
    d = np.linalg.norm(TRIANGLE) / np.linalg.norm(x, axis=-1)
    return np.log(d / 4) / 0.7


def pairs(n, seed, signed=True):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    s = rng.choice([-1.0, 1.0], n)
    r = R if signed else 0.5
    y = (s if signed else 1.0) * r * z + np.sqrt(1 - r * r) * rng.normal(size=n)
    return z, y, s


def lognormal(y, mean, variance):
    return -0.5 * (np.log(2 * np.pi * variance) + (y - mean) ** 2 / variance)


def signed_logdensity(z, y, r=R):
    return np.logaddexp(
        lognormal(y, r * z, 1 - r * r), lognormal(y, -r * z, 1 - r * r)
    ) - np.log(2)


def prediction(t, z, s, kind):
    a, lv = t
    mean = (
        a * z
        if kind == "mean"
        else a * s * z if kind == "conditioned" else np.zeros_like(z)
    )
    return mean, np.exp(lv) + (
        a * a * z * z if kind in ["mixture", "heteroscedastic"] else 0
    )


def logdensity(t, z, y, s, kind):
    if kind == "mixture":
        a, lv = t
        return np.logaddexp(
            lognormal(y, a * z, np.exp(lv)), lognormal(y, -a * z, np.exp(lv))
        ) - np.log(2)
    mean, v = prediction(t, z, s, kind)
    return lognormal(y, mean, v)


def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    result = {
        "settings": dict(
            seeds=list(range(101, 106)),
            moment_pairs=200000,
            train_pairs=16384,
            test_pairs=65536,
            maxiter=150,
            r=R,
        ),
        "analytic": dict(
            signed_mi_lower_bound=-0.5 * np.log(1 - R * R) - np.log(2),
            ar_mi=-0.5 * np.log(0.75),
            signed_mse=1.0,
            ar_mse=0.75,
        ),
        "moments": [],
        "predictors": [],
    }
    for seed in range(101, 106):
        for signed in [False, True]:
            z, y, s = pairs(200000, seed + 10000, signed)
            lp = signed_logdensity(z, y) if signed else lognormal(y, 0.5 * z, 0.75)
            info = lp - lognormal(y, 0, 1)
            residual = y if signed else y - 0.5 * z
            result["moments"].append(
                dict(
                    seed=seed,
                    law="signed" if signed else "ar",
                    mean=float(y.mean()),
                    variance=float(y.var()),
                    fourth_moment=float(np.mean(y**4)),
                    mse=float(np.mean(residual**2)),
                    mi=float(info.mean()),
                    mi_mc_se=float(info.std(ddof=1) / np.sqrt(len(y))),
                    magnitude_correlation=float(np.corrcoef(z * z, y * y)[0, 1]),
                    correlation=float(np.corrcoef(z, y)[0, 1]),
                )
            )
        z, y, s = pairs(16384, seed + 20000)
        train = calibrated_coordinate(observe(z)), calibrated_coordinate(observe(y)), s
        test = pairs(65536, seed + 30000)
        for kind in ["mean", "heteroscedastic", "mixture", "conditioned"]:
            fit = minimize(
                lambda t: -logdensity(t, *train, kind).mean(),
                [0.5, -0.5],
                method="L-BFGS-B",
                bounds=[(-2, 2), (-8, 3)],
                options={"maxiter": 150},
            )
            z, y, s = test
            mean, variance = prediction(fit.x, z, s, kind)
            result["predictors"].append(
                dict(
                    seed=seed,
                    kind=kind,
                    parameters=fit.x.tolist(),
                    converged=bool(fit.success),
                    iterations=int(fit.nit),
                    message=str(fit.message),
                    nll=float(-logdensity(fit.x, z, y, s, kind).mean()),
                    mse=float(np.mean((y - mean) ** 2)),
                    second_moment_mse=float(
                        np.mean((y * y - (variance + mean * mean)) ** 2)
                    ),
                    conditional_second_moment_error=float(
                        np.mean(
                            (1 - R * R + R * R * z * z - (variance + mean * mean)) ** 2
                        )
                    ),
                )
            )
    (out / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
    for kind in ["mean", "heteroscedastic", "mixture", "conditioned"]:
        rows = [x for x in result["predictors"] if x["kind"] == kind]
        print(
            kind,
            "NLL",
            np.mean([x["nll"] for x in rows]),
            "MSE",
            np.mean([x["mse"] for x in rows]),
        )


if __name__ == "__main__":
    main()

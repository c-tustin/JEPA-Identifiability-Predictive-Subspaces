"""Matched-budget scalar triangle JEPA pilot; CPU only."""

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.stats import kstest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "identifiability"))
from reproduce import Encoder
from sigreg import SIGReg1D
from closure_controls import pairs, observe

KINDS = ["mean", "gaussian", "mixture", "conditioned"]


def network():
    return nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 3)
    )


def parameters(raw, kind):
    if kind == "mean":
        return raw.sum(-1) / math.sqrt(3), None, None
    scale = torch.nn.functional.softplus(raw[:, 2]) + 0.03
    mean = raw[:, :2].sum(-1) / math.sqrt(2)
    return mean, scale, raw[:, :2]


def losses(raw, target, kind):
    mean, scale, components = parameters(raw, kind)
    if kind == "mean":
        return (mean - target).square()
    if kind == "mixture":
        logp = (
            -0.5 * ((target[:, None] - components) / scale[:, None]).square()
            - scale[:, None].log()
            - 0.5 * math.log(2 * math.pi)
        )
        return -torch.logsumexp(logp, dim=-1) + math.log(2)
    return (
        0.5 * ((target - mean) / scale).square()
        + scale.log()
        + 0.5 * math.log(2 * math.pi)
    )


def batch(n, seed):
    z, y, s = pairs(n, seed)
    return (
        torch.tensor(observe(z), dtype=torch.float32),
        torch.tensor(observe(y), dtype=torch.float32),
        torch.tensor(s, dtype=torch.float32),
        z,
    )


def inputs(y, s, kind):
    return torch.stack([y, s if kind == "conditioned" else torch.zeros_like(s)], -1)


def probe_metrics(encoder, seed):
    x, _, _, z = batch(24576, seed)
    with torch.no_grad():
        h = encoder(x).numpy()
    mu, sd = h[:8192].mean(), h[:8192].std()
    h = (h - mu) / sd
    design = np.polynomial.polynomial.polyvander(h, 7)
    results = {}
    for key, target in [
        ("distance", 4 * np.exp(0.7 * z)),
        ("log_distance", z),
        ("magnitude", abs(z)),
        ("sign", (z > 0).astype(float)),
    ]:
        coef = np.linalg.lstsq(design[:8192], target[:8192], rcond=None)[0]
        estimate = design[8192:] @ coef
        truth = target[8192:]
        results[key + "_r2"] = float(
            1 - np.mean((estimate - truth) ** 2) / np.var(truth)
        )
        if key == "sign":
            results["sign_accuracy"] = float(np.mean((estimate > 0.5) == (truth > 0.5)))
    with torch.no_grad():
        grid = np.linspace(1, 16, 2049)
        values = (
            encoder(
                torch.tensor(observe(np.log(grid / 4) / 0.7), dtype=torch.float32)
            ).numpy()
            - mu
        ) / sd
    diff = np.diff(values)
    orientation = 1 if values[-1] >= values[0] else -1
    results.update(
        latent_mean=float(h[8192:].mean()),
        latent_variance=float(h[8192:].var()),
        gaussian_ks=float(kstest(h[8192:], "norm").statistic),
        reversal_fraction=float(np.mean(orientation * diff < -1e-6)),
        minimum_grid_sensitivity=float(np.min(abs(diff) / np.diff(grid))),
        sensitivity_p05=float(np.quantile(abs(diff) / np.diff(grid), 0.05)),
    )
    return results


def evaluate(encoder, predictor, kind, seed):
    cx, cy, cs, _ = batch(8192, seed)
    x, y, s, _ = batch(16384, seed + 1)
    with torch.no_grad():
        cal = encoder(torch.cat([cx, cy]))
        mu, sd = cal.mean(), cal.std()
        source, target = encoder(x), encoder(y)
        raw = predictor(inputs(source, s, kind))
        mean, scale, components = parameters(raw, kind)
        if kind == "mixture":
            mean = components.mean(-1)
        mse = ((target - mean) / sd).square().mean()
        if kind == "mean":
            cmean = parameters(predictor(inputs(encoder(cx), cs, kind)), kind)[0]
            variance = (encoder(cy) - cmean).square().mean().clamp_min(1e-8)
            nll = (
                0.5 * (target - mean).square() / variance
                + 0.5 * torch.log(2 * math.pi * variance)
            ).mean()
        else:
            nll = losses(raw, target, kind).mean()
        return dict(
            normalized_mse=float(mse),
            normalized_nll=float(nll - sd.log()),
            normalization=[float(mu), float(sd)],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 106)))
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(exist_ok=True, parents=True)
    settings = dict(
        steps=args.steps,
        batch=256,
        lr=0.001,
        sigreg_weight=0.1,
        seeds=args.seeds,
        torch=torch.__version__,
        numpy=np.__version__,
    )
    (args.output / "learned-settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    rows = []
    for seed in args.seeds:
        for kind in KINDS:
            torch.manual_seed(seed)
            encoder, predictor = Encoder(), network()
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(predictor.parameters()), lr=0.001
            )
            regularizer = SIGReg1D()
            history = []
            for step in range(args.steps):
                x, y, s, _ = batch(256, seed * 100000 + step)
                h = encoder(torch.cat([x, y]))
                source, target = h.chunk(2)
                raw = predictor(inputs(source, s, kind))
                prediction = losses(raw, target, kind).mean()
                reg = regularizer(h)
                loss = prediction + 0.1 * reg
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite objective")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 200 == 0 or step == args.steps - 1:
                    history.append(
                        dict(
                            step=step,
                            prediction=float(prediction.detach()),
                            sigreg=float(reg.detach()),
                        )
                    )
            metrics = evaluate(encoder, predictor, kind, seed + 50000000)
            metrics.update(probe_metrics(encoder, seed + 60000000))
            row = dict(
                seed=seed,
                kind=kind,
                parameters=sum(
                    p.numel()
                    for p in list(encoder.parameters()) + list(predictor.parameters())
                ),
                history=history,
                **metrics,
            )
            rows.append(row)
            torch.save(
                dict(
                    encoder=encoder.state_dict(),
                    predictor=predictor.state_dict(),
                    settings=settings,
                    kind=kind,
                    seed=seed,
                ),
                args.output / f"{kind}-{seed}.pt",
            )
            (args.output / "learned.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                seed,
                kind,
                "log-distance R2",
                round(metrics["log_distance_r2"], 4),
                "magnitude R2",
                round(metrics["magnitude_r2"], 4),
                "sign",
                round(metrics["sign_accuracy"], 4),
                flush=True,
            )


if __name__ == "__main__":
    main()

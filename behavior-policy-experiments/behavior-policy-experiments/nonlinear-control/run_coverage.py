"""Matched-source/action reset queries isolate the drift intervention.

These simulator queries are a coverage control, not a behavior policy. Only
rendered observation pairs enter the unchanged flexible-encoder training loop.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import run_trajectories as study
import numpy as np
import torch


def reset_queries(settings, seed, nonlinear, count=None):
    count = count or settings.episodes * settings.horizon
    source_rng = torch.Generator().manual_seed(3100000 + seed)
    noise_rng = torch.Generator().manual_seed(3200000 + seed)
    distance = 1 + 15 * torch.rand(count, generator=source_rng)
    action = torch.randint(3, (count,), generator=source_rng)
    x = study.behavior.sensor(distance, study.behavior.Config())
    for _ in range(settings.substeps):
        noise = torch.randn(count, generator=noise_rng)
        distance = study.advance(distance, action - 1, noise, nonlinear, settings.dt)
    return {
        "x": x,
        "xp": study.behavior.sensor(distance, study.behavior.Config()),
        "action": action,
    }


def fingerprint(data):
    digest = hashlib.sha256()
    for key in ["x", "xp", "action"]:
        digest.update(data[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Matched source-state and action coverage",
        "",
        "Simulator-reset controls, not trajectory policies. Same source observations/actions/noise across worlds; only drift differs. Paired seeds 23–25, unchanged encoder and fitting budget. [Protocol](../EXTENSION_PROTOCOL.md).",
        "",
        "| Drift | SIGReg | Runs | Distance $R^2$ | Log-distance $R^2$ | Worst reversal (% range) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if result["quick"]:
        lines.insert(2, "**Execution check only.**")
    for nonlinear in [False, True]:
        for sampler in ["raw", "balanced"]:
            rows = [
                r
                for r in result["runs"]
                if r["nonlinear"] == nonlinear and r["sampler"] == sampler
            ]
            if rows:
                means = [
                    np.mean([r["metrics"][key] for r in rows])
                    for key in ["distance_r2", "log_distance_r2"]
                ]
                worst = (
                    max(r["metrics"]["largest_reversal_fraction"] for r in rows) * 100
                )
                lines.append(
                    f"| {'Nonlinear' if nonlinear else 'Zero'} | {sampler} | {len(rows)} | {means[0]:.4f} | {means[1]:.4f} | {worst:.4f} |"
                )
    lines += [
        "",
        "The source distribution is matched; the endpoint mixture seen by SIGReg can still differ because successors depend on dynamics. Reset queries change both state and action coverage relative to trajectories. Complete per-seed metrics, query hashes and checkpoints are retained.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output-coverage"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = study.Settings()
    if args.quick:
        settings = study.Settings(
            episodes=8, horizon=8, burnin=2, substeps=2, updates=20, batch=16
        )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"settings": asdict(settings), "quick": args.quick, "runs": []}
    if args.resume or args.evaluate_only:
        result = json.loads((args.output / "metrics.json").read_text())
        if result["settings"] != asdict(settings) or result["quick"] != args.quick:
            raise ValueError("Saved settings differ")
    elif (args.output / "metrics.json").exists():
        raise ValueError("Use --resume or a new output directory")
    for seed in [23] if args.quick else [23, 24, 25]:
        for nonlinear in [False, True]:
            data = reset_queries(settings, seed, nonlinear)
            source_hash = fingerprint(data)
            for sampler in ["raw", "balanced"]:
                filename = f"encoder_{'nonlinear' if nonlinear else 'zero'}_{sampler}_seed{seed}.pt"
                found = next(
                    (r for r in result["runs"] if r["checkpoint"] == filename), None
                )
                if args.evaluate_only:
                    if found is None or found["query_sha256"] != source_hash:
                        raise RuntimeError("Missing model or changed source queries")
                    model = study.behavior.Encoder()
                    model.load_state_dict(
                        torch.load(args.output / filename, map_location="cpu")[
                            "encoder"
                        ]
                    )
                    metrics, _ = study.evaluate(model, seed)
                    for key, value in metrics.items():
                        if not np.isclose(
                            value, found["metrics"][key], atol=1e-8, rtol=0
                        ):
                            raise RuntimeError(f"Changed evaluation: {filename}: {key}")
                elif found is None:
                    print(filename, flush=True)
                    model = study.fit(data, settings, seed, sampler)
                    metrics, arrays = study.evaluate(model, seed)
                    torch.save(
                        {
                            "encoder": model.state_dict(),
                            "settings": asdict(settings),
                            "seed": seed,
                            "nonlinear": nonlinear,
                            "sampler": sampler,
                            "query_sha256": source_hash,
                        },
                        args.output / filename,
                    )
                    np.savez_compressed(
                        args.output / filename.replace(".pt", ".npz"), **arrays
                    )
                    result["runs"].append(
                        {
                            "seed": seed,
                            "nonlinear": nonlinear,
                            "sampler": sampler,
                            "checkpoint": filename,
                            "query_sha256": source_hash,
                            "coverage": study.coverage(data),
                            "metrics": metrics,
                        }
                    )
                    report(args.output, result)
    print(
        f"{'Reproduced' if args.evaluate_only else 'Completed'} {len(result['runs'])} coverage controls.",
        flush=True,
    )


if __name__ == "__main__":
    main()

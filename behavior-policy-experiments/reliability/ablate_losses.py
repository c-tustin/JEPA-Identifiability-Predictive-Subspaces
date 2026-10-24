"""Post-repair mechanistic test: cross alignment and SIGReg sampling measures.

New seeds 18–22 plus old diagnostic seeds 3 and 7. Two independent batches per
update are used in every cell, so original shared-batch scores are not spliced in.
"""

import argparse
import copy
import json
from pathlib import Path

import run_study as study
from policies import behavior, size_bin
import numpy as np
import torch

CELLS = [
    ("raw", "raw"),
    ("raw", "balanced"),
    ("balanced", "raw"),
    ("balanced", "balanced"),
]


def balanced_views(data, seed):
    bins = size_bin(data["x"])
    counts = torch.bincount(bins, minlength=32).clamp_min(1)
    generator = torch.Generator().manual_seed(900000 + seed)
    selected = torch.multinomial(
        1 / counts[bins].float(), len(bins), replacement=True, generator=generator
    )
    return {k: data[k][selected] for k in ["x", "xp"]}


def fit(data, c, seed, alignment_source, regularizer_source, reference):
    model, plan = study.make_plan(c, seed, seed, "icm")
    ou = study.controls.ou_data(c, seed)
    raw = {k: data[k] for k in ["x", "xp"]}
    balanced = balanced_views(data, seed)
    sources = {"raw": raw, "balanced": balanced}
    second = copy.copy(plan)
    generator = torch.Generator().manual_seed(1900000 + seed)
    second["indices"] = torch.randint(
        len(raw["x"]), plan["indices"].shape, generator=generator
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regularizer = behavior.SIGReg1D()
    for step in range(c.jepa_steps):
        optimizer.param_groups[0]["lr"] = study.controls.learning_rate(
            step, c.jepa_steps
        )
        x, xp = study.controls.batch_at(sources[alignment_source], ou, plan, step)
        y = model(torch.cat([x, xp]))
        alignment = (y[: len(x)] - y[len(x) :]).square().mean()
        rx, rxp = study.controls.batch_at(sources[regularizer_source], ou, second, step)
        penalty = regularizer(model(torch.cat([rx, rxp])))
        loss = (1 - c.sigreg_weight) * alignment + c.sigreg_weight * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Which loss term is sensitive to replay?",
        "",
        "Exploratory mechanism test after the repair study; all four cells fixed before inspecting new seeds 18–22. "
        "Seeds 3 and 7 are historical diagnostics. Every update uses separate 512-pair alignment and regularizer batches in all four conditions. "
        "This doubles observation evaluations relative to the shared-batch experiment, equally in every cell.",
        "",
        "| Seed group | Alignment data | SIGReg data | Runs | Shared-range $R^2$ | Broader OU $R^2$ | Reversed grid steps |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in ["new", "diagnostic"]:
        for alignment, reg in CELLS:
            rows = [
                r
                for r in result["runs"]
                if r["group"] == group
                and r["alignment"] == alignment
                and r["regularizer"] == reg
            ]
            if not rows:
                continue
            values = [
                [r["metrics"][key] for r in rows]
                for key in [
                    "common_grid_r2",
                    "reference_ou_r2",
                    "reversed_grid_fraction",
                ]
            ]
            lines.append(
                f"| {group} | {alignment} | {reg} | {len(rows)} | "
                + " | ".join(study.controls.summary(v) for v in values)
                + " |"
            )
    lines += [
        "",
        "Both losses finish with the same OU sources. Changing a sampling measure changes the finite objective and its optimization path; "
        "this does not prove a particular local minimum or establish Gaussianity. Complete per-seed metrics and checkpoints are retained.",
        "",
    ]
    if result["quick"]:
        lines.insert(2, "**Execution check only.**")
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output-loss-ablation"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    c = behavior.Config()
    if args.quick:
        c.jepa_steps, c.policy_updates, c.train_episodes, c.eval_episodes = 40, 2, 8, 4
        c.nenv, c.rollout, c.horizon, c.burnin, c.batch = 4, 4, 8, 4, 16
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"quick": args.quick, "config": vars(c), "runs": []}
    if args.resume or args.evaluate_only:
        result = json.loads((args.output / "metrics.json").read_text())
        if result["config"] != vars(c) or result["quick"] != args.quick:
            raise RuntimeError("Configuration differs")
    seeds = [3] if args.quick else [3, 7, 18, 19, 20, 21, 22]
    for seed in seeds:
        reference = (
            behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
            behavior.collect(c, "ou", c.eval_episodes, 30000 + seed),
        )
        data, _ = study.prepare_data(c, seed, "icm", args.output, args.quick)
        for alignment, reg in CELLS:
            name = f"align_{alignment}_reg_{reg}"
            found = next(
                (
                    r
                    for r in result["runs"]
                    if r["seed"] == seed
                    and r["alignment"] == alignment
                    and r["regularizer"] == reg
                ),
                None,
            )
            filename = f"encoder_{name}_seed{seed}.pt"
            if args.evaluate_only:
                model = behavior.Encoder()
                model.load_state_dict(
                    torch.load(args.output / filename, map_location="cpu")["encoder"]
                )
                metrics, _ = study.controls.evaluate(model, reference, c, seed)
                for key in metrics:
                    if not np.isclose(
                        metrics[key], found["metrics"][key], atol=1e-7, rtol=0
                    ):
                        raise RuntimeError("Evaluation changed")
            elif found is None:
                print(f"seed={seed}: {name}", flush=True)
                model = fit(data, c, seed, alignment, reg, reference)
                record = study.metric_record(
                    model, [], name, name, seed, seed, c, reference, args.output
                )
                # Keep a simple canonical path for checkpoint-only reevaluation.
                torch.save(
                    {"encoder": model.state_dict(), "config": vars(c)},
                    args.output / filename,
                )
                result["runs"].append(
                    {
                        "seed": seed,
                        "group": "new" if seed >= 18 else "diagnostic",
                        "alignment": alignment,
                        "regularizer": reg,
                        "metrics": record["metrics"],
                    }
                )
                report(args.output, result)
    print(
        f"{'Reproduced' if args.evaluate_only else 'Completed'} {len(result['runs'])} loss-ablation models.",
        flush=True,
    )


if __name__ == "__main__":
    main()

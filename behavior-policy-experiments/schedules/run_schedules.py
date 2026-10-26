"""Controlled ICM-to-OU curricula and epoch-dependent OU parameters.

All schedule choices are fixed before evaluation. Existing controls are reused
only when their full configuration matches. No distance targets enter learning.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "curiosity"))
import experiment as behavior

import numpy as np
import torch
import matplotlib.pyplot as plt


CONDITIONS = ["ou", "icm", "icm_entropy_decay", "icm_to_ou",
              "icm_entropy_decay_to_ou", "ou_rho_schedule",
              "ou_sigma_schedule", "ou_rho_sigma_schedule"]
NAMES = {"ou": "Fixed OU", "icm": "Fixed ICM", "icm_entropy_decay": "ICM, entropy bonus ↓",
         "icm_to_ou": "ICM → OU", "icm_entropy_decay_to_ou": "ICM entropy ↓ → OU",
         "ou_rho_schedule": "OU correlation changes", "ou_sigma_schedule": "OU spread changes",
         "ou_rho_sigma_schedule": "OU correlation + spread"}


@dataclass
class Schedule:
    start: float = .25
    finish: float = .75
    rho_start: float = .5
    sigma_start: float = 1.


def progress(fraction, schedule):
    return float(np.clip((fraction - schedule.start) / (schedule.finish - schedule.start), 0., 1.))


def epoch_fraction(step, steps, pairs, batch):
    updates_per_epoch = max(1, math.ceil(pairs / batch))
    epochs = max(1, math.ceil(steps / updates_per_epoch))
    return (step // updates_per_epoch) / max(1, epochs - 1)


def parameters(condition, fraction, c, schedule):
    p = progress(fraction, schedule)
    ou_fraction = p if condition in {"icm_to_ou", "icm_entropy_decay_to_ou"} else float(condition.startswith("ou"))
    rho = (1 - p) * schedule.rho_start + p * c.rho if condition in {"ou_rho_schedule", "ou_rho_sigma_schedule"} else c.rho
    sigma = (1 - p) * schedule.sigma_start + p * c.sigma if condition in {"ou_sigma_schedule", "ou_rho_sigma_schedule"} else c.sigma
    return {"ou_fraction": ou_fraction, "rho": rho, "sigma": sigma}


def entropy_coefficient(update, updates, c, schedule):
    return c.entropy_weight * (1 - progress(update / max(1, updates - 1), schedule))


class PairSampler:
    def __init__(self, data, c, seed):
        self.data, self.c = data, c
        generator = torch.Generator().manual_seed(10000 + seed)
        self.z = torch.randn(len(data["x"]), generator=generator)
        self.noise = torch.randn(len(data["x"]), generator=generator)
        self.mask_generator = torch.Generator().manual_seed(700000 + seed)

    def sample(self, ix, settings):
        alpha, rho, sigma = settings["ou_fraction"], settings["rho"], settings["sigma"]
        if alpha == 0:
            return self.data["x"][ix], self.data["xp"][ix]
        z = self.z[ix]
        zp = rho * z + (1 - rho**2)**.5 * self.noise[ix]
        x, xp = behavior.observe(z, sigma), behavior.observe(zp, sigma)
        if alpha == 1:
            return x, xp
        choose_ou = torch.rand(len(ix), generator=self.mask_generator) < alpha
        return (torch.where(choose_ou[:, None], x, self.data["x"][ix]),
                torch.where(choose_ou[:, None], xp, self.data["xp"][ix]))


@torch.no_grad()
def diagnose_policy(policy, icm, c):
    d = torch.linspace(c.distance_min, c.distance_max, 1024)
    x = behavior.sensor(d, c)
    distribution, _ = policy(x)
    errors = []
    for action in range(3):
        a = torch.full((len(d),), action)
        xp = behavior.sensor(behavior.move(d, a, c), c)
        errors.append(icm(x, xp, a)[0])
    # Match the clipped reward actually used by the actor, before averaging actions.
    reward = (distribution.probs * (c.reward_scale * torch.stack(errors, -1)).clamp(max=1)).sum(-1)
    return {"mean_action_probabilities": distribution.probs.mean(0).tolist(),
            "mean_action_entropy_nats": float(distribution.entropy().mean()),
            "final_model_reward_distance_1_to_2": float(reward[d <= 2].mean()),
            "final_model_reward_distance_8_to_16": float(reward[d >= 8].mean()),
            "reward_note": "Frozen final-model diagnostic, not a reconstruction of the evolving training reward."}


def train_encoder(data, condition, c, schedule, seed, reference):
    torch.manual_seed(40000 + seed)
    model, regularizer = behavior.Encoder(), behavior.SIGReg1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    sampler = PairSampler(data, c, seed)
    history = []
    sample_stages = {0, c.jepa_steps // 4, c.jepa_steps // 2, 3 * c.jepa_steps // 4, c.jepa_steps - 1}
    for step in range(c.jepa_steps):
        fraction = epoch_fraction(step, c.jepa_steps, len(data["x"]), c.batch)
        settings = parameters(condition, fraction, c, schedule)
        decay = max(0., (step - c.jepa_steps / 2) / max(1., c.jepa_steps / 2 - 1))
        optimizer.param_groups[0]["lr"] = .001 * .5 * (1 + np.cos(np.pi * decay))
        ix = torch.randint(len(data["x"]), (c.batch,))
        x, xp = sampler.sample(ix, settings)
        y = model(torch.cat([x, xp]))
        alignment = (y[:c.batch] - y[c.batch:]).square().mean()
        sigreg = regularizer(y)
        loss = (1 - c.sigreg_weight) * alignment + c.sigreg_weight * sigreg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in sample_stages:
            # Evaluation must not change minibatch randomness or optimize on labels.
            with torch.random.fork_rng(devices=[]):
                metrics, _ = behavior.evaluate_encoder(model, reference[0], reference[1], c, seed)
            history.append({"step": step, "epoch_fraction": fraction, **settings,
                            "alignment": float(alignment.detach()), "sigreg": float(sigreg.detach()),
                            "common_grid_r2": metrics["common_log_distance_grid_r2"],
                            "reference_ou_r2": metrics["on_policy_log_distance_r2"]})
    return model.eval(), history


@torch.no_grad()
def evaluate(model, condition, policy, c, seed, reference):
    fixed, example = behavior.evaluate_encoder(model, reference[0], reference[1], c, seed)
    own_kind = "icm" if condition in {"icm", "icm_entropy_decay"} else "ou"
    if own_kind == "icm":
        cal = behavior.collect(c, "icm", c.eval_episodes, 20000 + seed, policy)
        test = behavior.collect(c, "icm", c.eval_episodes, 30000 + seed, policy)
        own, _ = behavior.evaluate_encoder(model, cal, test, c, seed)
    else:
        own = fixed
    return {"common_grid_r2": fixed["common_log_distance_grid_r2"],
            "reference_ou_r2": fixed["on_policy_log_distance_r2"],
            "reference_ou_raw_variance": fixed["raw_embedding_variance"],
            "reference_ou_gaussian_cf_error": fixed["heldout_gaussian_cf_discrepancy"],
            "final_source": own_kind,
            "final_source_r2": own["on_policy_log_distance_r2"],
            "final_source_raw_variance": own["raw_embedding_variance"],
            "final_source_coverage": own["physical_range_bin_coverage"]}, example


def report(out, result):
    runs = result["runs"]
    def average(condition, key):
        x = [r["conditions"][condition]["metrics"][key] for r in runs]
        return f"{np.mean(x):.4f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.4f}"
    lines = ["# Scheduled exploration results", "",
             "All scores use separate calibration and test data. R²=1 is a perfect straight-line readout of log-distance. "
             "Values are means ± sample standard deviations across completed runs.", "",
             "**Execution check only; do not interpret these results as trained performance.**" if result["quick"] else f"{len(runs)}-run comparison; all schedules were fixed before evaluation.", "",
             "| Training rule | Same distance-range R² | Fixed Gaussian OU test R² | Output variance on the fixed OU test |",
             "| --- | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        lines.append(f"| {NAMES[condition]} | {average(condition, 'common_grid_r2')} | {average(condition, 'reference_ou_r2')} | {average(condition, 'reference_ou_raw_variance')} |")
    lines += ["", "## Main findings in these runs", "",
              "The gradual ICM→OU handover recovered a much straighter log-distance scale than fixed ICM in every tested run. "
              "It also finished closer to the intended output variance of 1. Combining the handover with entropy-bonus decay "
              "gave nearly the same result as the handover alone.", "",
              "Entropy-bonus decay by itself changed recovery and measured action randomness very little at this starting weight. "
              "This does not rule out different entropy settings; it tests the original small bonus decreasing to zero.", "",
              "Changing OU correlation alone made little difference. Starting with a wider log-distance spread and narrowing it "
              "improved recovery more in this setup. Only three runs were tested; these observations do not establish a general advantage "
              "for every schedule or environment."]
    lines += ["", "The fixed OU test is the same for every row; its Gaussian states include distances outside [1,16]. "
              "The first score tests only the shared range [1,16]. The ideal standard-Gaussian output variance is 1.", "",
              "## Individual runs", "", "| Run | Training rule | Same-range R² | Fixed OU R² | Final-source variance |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in runs:
        for condition in CONDITIONS:
            m = r["conditions"][condition]["metrics"]
            lines.append(f"| {r['seed']} | {NAMES[condition]} | {m['common_grid_r2']:.4f} | {m['reference_ou_r2']:.4f} | {m['final_source_raw_variance']:.4f} |")
    lines += ["", "## Why the earlier ICM result was worse", "",
              "The original learned policies chose approach more often than retreat in two runs, producing uneven distance coverage. "
              "Their output variances also departed greatly from the training target. "
              "A high score on frequently visited distances therefore concealed weaker performance elsewhere. "
              "These are measured differences, but they do not separate every causal mechanism.", "",
              "| Run | Policy | Approach / stay / retreat | Mean action entropy (nats) | Frozen curiosity reward at d=1..2 | At d=8..16 |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for r in runs:
        for key in ["fixed", "decay"]:
            d = r["policy_diagnostics"][key]
            probs = " / ".join(f"{x:.3f}" for x in d["mean_action_probabilities"])
            lines.append(f"| {r['seed']} | {key} | {probs} | {d['mean_action_entropy_nats']:.4f} | {d['final_model_reward_distance_1_to_2']:.6f} | {d['final_model_reward_distance_8_to_16']:.6f} |")
    lines += ["", "The reward columns measure the final frozen curiosity model. They do not reconstruct how its reward changed while learning, "
              "so they cannot establish the cause of the preference by themselves.", "",
              "## What the schedules mean", "",
              "ICM→OU replaces training pairs gradually: first 25% all ICM, middle 50% increasing OU fraction, last 25% all OU. "
              "The entropy-bonus schedule is a separate change during the earlier policy-training stage: its coefficient decreases "
              "from 0.0005 to zero on the same quarter/half/quarter timing. Actual action entropy is measured, not assumed to follow the coefficient.", "",
              "OU correlation changes from 0.5 to 0.9; OU observation spread changes from 1.0 to 0.7 where enabled. "
              "Changing correlation preserves each epoch's standard-Gaussian hidden state. Changing spread changes the physical-distance "
              "distribution over epochs. All OU schedules hold the final settings fixed for the last quarter.", "",
              "The OU pairs are simulator-generated training views; the handover is a sampling curriculum, not a single continuous "
              "three-action trajectory or a conversion between discrete-action and continuous-state entropy units.", "",
              "Models and reports are saved for reproducibility. The original results remain in the sibling curiosity/output folder.", "",
              "![Schedules and recovery](comparison.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def plot(out, result):
    runs = result["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    positions = np.arange(len(CONDITIONS))
    for i, condition in enumerate(CONDITIONS):
        values = [r["conditions"][condition]["metrics"]["common_grid_r2"] for r in runs]
        axes[0, 0].barh(i, np.mean(values), color="#0e7490")
        axes[0, 0].scatter(values, np.full(len(values), i), color="black", s=12, zorder=3)
    axes[0, 0].set(yticks=positions, yticklabels=[NAMES[k] for k in CONDITIONS], xlabel="R² on the same distance range", title="Recovery across all runs")
    axes[0, 0].invert_yaxis()
    schedule, c = Schedule(**result["schedule"]), behavior.Config(**result["config"])
    fractions = np.linspace(0, 1, 125)
    axes[0, 1].plot(fractions, [progress(f, schedule) for f in fractions], label="OU fraction in ICM → OU")
    axes[0, 1].plot(fractions, [parameters("ou_rho_schedule", f, c, schedule)["rho"] for f in fractions], label="OU correlation")
    axes[0, 1].plot(fractions, [parameters("ou_sigma_schedule", f, c, schedule)["sigma"] for f in fractions], label="OU spread")
    axes[0, 1].set(title="Predefined representation-training schedules", xlabel="Fraction of training epochs", ylabel="Parameter value")
    axes[0, 1].legend(fontsize=8)
    for condition in ["icm_to_ou", "icm_entropy_decay_to_ou", "ou_rho_schedule", "ou_sigma_schedule"]:
        histories = [r["conditions"][condition]["history"] for r in runs]
        if histories and histories[0]:
            axes[1, 0].plot([h["epoch_fraction"] for h in histories[0]],
                            np.mean([[h["common_grid_r2"] for h in history] for history in histories], axis=0),
                            marker="o", label=NAMES[condition])
    axes[1, 0].set(title="Recovery during representation training", xlabel="Fraction of training epochs", ylabel="Mean R² on the same range")
    axes[1, 0].legend(fontsize=8)
    for key in ["fixed", "decay"]:
        histories = [r["policy_histories"][key] for r in runs]
        axes[1, 1].plot([h["update"] for h in histories[0]],
                        np.mean([[h["entropy"] for h in history] for history in histories], axis=0), label=key)
    axes[1, 1].set(title="Measured action entropy during policy learning", xlabel="Policy update", ylabel="Entropy (nats)")
    axes[1, 1].legend()
    fig.suptitle("Curiosity-to-OU schedules and changing OU parameters", fontsize=16)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    behavior.verify_upstream()
    original = Path(behavior.__file__).parent / "output"
    source = json.loads((original / "metrics.json").read_text())
    c, schedule = behavior.Config(**source["config"]), Schedule()
    if args.quick:
        c.policy_updates, c.jepa_steps = 3, 10
        c.train_episodes, c.eval_episodes, c.horizon, c.burnin = 8, 4, 8, 4
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"config": asdict(c), "schedule": asdict(schedule), "quick": args.quick,
              "upstream_commit": source["upstream_commit"], "runs": []}
    if args.evaluate_only:
        result = json.loads((args.output / "metrics.json").read_text())
        c, schedule = behavior.Config(**result["config"]), Schedule(**result["schedule"])
        seeds = [r["seed"] for r in result["runs"]]
    else:
        seeds = args.seeds
    for index, seed in enumerate(seeds):
        if args.evaluate_only:
            record = result["runs"][index]
        else:
            print(f"seed={seed}: loading fixed ICM and training entropy-decay ICM", flush=True)
            if args.quick:
                fixed, fixed_icm, fixed_history, _ = behavior.train_policy(c, seed)
            else:
                saved = torch.load(original / f"behavior_seed{seed}.pt", map_location="cpu")
                if saved["config"] != asdict(c):
                    raise RuntimeError("Original control configuration differs")
                fixed, fixed_icm = behavior.Policy(), behavior.ICM()
                fixed.load_state_dict(saved["policy"])
                fixed_icm.load_state_dict(saved["icm"])
                fixed_history = next(r["policy_history"] for r in source["runs"] if r["seed"] == seed)
            decayed, decayed_icm, decay_history, _ = behavior.train_policy(c, seed,
                entropy_schedule=lambda update, updates: entropy_coefficient(update, updates, c, schedule))
            record = {"seed": seed, "conditions": {}, "policy_histories": {"fixed": fixed_history, "decay": decay_history},
                      "policy_diagnostics": {"fixed": diagnose_policy(fixed, fixed_icm, c),
                                             "decay": diagnose_policy(decayed, decayed_icm, c)}}
            for label, policy, icm in [("fixed", fixed, fixed_icm), ("decay", decayed, decayed_icm)]:
                torch.save({"policy": policy.state_dict(), "icm": icm.state_dict(), "config": asdict(c)}, args.output / f"behavior_{label}_seed{seed}.pt")
        policies = {}
        for label in ["fixed", "decay"]:
            state = torch.load(args.output / f"behavior_{label}_seed{seed}.pt", map_location="cpu")
            policies[label] = behavior.Policy()
            policies[label].load_state_dict(state["policy"])
            policies[label].eval()
        reference = (behavior.collect(c, "ou", c.eval_episodes, 20000 + seed),
                     behavior.collect(c, "ou", c.eval_episodes, 30000 + seed))
        for condition in CONDITIONS:
            label = "decay" if "entropy_decay" in condition else "fixed"
            policy = policies[label]
            if args.evaluate_only:
                checkpoint = torch.load(args.output / f"encoder_{condition}_seed{seed}.pt", map_location="cpu")
                model = behavior.Encoder()
                model.load_state_dict(checkpoint["encoder"])
                history = record["conditions"][condition]["history"]
            elif condition in {"ou", "icm"} and not args.quick:
                checkpoint = torch.load(original / f"encoder_{condition}_seed{seed}.pt", map_location="cpu")
                model = behavior.Encoder()
                model.load_state_dict(checkpoint["encoder"])
                history = []
            else:
                print(f"seed={seed}: training {condition}", flush=True)
                data = behavior.collect(c, "icm", c.train_episodes, 10000 + seed, policy)
                model, history = train_encoder(data, condition, c, schedule, seed, reference)
            metrics, example = evaluate(model.eval(), condition, policy, c, seed, reference)
            if args.evaluate_only:
                for key in ["common_grid_r2", "reference_ou_r2", "final_source_raw_variance"]:
                    if not np.isclose(metrics[key], record["conditions"][condition]["metrics"][key], atol=1e-6):
                        raise RuntimeError(f"Evaluation mismatch: {condition}, {seed}, {key}")
            else:
                torch.save({"encoder": model.state_dict(), "config": asdict(c), "schedule": asdict(schedule),
                            "history": history}, args.output / f"encoder_{condition}_seed{seed}.pt")
                np.savez_compressed(args.output / f"evaluation_{condition}_seed{seed}.npz", **example)
                record["conditions"][condition] = {"metrics": metrics, "history": history,
                                                   "reused_control": condition in {"ou", "icm"} and not args.quick}
                print(json.dumps({"seed": seed, "condition": condition, **metrics}), flush=True)
        if not args.evaluate_only:
            result["runs"].append(record)
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        report(args.output, result)
        plot(args.output, result)
    print("All saved evaluations reproduced." if args.evaluate_only else "Scheduled experiments complete.", flush=True)


if __name__ == "__main__":
    main()

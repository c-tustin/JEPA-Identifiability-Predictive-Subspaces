"""Draw the moving curve and finite one-step interventions from saved weights.

No training is run. The example is the previously studied seed 131, gain 4.
"""

import numpy as np
import torch

from report_mechanism import distance_axis, load_run, normalize, roots, save
from run_mechanism import OUT, ScaledEncoder, trials
from verify_root_motion import refined_roots


def evaluate(model, config, q):
    point = torch.as_tensor(q, dtype=torch.float64).clone().requires_grad_(True)
    y = model(trials.behavior.sensor(point.exp(), config))
    slope = torch.autograd.grad(y.sum(), point)[0]
    return y.detach().numpy(), slope.detach().numpy()


def curve_history(run, steps):
    """Use one fixed affine display transform for the entire training history."""
    factor, offset = normalize(run)
    frames = {f["step"]: f for f in run["frames"]}
    history = []
    for step in steps:
        if step == 4000:
            y, slope = run["final_curve"], run["final_slope"]
        else:
            y, slope = frames[step]["curve_before"], frames[step]["slope_before"]
        turns = roots(run["log_distance"], slope)
        history.append((step, factor*y+offset, factor*slope, turns))
    return history


def main():
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    run = load_run("ordinary", 4, 131)
    balanced = load_run("balanced", 4, 131)
    q = run["log_distance"]
    d = np.exp(q)
    steps = (100, 250, 500, 1000, 2000, 4000)
    palette = dict(zip(steps, plt.cm.viridis(np.linspace(.08, .9, len(steps)))))
    history = curve_history(run, steps)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, selected in enumerate(((100, 250, 500), (500, 1000, 2000, 4000))):
        for step, y, slope, turns in history:
            if step not in selected:
                continue
            color = palette[step]
            label = f"{step:,} updates"
            axes[0, col].plot(d, y, color=color, lw=2.3, label=label)
            axes[1, col].plot(d, slope, color=color, lw=2.3)
            for turn in turns:
                x = np.exp(turn)
                axes[0, col].scatter(x, np.interp(turn, q, y), color=color, s=50, zorder=5)
                axes[1, col].scatter(x, 0, color=color, s=50, zorder=5)
        axes[0, col].set_title("Early: the turn moves rapidly left" if col == 0 else "Later: it continues toward 2.4")
        axes[0, col].legend(frameon=False)
        axes[1, col].axhline(0, color=".3", ls="--", lw=1)
        axes[0, col].set_ylabel("Learned coordinate")
        axes[1, col].set_ylabel("Slope with respect to log distance")
        for ax in axes[:, col]:
            distance_axis(ax)
    fig.suptitle("Watch the turn move: ICM, seed 131\nDots mark turns; a turn occurs where the slope crosses zero. Shading marks distances 2–4.")
    fig.supxlabel("One fixed affine display normalization across all updates; no unfolding or per-frame rescaling.", fontsize=10)
    save(fig, "leftward_curve_and_slope")

    config = trials.behavior.Config(**run["config"])
    model = ScaledEncoder(4).double()
    factor, _ = normalize(run)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    colors = {"before": ".35", "actual": "black", "alignment_only": "#0072B2", "sigreg_only": "#D55E00"}
    labels = {"before": "Before update", "actual": "Actual full update", "alignment_only": "Alignment component only", "sigreg_only": "SIGReg component only"}
    grid = np.linspace(0, np.log(16), 2049)
    for ax, step in zip(axes, (250, 500, 1000)):
        snapshot = run["snapshots"][step]
        before = {k: v.double() for k, v in snapshot["before"].items()}
        states = {"before": before, "actual": snapshot["after"]}
        for name, direction in (("alignment_only", "direction_a"), ("sigreg_only", "direction_r")):
            states[name] = {k: before[k] + snapshot[direction][k].double() for k in before}
        turn = {}
        for name, state in states.items():
            model.load_state_dict(state)
            _, slope = evaluate(model, config, grid)
            candidates = refined_roots(model, config, grid, slope)
            assert len(candidates) == 1, (step, name, candidates)
            turn[name] = candidates[0]
        root = turn["before"]
        width = 1.6 * max(abs(v-root) for v in turn.values())
        local_q = np.linspace(root-width, root+width, 401)
        for name, state in states.items():
            model.load_state_dict(state)
            _, slope = evaluate(model, config, local_q)
            ax.plot(100*(local_q-root), factor*slope, color=colors[name],
                    ls="--" if name == "before" else "-", lw=2,
                    alpha=.75 if name == "actual" else 1, label=labels[name])
            ax.scatter(100*(turn[name]-root), 0, color=colors[name], s=35, zorder=5)
        ax.axhline(0, color=".65", lw=.8)
        ax.axvline(0, color=".65", lw=.8)
        ax.set_title(f"Update {step} → {step+1}\nOld turn at distance {np.exp(root):.3f}")
        ax.set_xlabel("100 × log(distance / old turning distance)\n← leftward movement")
        ax.set_ylabel("Slope with respect to log distance")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=.15)
        row = dict(updates_before=step, distance_before=float(np.exp(root)))
        for name, value in turn.items():
            row[name+"_distance"] = float(np.exp(value))
            row[name+"_log_shift"] = float(value-root)
        assert row["actual_log_shift"] < 0
        rows.append(row)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=4, loc="outside lower center", frameon=False, fontsize=10)
    fig.suptitle("Why left? These updates move the zero crossing left\nExact finite parameter interventions, with Adam's shared denominator and saved momentum. Each panel has its own zoom.")
    save(fig, "leftward_one_update_interventions")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    for ax, condition, data in zip(axes, ("Ordinary sampling", "Balance only SIGReg sampling"), (run, balanced)):
        for step, y, slope, turns in curve_history(data, steps):
            if step not in (250, 500, 1000, 4000):
                continue
            ax.plot(d, y, color=palette[step], lw=2, label=f"{step:,} updates")
            for turn in turns:
                ax.scatter(np.exp(turn), np.interp(turn, q, y), color=palette[step], s=45)
        ax.set_title(condition)
        distance_axis(ax)
        ax.set_ylabel("Learned coordinate")
        ax.legend(frameon=False)
    fig.suptitle("Same ICM observations and alignment pairs; change which views SIGReg receives\nSeed 131, gain 4. Fixed normalization over time within each condition; dots mark turns.")
    save(fig, "leftward_sampling_control")
    trials.atomic_json(OUT / "leftward-explanation.json", dict(
        seed=131, gain=4, new_training=False, finite_update_interventions=rows,
        note="Counterfactual components include momentum history and the actual shared Adam denominator. They are not independently trained optimizers. Display normalization is fixed over time."))
    print(rows)


if __name__ == "__main__":
    main()

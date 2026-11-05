"""Show actual collected views and motion, including near-static policies."""

import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_targeted import OUT, load, source_path, targeted_policies
from analyze_targeted import save


def main():
    seed = 8  # Fixed first discovery seed, not selected by its performance.
    names = list(targeted_policies.POLICIES)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained", sharex=True)
    grid = np.linspace(-.5, 3.3, 400)
    for ax, name in zip(axes.flat, names + ["ou"]):
        saved = load(source_path(name, seed))
        q = math.log(4) + .7 * saved["data"]["z"].numpy()
        ax.hist(q, bins=np.linspace(-.5, 3.3, 45), density=True, color="#0072B2", alpha=.65)
        if name == "ou":
            deviation = .7
            label = "OU marginal"
        else:
            deviation = targeted_policies.POLICIES[name]().standard_deviation
            label = "Initial Gaussian target" if name == "gaussian_start_random" else "Design target"
        density = np.exp(-.5 * ((grid - math.log(4)) / deviation)**2) / (deviation * np.sqrt(2 * np.pi))
        ax.plot(grid, density, color="#D55E00", ls="--", label=label)
        ax.axvline(0, color=".4", lw=.8)
        ax.axvline(math.log(16), color=".4", lw=.8)
        stats = saved["statistics"]
        motion = "independent pairs; no actions" if name == "ou" else f"stay={stats['stay_fraction']:.2f}"
        ax.set_title(name.replace("gaussian_", "") + f"\n{motion}, Gaussian discrepancy={stats['gaussian_cdf_distance']:.3f}", fontsize=9)
        ax.set(xlabel="Log physical distance", ylabel="Density")
        ax.grid(alpha=.12)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("What the controllers actually collect: seed 8\nVertical lines mark physical bounds. OU is the legacy Gaussian pair generator; new policies use actual physical actions.")
    save(fig, "collected_distributions")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained", sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        saved = load(source_path(name, seed))
        z = saved["data"]["z"].numpy().reshape(64, 256)
        distance = 4 * np.exp(.7 * z)
        for agent in (0, 1, 2):
            ax.plot(range(64), distance[:, agent], lw=1.5, label=f"Agent {agent}")
        ax.set(title=name.replace("gaussian_", ""), xlabel="Recorded physical step", ylabel="Physical distance")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.grid(alpha=.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Same interaction budget: three fixed agents from each physical policy\nAll controllers use 128 burn-in steps plus these 64 recorded steps; no policy changes the reset distribution.")
    save(fig, "physical_trajectories")


if __name__ == "__main__":
    main()

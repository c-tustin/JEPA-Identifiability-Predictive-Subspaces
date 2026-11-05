"""Compare broad collection with the earlier Gaussian-steered collection."""

import math
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_broad import HERE, OUT, broad_policies, load, source_path
from analyze_broad import save


def main():
    seed = 8
    names = list(broad_policies.POLICIES) + ["gaussian_wide_diffusion", "ou"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained", sharex=True)
    for ax, name in zip(axes.flat, names):
        path = source_path(name, seed)
        if name == "gaussian_wide_diffusion":
            path = HERE.parent / "targeted-policies" / "output" / "sources" / f"{name}-{seed}.pt"
        source = load(path)
        q = math.log(4) + .7 * source["data"]["z"].numpy()
        ax.hist(q, bins=np.linspace(-.5, 3.3, 45), density=True, color="#0072B2", alpha=.65)
        ax.axvline(0, color=".4", lw=.8)
        ax.axvline(math.log(16), color=".4", lw=.8)
        statistics = source["statistics"]
        motion = "no physical actions" if name == "ou" else f"stay={statistics['stay_fraction']:.2f}"
        ax.set_title(f"{name}\n{motion}, Gaussian discrepancy={statistics['gaussian_cdf_distance']:.3f}", fontsize=9)
        ax.set_xticks(np.log([1, 2, 4, 8, 16]), ["1", "2", "4", "8", "16"])
        ax.set(xlabel="Physical distance (log scale)", ylabel="Density per log distance")
        ax.grid(alpha=.15)
    fig.suptitle("Broader-coverage follow-up: actual collected states, seed 8\nPhysical controllers share the action budget. Earlier Gaussian steering and the abstract fixed-OU pair reference are shown for context.")
    save(fig, "collected_distributions")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", sharex=True, sharey=True)
    for ax, name in zip(axes.flat, broad_policies.POLICIES):
        source = load(source_path(name, seed))
        distance = 4 * np.exp(.7 * source["data"]["z"].numpy().reshape(64, 256))
        for agent in (0, 1, 2):
            ax.plot(range(64), distance[:, agent], lw=1.5, label=f"Agent {agent}")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set(title=name, xlabel="Recorded physical step", ylabel="Physical distance")
        ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside lower center", frameon=False)
    fig.suptitle("Three fixed agents from each new broad-coverage policy\n128 burn-in actions precede these 64 recorded actions. Agents 0, 1 and 2 were not selected by performance.")
    save(fig, "physical_trajectories")


if __name__ == "__main__":
    main()

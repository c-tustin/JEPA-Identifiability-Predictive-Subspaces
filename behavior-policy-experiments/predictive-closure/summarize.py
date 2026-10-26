"""Generate a concise report from saved per-seed results."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "output"


def load(name):
    return json.loads((OUT / name).read_text())


def avg(rows, key):
    return np.mean([r[key] for r in rows])


def cell(rows, key):
    values = [r[key] for r in rows]
    return f"{np.mean(values):.4f} ± {np.std(values,ddof=1):.4f}"


def main():
    c = load("controls.json")
    learned = load("learned.json")
    text = [
        "# Predictive Closure extensions: measured results",
        "",
        "All runs used cloud CPU compute and one physical distance coordinate. Values are mean ± sample SD across five seeds unless stated. [Protocol](PROTOCOL.md) · [Run status](RUN_STATUS.md).",
        "",
        "## Conditional means can miss predictive information",
        "",
        "The signed transition has estimated MI "
        + cell([r for r in c["moments"] if r["law"] == "signed"], "mi")
        + " nats, versus "
        + cell([r for r in c["moments"] if r["law"] == "ar"], "mi")
        + " for ordinary AR. Its analytic lower bound is "
        + f"{c['analytic']['signed_mi_lower_bound']:.4f}"
        + " nats. Signed mean-only MSE is 1; AR MSE is .75. This compares two laws, not competing encoders in one world. The signed law violates the conditional Gaussian assumption in Appendix A.2. It illustrates why that assumption matters; it does not refute the theorem.",
        "",
        "| Frozen calibrated coordinate / fitted predictor | Test NLL | Test MSE |",
        "| --- | ---: | ---: |",
    ]
    for kind in ["mean", "heteroscedastic", "mixture", "conditioned"]:
        rows = [r for r in c["predictors"] if r["kind"] == kind]
        text.append(f'| {kind} | {cell(rows,"nll")} | {cell(rows,"mse")} |')
    text += [
        "",
        "All 20 two-parameter fits converged. NLL above is comparable in the same standardized log-distance coordinate. A mixture captures magnitude-dependent uncertainty without improving the Bayes mean. The current sign/action makes the conditional mean useful. The inverse renderer is explicit calibration, not learned identification.",
        "",
        "## The learned pilot retained distance in every arm",
        "",
        "| Training objective | Log-distance probe R² | Distance probe R² | Magnitude probe R² | Sign accuracy | Normalized latent MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    explicit = load("explicit-encoders.json")
    section = [
        "",
        "## Explicit encoders in the same world",
        "",
        "A prospective follow-up to [the same-world derivation](../research-next/SIGNED_OU_EXTENSION.md) compares full Z with Gaussianized |Z|, without changing the transition law. These encoders use known geometry and are not learned by JEPA.",
        "",
        "| Explicit encoder, clipping 1e-8, no rounding | Held-out spline mean MSE | MSE / target variance |",
        "| --- | ---: | ---: |",
    ]
    for kind in ["full", "magnitude"]:
        rows = [
            r
            for r in explicit["fits"]
            if r["kind"] == kind and r["epsilon"] == 1e-8 and r["step"] is None
        ]
        section.append(
            f'| {kind} | {cell(rows,"test_mse")} | {cell(rows,"test_normalized_mse")} |'
        )
    section += [
        "",
        "Both predictors have 23 spline coefficients, identical training sample sizes and fixed knots; five fresh seeds. The folded encoder predicts its own future much better but cannot distinguish physical distances d and 16/d. Dropping sign has irreducible physical-distance MSE "
        + cell(
            explicit["continuous_identity"], "irreducible_distance_mse_from_lost_sign"
        )
        + " under the stationary distribution (estimated from fresh test samples). This is a mean-error diagnostic across explicit coordinates, not coordinate-independent physical prediction performance.",
        "",
        "The ideal continuous density ratios agree pointwise to <2e-15, verifying the analytic equal-information identity. Clipping and rounding alter that statement. Known-law quadrature gives:",
        "",
        "| Rounding step, clipping 1e-8 | Full MI, nats | Magnitude MI, nats |",
        "| ---: | ---: | ---: |",
    ]
    for step in [0.5, 0.25, 0.125]:
        rows = [
            r
            for r in explicit["quantized"]
            if r["epsilon"] == 1e-8 and r["step"] == step
        ]
        values = {r["kind"]: r["mi"] for r in rows}
        section.append(f'| {step} | {values["full"]:.6f} | {values["magnitude"]:.6f} |')
    section += [
        "",
        "All three clipping levels and three rounding steps are retained in explicit-encoders.json. The 64/128-node quadrature difference is <1e-8, mass error <1e-10 and marginal error <1e-9. At these finite resolutions MI does not tie: the two maps allocate magnitude resolution differently. This construction violates conditional-Gaussian assumptions and has not checked the paper’s reader/capacity conditions. It does not refute Appendix A.2 or show the neural optimizer selects the fold.",
        "",
    ]
    index = text.index("## The learned pilot retained distance in every arm")
    text[index:index] = section
    for kind in ["mean", "gaussian", "mixture", "conditioned"]:
        rows = [r for r in learned if r["kind"] == kind]
        text.append(
            "| "
            + kind
            + " | "
            + " | ".join(
                cell(rows, k)
                for k in [
                    "log_distance_r2",
                    "distance_r2",
                    "magnitude_r2",
                    "sign_accuracy",
                    "normalized_mse",
                ]
            )
            + " |"
        )
    text += [
        "",
        "Twenty encoders, five independent seeds, identical total parameter counts and updates: 2,737 encoder parameters plus 1,251 predictor parameters (99 in the final three-output layer), 3,988 total. Each received 2,000 updates of batch 256. Output-channel redundancy means effective capacities differ despite identical counts. All tested grids have zero detected reversals. Gaussian KS statistics range from "
        + f"{min(r['gaussian_ks'] for r in learned):.3f} to {max(r['gaussian_ks'] for r in learned):.3f}"
        + "; Gaussian regularization is approximate. These results do **not** show a feature-selection separation or that distributional training preserves more distance than mean training. Approximate Gaussian regularization and the smooth scalar encoder may favor an informative coordinate here; that explanation is a conjecture. The conditional mean is not the whole predictive distribution, but that alone did not make this mean-trained encoder discard distance.",
        "",
        "Held-out density in learned encoders is saved in learned.json, measured in each encoder’s own standardized coordinate. Those NLLs cannot rank physical predictive density across encoders. Fixed polynomial probes and a finite grid are diagnostics, not proofs of global invertibility or good control. All 20 checkpoint evaluations reproduce exactly.",
        "",
        "## Independent-noise drift learning: extrapolation increased error",
        "",
        "| Study / fit | Drift RMSE | One-second transition RMSE |",
        "| --- | ---: | ---: |",
    ]
    for filename, label in [
        ("drift.json", "Initial"),
        ("drift-weighted.json", "Fresh-seed follow-up"),
    ]:
        data = load(filename)
        for kind in sorted(set(r["kind"] for r in data["fits"])):
            rows = [r for r in data["fits"] if r["kind"] == kind]
            text.append(
                f'| {label}: {kind} | {cell(rows,"drift_rmse")} | {cell(rows,"transition_rmse")} |'
            )
    text += [
        "",
        "Unweighted multi-interval fitting was worse than single-interval fitting in the initial five seeds. A prospective fresh-seed follow-up tested inverse-variance weighting after that finding; keep the two studies separate. The true drift and diffusion shape are included in the fitted/assumed families, and exact sensor calibration is supplied. The finite correction basis is approximate; variance, correction misspecification and discretization are not isolated by these results. These are favorable simulator controls, not learned representation identification.",
        "",
        "| Study / controller | Closest physical error | Hit rate within .1 | Final physical error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for filename, label in [
        ("drift.json", "Initial"),
        ("drift-weighted.json", "Follow-up"),
    ]:
        data = load(filename)
        for kind, penalty in sorted(
            set((r["kind"], r["penalty"]) for r in data["planning"])
        ):
            rows = [
                r
                for r in data["planning"]
                if r["kind"] == kind and r["penalty"] == penalty
            ]
            # Tasks are nested within seed: aggregate before estimating spread.
            aggregated = [
                {
                    k: avg([r for r in rows if r["seed"] == s], k)
                    for k in ["closest_error", "hit", "final_error"]
                }
                for s in data["settings"]["seeds"]
            ]
            text.append(
                f"| {label}: {kind}, risk {penalty:g} | "
                + " | ".join(
                    cell(aggregated, k) for k in ["closest_error", "hit", "final_error"]
                )
                + " |"
            )
    text += [
        "",
        "Sixteen tasks per seed are nested, not 80 independent model replications. Mean-plus-SD is a risk heuristic over fitted stochastic rollouts, not a worst-case or calibrated robustness guarantee. No clear control advantage follows from extrapolation or this risk penalty. The task is local and calibration-assisted; the simulator has no reflecting boundary in this interior diagnostic.",
        "",
        "## Best next research steps",
        "",
        "1. Allocate interval samples by variance and bias, and test misspecified drift/diffusion bases. The fresh-seed weighted fit is a starting control, not a general solution.",
        "2. Replace exact inverse-renderer calibration with an information-preserving learned coordinate; measure inverse sensitivity and physical control before claiming identification.",
        "3. The explicit same-world fold is now checked numerically. Next study actual learned selection with reader tolerances, capacity and finite precision made explicit; the present neural pilot did not select that fold.",
        "4. Test action masking or noisy action side information with a fixed encoder before another joint-training comparison. This isolates predictor information from representation changes.",
        "",
        "No universal objective ranking, nonlinear JEPA identification theorem, or reproduction of the paper’s CIFAR experiments is claimed.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(text) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for i, kind in enumerate(["mean", "heteroscedastic", "mixture", "conditioned"]):
        rows = [r for r in c["predictors"] if r["kind"] == kind]
        ax[0].scatter([i] * 5, [r["nll"] for r in rows])
    ax[0].set_xticks(range(4), ["Mean", "Gaussian\nvariance", "Mixture", "Action"])
    ax[0].set_ylabel("Test NLL (common coordinate)")
    ax[0].set_title("Frozen calibrated coordinate")
    for i, kind in enumerate(["mean", "gaussian", "mixture", "conditioned"]):
        rows = [r for r in learned if r["kind"] == kind]
        ax[1].scatter([i] * 5, [r["log_distance_r2"] for r in rows])
    ax[1].set_xticks(range(4), ["Mean", "Gaussian", "Mixture", "Action"])
    ax[1].set_ylim(0.99, 1)
    ax[1].set_ylabel("Held-out log-distance probe R²")
    ax[1].set_title("Learned scalar encoders")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

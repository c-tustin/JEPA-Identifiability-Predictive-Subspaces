# Predictive Closure extensions: measured results

All runs used cloud CPU compute and one physical distance coordinate. Values are mean ± sample SD across five seeds unless stated. [Protocol](PROTOCOL.md) · [Run status](RUN_STATUS.md).

## Conditional means can miss predictive information

The signed transition has estimated MI 0.6638 ± 0.0022 nats, versus 0.1439 ± 0.0009 for ordinary AR. Its analytic lower bound is 0.4708 nats. Signed mean-only MSE is 1; AR MSE is .75. This compares two laws, not competing encoders in one world. The signed law violates the conditional Gaussian assumption in Appendix A.2. It illustrates why that assumption matters; it does not refute the theorem.

| Frozen calibrated coordinate / fitted predictor | Test NLL | Test MSE |
| --- | ---: | ---: |
| mean | 1.4189 ± 0.0012 | 0.9998 ± 0.0025 |
| heteroscedastic | 1.0959 ± 0.0030 | 0.9996 ± 0.0025 |
| mixture | 0.7554 ± 0.0039 | 0.9996 ± 0.0025 |
| conditioned | 0.2567 ± 0.0029 | 0.0978 ± 0.0006 |

All 20 two-parameter fits converged. NLL above is comparable in the same standardized log-distance coordinate. A mixture captures magnitude-dependent uncertainty without improving the Bayes mean. The current sign/action makes the conditional mean useful. The inverse renderer is explicit calibration, not learned identification.


## Explicit encoders in the same world

A prospective follow-up to [the same-world derivation](../research-next/SIGNED_OU_EXTENSION.md) compares full Z with Gaussianized |Z|, without changing the transition law. These encoders use known geometry and are not learned by JEPA.

| Explicit encoder, clipping 1e-8, no rounding | Held-out spline mean MSE | MSE / target variance |
| --- | ---: | ---: |
| full | 1.0067 ± 0.0064 | 1.0059 ± 0.0073 |
| magnitude | 0.3079 ± 0.0018 | 0.3085 ± 0.0028 |

Both predictors have 23 spline coefficients, identical training sample sizes and fixed knots; five fresh seeds. The folded encoder predicts its own future much better but cannot distinguish physical distances d and 16/d. Dropping sign has irreducible physical-distance MSE 13.3684 ± 0.1415 under the stationary distribution (estimated from fresh test samples). This is a mean-error diagnostic across explicit coordinates, not coordinate-independent physical prediction performance.

The ideal continuous density ratios agree pointwise to <2e-15, verifying the analytic equal-information identity. Clipping and rounding alter that statement. Known-law quadrature gives:

| Rounding step, clipping 1e-8 | Full MI, nats | Magnitude MI, nats |
| ---: | ---: | ---: |
| 0.5 | 0.540318 | 0.599453 |
| 0.25 | 0.626938 | 0.645953 |
| 0.125 | 0.653900 | 0.658964 |

All three clipping levels and three rounding steps are retained in explicit-encoders.json. The 64/128-node quadrature difference is <1e-8, mass error <1e-10 and marginal error <1e-9. At these finite resolutions MI does not tie: the two maps allocate magnitude resolution differently. This construction violates conditional-Gaussian assumptions and has not checked the paper’s reader/capacity conditions. It does not refute Appendix A.2 or show the neural optimizer selects the fold.

## The learned pilot retained distance in every arm

| Training objective | Log-distance probe R² | Distance probe R² | Magnitude probe R² | Sign accuracy | Normalized latent MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean | 0.9983 ± 0.0002 | 0.9919 ± 0.0101 | 0.9940 ± 0.0006 | 0.9967 ± 0.0012 | 1.0023 ± 0.0224 |
| gaussian | 0.9984 ± 0.0002 | 0.9923 ± 0.0098 | 0.9941 ± 0.0005 | 0.9968 ± 0.0011 | 1.0029 ± 0.0231 |
| mixture | 0.9984 ± 0.0002 | 0.9920 ± 0.0100 | 0.9940 ± 0.0006 | 0.9967 ± 0.0012 | 1.0024 ± 0.0227 |
| conditioned | 0.9984 ± 0.0003 | 0.9922 ± 0.0099 | 0.9941 ± 0.0006 | 0.9966 ± 0.0010 | 0.1086 ± 0.0029 |

Twenty encoders, five independent seeds, identical total parameter counts and updates: 2,737 encoder parameters plus 1,251 predictor parameters (99 in the final three-output layer), 3,988 total. Each received 2,000 updates of batch 256. Output-channel redundancy means effective capacities differ despite identical counts. All tested grids have zero detected reversals. Gaussian KS statistics range from 0.044 to 0.058; Gaussian regularization is approximate. These results do **not** show a feature-selection separation or that distributional training preserves more distance than mean training. Approximate Gaussian regularization and the smooth scalar encoder may favor an informative coordinate here; that explanation is a conjecture. The conditional mean is not the whole predictive distribution, but that alone did not make this mean-trained encoder discard distance.

Held-out density in learned encoders is saved in learned.json, measured in each encoder’s own standardized coordinate. Those NLLs cannot rank physical predictive density across encoders. Fixed polynomial probes and a finite grid are diagnostics, not proofs of global invertibility or good control. All 20 checkpoint evaluations reproduce exactly.

## Independent-noise drift learning: extrapolation increased error

| Study / fit | Drift RMSE | One-second transition RMSE |
| --- | ---: | ---: |
| Initial: multi | 0.0111 ± 0.0028 | 0.0121 ± 0.0029 |
| Initial: single | 0.0030 ± 0.0008 | 0.0033 ± 0.0007 |
| Fresh-seed follow-up: multi | 0.0107 ± 0.0022 | 0.0107 ± 0.0030 |
| Fresh-seed follow-up: multi_wls | 0.0094 ± 0.0023 | 0.0096 ± 0.0026 |
| Fresh-seed follow-up: single | 0.0032 ± 0.0012 | 0.0032 ± 0.0016 |

Unweighted multi-interval fitting was worse than single-interval fitting in the initial five seeds. A prospective fresh-seed follow-up tested inverse-variance weighting after that finding; keep the two studies separate. The true drift and diffusion shape are included in the fitted/assumed families, and exact sensor calibration is supplied. The finite correction basis is approximate; variance, correction misspecification and discretization are not isolated by these results. These are favorable simulator controls, not learned representation identification.

| Study / controller | Closest physical error | Hit rate within .1 | Final physical error |
| --- | ---: | ---: | ---: |
| Initial: multi, risk 0 | 0.0826 ± 0.0215 | 0.7250 ± 0.1218 | 0.2519 ± 0.0284 |
| Initial: multi, risk 1 | 0.0831 ± 0.0210 | 0.7250 ± 0.1135 | 0.2460 ± 0.0415 |
| Initial: single, risk 0 | 0.0823 ± 0.0105 | 0.7375 ± 0.1202 | 0.2305 ± 0.0242 |
| Initial: single, risk 1 | 0.0797 ± 0.0230 | 0.7375 ± 0.1118 | 0.2421 ± 0.0503 |
| Initial: stay, risk 0 | 0.4014 ± 0.0281 | 0.0250 ± 0.0342 | 1.0675 ± 0.3038 |
| Follow-up: multi, risk 0 | 0.0876 ± 0.0272 | 0.7375 ± 0.1118 | 0.2910 ± 0.0707 |
| Follow-up: multi, risk 1 | 0.0842 ± 0.0244 | 0.7500 ± 0.0884 | 0.2777 ± 0.0519 |
| Follow-up: multi_wls, risk 0 | 0.0850 ± 0.0177 | 0.7500 ± 0.0765 | 0.3000 ± 0.0748 |
| Follow-up: multi_wls, risk 1 | 0.0814 ± 0.0182 | 0.7500 ± 0.0765 | 0.2748 ± 0.0399 |
| Follow-up: single, risk 0 | 0.0873 ± 0.0125 | 0.7250 ± 0.0713 | 0.3002 ± 0.0757 |
| Follow-up: single, risk 1 | 0.0725 ± 0.0148 | 0.8000 ± 0.0523 | 0.2548 ± 0.0525 |
| Follow-up: stay, risk 0 | 0.3840 ± 0.0338 | 0.0250 ± 0.0342 | 1.0399 ± 0.1439 |

Sixteen tasks per seed are nested, not 80 independent model replications. Mean-plus-SD is a risk heuristic over fitted stochastic rollouts, not a worst-case or calibrated robustness guarantee. No clear control advantage follows from extrapolation or this risk penalty. The task is local and calibration-assisted; the simulator has no reflecting boundary in this interior diagnostic.

## Best next research steps

1. Allocate interval samples by variance and bias, and test misspecified drift/diffusion bases. The fresh-seed weighted fit is a starting control, not a general solution.
2. Replace exact inverse-renderer calibration with an information-preserving learned coordinate; measure inverse sensitivity and physical control before claiming identification.
3. The explicit same-world fold is now checked numerically. Next study actual learned selection with reader tolerances, capacity and finite precision made explicit; the present neural pilot did not select that fold.
4. Test action masking or noisy action side information with a fixed encoder before another joint-training comparison. This isolates predictor information from representation changes.

No universal objective ranking, nonlinear JEPA identification theorem, or reproduction of the paper’s CIFAR experiments is claimed.

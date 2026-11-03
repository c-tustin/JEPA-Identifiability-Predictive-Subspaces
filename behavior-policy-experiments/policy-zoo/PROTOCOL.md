# Fixed protocol

Written before any full-size zoo result was produced. Policy settings, filter cutoffs, seeds, outcomes and tests below are not changed after inspecting results.

## Training (identical to the reliability study)

Each fit calls the reliability-study code: encoder initialization `40000 + seed`, the historical pair-index stream for that data seed, 4,000 Adam updates, batch 512, loss $0.9\,\text{alignment}+0.1\,\text{SIGReg}$, original learning-rate schedule. Training data are 256 trajectories × 64 pairs after 128 burn-in steps (16,384 pairs), collected with seed `10000 + seed`. Fitting receives only observation pairs.

- **Handover**: the policy's data, then the original 25%→75% handover to fixed OU pairs. This is how random, ICM, RND, disagreement, count and action-balanced were tested before.
- **Pure**: the policy's data only, for all 4,000 updates. This isolates the policy itself.

Data seeds 8–27 (20 seeds). Seeds 8–12 for previously tested sources must reproduce the saved reliability metrics exactly; this is checked, not assumed. All 20 seeds are rerun for previously tested sources, so every comparison is paired by data seed.

## Physical policies (three actions: approach, stay, retreat by 0.25)

| Name | Rule |
| --- | --- |
| sticky | Repeat previous action with probability 0.9, else uniform |
| lazy | Stay 0.8, approach 0.1, retreat 0.1 |
| biased | Approach 0.36, stay 0.33, retreat 0.31 |
| levy | Uniform action held for Pareto(1.5) run lengths, cap 1,000 |
| sweep | Move one way until apparent height ≥ 0.9 or ≤ 1/15, then reverse; 10% uniform |
| homing | Step toward apparent height 1/4 (outside ±0.1 in log height); 30% uniform |
| task | A2C, same actor/optimizer/budget as ICM, extrinsic reward $\exp(-(\log(4h)/0.35)^2)$ |
| ou_actions | OU(ρ=0.9) noise quantized at ±Φ⁻¹(2/3), so each action has marginal 1/3 |
| ou_actions_lp_w0.1 | Same, noise first low-passed at ω=0.1 |
| ou_actions_hp_w0.3 | Same, noise first high-passed at ω=0.3 |
| ou_actions_hp2_w3 | Same, noise second-order high-passed at ω=3 (anti-persistent) |

Previously tested: random, ICM, RND, disagreement, count, action-balanced (unchanged code).

## Latent Gaussian sources (pure only)

Stationary OU(ρ=0.9) trajectories in $z$, filtered by first-order stages with $a=e^{-\omega}$: low-pass $y_t=ay_{t-1}+(1-a)x_t$, high-pass $y_t=a(y_{t-1}+x_t-x_{t-1})$. Output is scaled to unit stationary variance after 1,000 warm-up steps, so every source has exact N(0,1) marginals. Consecutive pairs are used.

| Name | Filter | Lag-1 correlation ρ |
| --- | --- | ---: |
| ou_traj | none | 0.900 |
| ou_lp_w1 / w0.3 / w0.1 | low-pass | 0.953 / 0.984 / 0.995 |
| ou_hp_w0.1 / w0.3 / w1 | high-pass | 0.810 / 0.654 / 0.300 |
| ou_hp2_w1 / w3 | two high-pass stages | −0.073 / −0.452 |

Every filtered source has a matched control `iid_<name>`: independent Gaussian pairs with the same ρ (same RNG stream as fixed OU). Any Gaussian stationary pair is bivariate normal, so filtering should matter only through ρ. The population alignment loss $2(1-\sum_k c_k^2\rho^k)$ predicts recovery for ρ>0 with a gap $\rho-\rho^2$, and a folded (second-Hermite) optimum for ρ<0.

## Outcomes and statistics

Primary: shared-range affine log-distance $R^2$ on the common grid. Secondary: broader fixed-OU $R^2$; fold = any reversed grid step. Descriptive source statistics (size-bin entropy, coverage, Gaussian-CDF distance, lag correlation, stay fraction) use labels only for description.

- Per condition: mean, SD, median, 95% percentile bootstrap CI of the mean (10,000 resamples, fixed seed), fold count with Wilson 95% CI.
- Paired by seed, two-sided exact sign-flip permutation test on the mean $R^2$ difference against three references: random (same protocol), ICM (same protocol) and fixed OU (pure). Holm correction within each protocol × reference family.
- Fold rates against ICM: exact McNemar test, Holm within protocol.
- Handover vs pure for each physical policy: paired permutation test, Holm.
- Each filtered latent source against its matched i.i.d. control: paired permutation test, Holm.
- Across physical policies: Spearman correlation of mean pure $R^2$ with each source statistic (descriptive, n = 17 policies).

Twenty seeds per condition can detect large effects only; a non-significant difference is not evidence of equivalence. Results are conditional on this toy world, encoder and budget.

## Addendum (after the main results)

The `pure_long` runs were added after inspecting the main results, so they are exploratory. They cover fixed OU, `ou_hp2_w1`, `ou_hp2_w3` and both matched i.i.d. controls, at 8,000 updates (the reliability study's doubling). They test whether the negative-ρ sources reach the lower-loss folded optimum with more training. No preregistered result or test was changed.

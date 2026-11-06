# Fresh-seed confirmation: what carries over?

**Fixed before the cohort:** 60 new actor/training seeds; ±0.001 R² and ±0.005 physical goal error. The latter is 2% of an action step. No old outcomes are pooled into the tests.

**Primary conjunction passes after the four-claim Holm correction.** All policy pairs must be equivalent on all three recovery regions and shared-data reaching. This is an empirical statement about matched exposure, not a theorem for arbitrary behavior policies.

| Outcome | Range of policy means | Tolerance | Global TOST p |
| --- | ---: | ---: | ---: |
| affine_log_r2 | 0.997563–0.997565 | ±0.001 | 1.9e-104 |
| near_r2 | 0.991355–0.991385 | ±0.001 | 5.17e-97 |
| far_r2 | 0.997266–0.997270 | ±0.001 | 9.74e-138 |
| shared_goal_error | 0.076237–0.076693 | ±0.005 | 2.7e-20 |

| Confirmatory claim | p | Holm p | Pass? |
| --- | ---: | ---: | --- |
| joint_recovery_and_shared_planning_equivalence | 2.7e-20 | 1.08e-19 | Yes |
| exposure_only_icm_benefit_affine_log_r2 | 8.67e-19 | 2.6e-18 | Yes |
| exposure_only_icm_benefit_folded | 9.31e-10 | 9.31e-10 | Yes |
| exposure_only_icm_benefit_shared_goal_error | 8.67e-19 | 2.6e-18 | Yes |

## Why the ICM exposure control matters

The same freshly trained ICM actor supplies identical native transition pairs to both control encoders. Only the observation pool used by SIGReg changes. Its formula and weight, optimizer budget and initialization stay fixed. Both cells use independent SIGReg sampling; coupled-versus-independent batching is not changed.

| ICM cell | Mean recovery R² | Folds / 60 | Mean shared-data goal error |
| --- | ---: | ---: | ---: |
| native_exposure | 0.6611 | 30 | 0.8739 |
| common_exposure | 0.9975 | 0 | 0.0760 |
| matched | 0.9976 | 0 | 0.0767 |

The three mechanism tests concern directional seed-wise improvement. Mean improvements and unadjusted bootstrap intervals are supplied in the JSON. Common exposure adds broad real observations; this is not a cost-free reweighting of ICM trajectories.

## Significance and usefulness

A significant TOST rejects differences as large as the fixed tolerance; it does not prove exact equality. The prospective cohort strengthens the earlier exploratory finding. The sign-based robustness check concerns typical differences, not means. Similar scores do not prove that learned functions are the same or linearly recover the true latent exactly.

The scientifically useful question is whether exposure-only changes rescue geometry and actual reaching while keeping action transitions fixed. A confirmed effect would distinguish one finite-training failure mechanism from an intrinsic disadvantage of the ICM action rule. The paper already discusses sampling and finite-optimization limitations: this experiment offers a controlled mechanism test in one environment, not a new universal guarantee or an established novelty claim.

## Secondary diagnostics, not additional confirmatory claims

Policy-specific predictor-data equivalence at ±0.005: unadjusted p = 0.952.

Paired fold-risk equivalence at ±5 percentage points: unadjusted p = 0.0461. Rare-failure inference is limited even with 60 seeds; individual-policy upper bounds are pointwise. These diagnostics are outside the four-claim confirmatory family.

## Scope and next question

Starting observations and SIGReg exposure match; successor distributions and stationarity do not. These are physical OU action policies, not Gaussian latent OU dynamics. Shared predictor data retain the inherited endpoint-action omission. Results concern this architecture, data budget and goal distribution.

A next extension would test whether a common frozen representation removes remaining policy-specific planning differences when only predictor action data change. That separates representation quality from action-conditioned transition coverage. It is not claimed to be answered by this cohort.

[Frozen protocol](../PROTOCOL.md) · [Time/code lock](LOCK.json) · [Exact tests and effect sizes](INFERENCE.json) · [Verification](VERIFICATION.json) · [Paper §§7–8](https://arxiv.org/html/2605.26379v1#S7)

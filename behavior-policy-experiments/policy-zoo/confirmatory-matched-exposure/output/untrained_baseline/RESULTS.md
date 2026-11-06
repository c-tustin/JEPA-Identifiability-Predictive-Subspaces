# Does training help beyond the starting geometry?

**Secondary analysis added after training began; no additional confirmatory claims.** The exact starting encoder is frozen, while its predictor gets the same data/budget/goals as the trained-encoder shared-data arm.

Untrained encoder: mean recovery **0.8421**, **26/60** folds, mean physical goal error **0.4479**.

| Trained policy | Recovery gain over initialization | Goal-error reduction over initialization |
| --- | ---: | ---: |
| icm | 0.15543 [0.09346, 0.22716] | 0.37116 [0.20111, 0.56531] |
| random | 0.15543 [0.09347, 0.22716] | 0.37129 [0.20124, 0.56537] |
| count | 0.15543 [0.09347, 0.22716] | 0.37148 [0.20150, 0.56550] |
| rnd | 0.15543 [0.09347, 0.22716] | 0.37135 [0.20123, 0.56550] |
| disagreement | 0.15543 [0.09347, 0.22716] | 0.37161 [0.20162, 0.56595] |
| ou_actions_lp_w0.1 | 0.15543 [0.09347, 0.22716] | 0.37142 [0.20149, 0.56537] |
| ou_actions_hp_w0.3 | 0.15543 [0.09347, 0.22716] | 0.37142 [0.20137, 0.56543] |

Brackets are unadjusted descriptive bootstrap 95% intervals. Positive changes favor training. This comparison distinguishes acquiring useful geometry from preserving a useful initial map; it does not change the primary equivalence conclusion.

[Addendum](../../BASELINE_ADDENDUM.md) · [Exact results](SUMMARY.json) · [Verification](VERIFICATION.json)

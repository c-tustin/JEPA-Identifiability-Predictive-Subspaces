# Actual hypothesis tests for matched-policy performance

**Primary test:** permutation repeated-measures ANOVA, with seed as the block. It tests policy mean differences without choosing an equivalence margin. **Sensitivity check:** Friedman permutation test of policy ranks. Both use 100,000 within-seed label permutations. No goals or grid points are treated as independent replicates.

Null: no systematic policy effect within matched seed blocks. Alternative: some policy differs. Permutation inference assumes within-seed policy-label exchangeability under the null. One Holm correction covers all 12 mean/rank tests across both studies and three endpoints.

| Study | Outcome | Seeds × policies | F statistic | Mean-test permutation p | Mean-test Holm p | Friedman statistic | Rank-test permutation p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_stationary_four | recovery | 10 × 4 | 1.1709 | 0.3422 | 1.0000 | 3.4800 | 0.3270 |
| previous_stationary_four | shared_goal_error | 10 × 4 | 1.0000 | 1.0000 | 1.0000 | 3.0000 | 1.0000 |
| previous_stationary_four | native_goal_error | 10 × 4 | 1.2214 | 0.3178 | 1.0000 | 1.2857 | 0.7526 |
| original_policies_common_reset | recovery | 20 × 7 | 1.6599 | 0.1175 | 1.0000 | 8.8071 | 0.1828 |
| original_policies_common_reset | shared_goal_error | 20 × 7 | 0.9251 | 0.7158 | 1.0000 | 5.5714 | 0.7151 |
| original_policies_common_reset | native_goal_error | 20 × 7 | 0.5758 | 0.7498 | 1.0000 | 3.7480 | 0.7189 |

**Conclusion:** no systematic policy-performance difference is detected in these matched conditions at alpha 0.05. This is not proof that their true performances are exactly equal. It also does not imply equal performance in the original native-policy zoo.

- The earlier study used four modified stationary policies with both endpoint marginals matched.
- The new study uses seven original policies with starting observations and SIGReg exposure matched; successors and stationarity are not forced to match.
- “Native” prediction here means each matched cell’s own policy/source data; for common-reset cells those are reset transitions, not the old native trajectories.
- [Optional equivalence sensitivity](MATCHED_EQUIVALENCE.md) uses illustrative margins and extra assumptions. These margins were chosen after observing results; the primary tests above use no margins.

[Exact statistics, counts and permutation p-values](MATCHED_OMNIBUS.json).

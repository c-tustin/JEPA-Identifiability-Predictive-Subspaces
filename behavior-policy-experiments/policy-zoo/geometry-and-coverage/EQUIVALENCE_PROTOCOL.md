# Requested hypothesis test: are matched policies practically similar?

The requested claim is **practical equivalence**, not merely failure to detect a difference. Paired TOST below addresses that claim. [Permutation difference tests](output/MATCHED_OMNIBUS.md) remain a separate complementary analysis.

This is an additional **post hoc** analysis. It does not replace or relabel the campaign's 54 prespecified tests.

Analyze two studies separately:

1. Previous four-policy stationary matched-coverage study, seeds 100–109. Both endpoint marginals and action counts were matched, but policies included a modified ICM-shaped surrogate. Use its original stationary-weighted recovery R² and shared/native physical goal errors.
2. New seven-original-policy common-reset study, seeds 8–27. Starting observations and SIGReg input multisets match exactly; successors, actions and stationarity need not match. Use common-grid recovery R² and shared/native physical goal errors. Do not use incomplete seed blocks or pool outcomes across the two different designs.

## Difference test

For each study/endpoint, use a seed-blocked Friedman rank statistic with 100,000 within-seed policy-label permutations (fixed RNG seed 20261107). The permutation null assumes exchangeability of policy labels within a seed. Include all ties and use the standard tie correction; a fully tied matrix has statistic zero and p=1. Monte Carlo p=(exceedances+1)/(permutations+1). Holm-correct the six omnibus difference tests (two studies × three endpoints). A nonsignificant difference is not evidence of equivalence.

## Practical mean equivalence

Report illustrative tolerances of **0.001 in recovery R²** (0.1 percentage point) and **0.005 in physical goal error** (2% of the 0.25 action step). These tolerances are chosen after the experiments, so the analysis is exploratory rather than a prespecified equivalence claim. The sensitivity grid also retains the earlier proposed, broader tolerances of 0.005 R² and 0.05 physical goal error. Statistical analysis cannot decide which loss of performance is scientifically acceptable.

For every policy pair, let D be its paired seed-wise performance difference. Perform paired t-based TOST at alpha .05 for the null that the population mean difference is <= -margin OR >= margin. Equivalently, the ordinary 90% paired-t interval must lie strictly within the margins. Paired t inference assumes approximately normal seed differences; n=10 or 20 and discrete goal errors limit this approximation.

For **all policies equivalent** within an endpoint, the null is that at least one policy pair is outside the margin. The alternative is that every pair is inside. This is an intersection-union test: require every pairwise TOST to pass, and take the maximum pairwise TOST p as the global p. No within-global-test Bonferroni correction is needed, since rejecting the global null requires all component nulls to be rejected. Individual 90% intervals are not described as simultaneous confidence intervals, and no separate multiplicity-free claim is made for selected pairs.

The primary joint similarity claim requires both recovery and shared-data reaching to pass (again a conjunction/IUT). Native reaching is reported separately and in an optional stronger all-three-endpoint conjunction, because policy-native action data ask an additional question.

## Assumption sensitivity

Also report distribution-free paired sign-TOST for **typical/median** differences. At each boundary, count strict differences toward the alternative and retain all seeds; ties at the boundary count against equivalence. Its null concerns the paired median/probability, not the mean, so it supplements rather than replaces mean-TOST.

Report the minimum margin exceeding all absolute 90% mean-interval endpoints as a descriptive sensitivity threshold, plus a fixed grid of margins. This does not turn a post hoc minimum into a scientifically accepted margin.

Conclusions must distinguish: detectable but small differences; practical equivalence under the chosen margins and assumptions; insufficient precision; and native-data performance differences. No result implies the original unmodified zoo policies were equivalent under their native visitation distributions.

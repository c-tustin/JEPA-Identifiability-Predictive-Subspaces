# Are the matched policies equivalent?

**Yes for mean recovery and shared-data planning, within the stated tolerances; equivalence is not established for policy-specific-data planning at those tolerances.**

Use paired **TOST** (two one-sided tests). For each pair, test whether the mean seed-wise difference lies strictly between the negative and positive tolerance. Require every policy pair to pass. Seeds are the independent units, not individual goals.

Illustrative tolerances: **±0.001 R²** and **±0.005 physical goal error**. A physical action moves 0.25, so the goal-error tolerance is 2% of one action step. These tolerances were chosen after the experiments: this is exploratory evidence, not a prespecified confirmation.

A tolerance is the largest difference we are willing to treat as practically unimportant. It is not estimated by a hypothesis test. Choosing it after seeing outcomes risks making the test too easy: a wider tolerance makes equivalence easier to establish. For a confirmatory claim, fix the tolerance first and evaluate a fresh, fixed-size seed cohort. Exact equality cannot be established from these finite runs.

| Study | Recovery equivalence p | Shared-data planning equivalence p | Policy-specific-data planning equivalence p |
| --- | ---: | ---: | ---: |
| Four stationary surrogates, 10 seeds | 1.18e-21 | 4.44e-07 | 0.891 |
| Seven original policies, 20 seeds | 1.55e-33 | 9.67e-12 | 0.93 |

For seven policies, the joint claim that **both recovery and shared-data planning are equivalent** has p = 9.67e-12. The sign-based check also passes for typical paired differences (joint p = 9.54e-07); that check concerns typical differences, not the population mean.

The global null is that at least one pair lies outside the tolerance. Its p-value is the largest pairwise TOST p-value: all pairs must pass. This intersection-union test does not need an extra Bonferroni correction for its single all-pairs equivalence claim.

Paired mean-TOST uses a t approximation for seed differences. With 10/20 seeds and discrete planning errors, interpret this with the sign check and tolerance sensitivity. A larger p-value for policy-specific-data planning means insufficient evidence of equivalence, not proof of a difference.

Why not chi-squared? These outcomes are continuous paired scores. A chi-squared test on bins would answer a different question about category frequencies, and a nonsignificant ordinary chi-squared test would still not establish equivalence.

Scope: the seven-policy experiment matches starting observations and SIGReg exposure, not successor distributions or stationarity. Policy-specific predictor data means the reset transitions from each matched cell. These results do not establish equivalence of the original native-policy zoo or exact mathematical identifiability.

[Focused identifiability and fold-risk tests](IDENTIFIABILITY_EQUIVALENCE.md) · [All pairwise intervals and tolerance sensitivity](MATCHED_EQUIVALENCE.md) · [Verification](EQUIVALENCE_VERIFICATION.json) · [Complementary difference tests](MATCHED_OMNIBUS.md)

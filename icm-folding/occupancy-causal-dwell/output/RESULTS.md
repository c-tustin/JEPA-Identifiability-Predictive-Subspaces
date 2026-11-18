# Occupancy intervention with invariant alignment

November 17, 2026. Forty completed 4,000-update fits on ten historical sources. Both losses use the same encoded endpoint tensor. Added stay activations are reused in both endpoint positions; their alignment value and gradient are exactly zero.

| Source-pair fraction | Stay allocation | Final folds in [1,16] | Final folds in source range | Mean mass below 2 | Mean occupancy TV to target |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.5 | native | 5/10 | 5/10 | 0.8848 | 0.00083 |
| 0.5 | balanced | 9/10 | 9/10 | 0.5794 | 0.00117 |
| 0.25 | native | 5/10 | 5/10 | 0.8848 | 0.00093 |
| 0.25 | balanced | 0/10 | 0/10 | 0.4267 | 0.00122 |

Only comparisons between allocations at the same source fraction have an identical alignment function. Changing source fraction changes alignment strength and is a separate stratum. Original ICM source pairs can already contain stays.

All 40 final float32 curves/slopes reproduce exactly after checkpoint reload; final geometry was checked independently on 20,001 float64 log-distance points. Output standard deviations on that grid range from 0.63214 to 6.26462.

Sixty same-state forks at updates 100/200/400 have maximum alignment-gradient difference 0; maximum residual after attributing the combined-gradient contrast to 0.1 times the SIGReg contrast is 4.196167e-05. Forks share weights, optimizer state and the next source batch.

The sampled regularizer occupancy is the manipulated training variable, with an analytically invariant alignment function. Changes in final geometry identify its causal effect in this controlled setting; they need not be caused exclusively by the population objective because finite-batch moments and gradient noise also change. No universal occupancy threshold, fresh-policy validation, real-trajectory stationary-distribution claim or continuous injectivity guarantee.

Normalized backward variation is min(total upward variation,total downward variation)/total variation on the evaluation grid. It is sign/scale invariant, zero for a monotone sampled curve, and at most 0.5. A source-range root means within recorded min/max distances, not proof that every intermediate state was observed.

See [summary](causal_summary.png), [histograms/encoders/slopes](occupancy_encoders_slopes.png), [same-state forks](same_state_forks.png), [full metrics](summary.json), [verification](verification.json), [input/alignment audit](audit.json), and [protocol](../PROTOCOL.md).

# Is code-space asymmetry (skewness) the occupancy feature that causes the fold?

Fixed November 21, 2026, before any of the fits below were trained.

## Hypothesis being tested

Every fit in this study shares one encoder initialization and source-131 alignment pairs; only the endpoint law P changes.
The sensor is a deterministic function of distance, so the encoder is a scalar function f(log d).

1. With its Gaussian frequency weights, `SIGReg1D` is exactly 1024 × MMD²(code law, N(0,1)) for a kernel numerically
   equal to √(2π)·exp(−u²/2), within 0.007 for |u| ≤ 6. Its sine term carries every odd-moment (asymmetry) mismatch of
   the code law; its cosine term carries the even moments.
2. For a positively skewed standardized code law, the exact per-sample SIGReg force changes sign at about z ≈ 1.1 and
   z ≈ 3.45. Codes from the bulk up to z ≈ 1.1 are pushed outward. Codes in 1.1 < z < 3.45 are pulled back toward the
   bulk. Beyond z ≈ 3.45, the force is weakly outward.
3. Early in training the code is an increasing reshaping of the initialization's own response f0(d), with shape
   correlation 0.9993 at update 40. The early code law is therefore the standardized law of f0(d) under P, a property of
   P alone.
4. A fold appears when SIGReg pulls a long, thin far tail back toward the bulk while pushing the junction outward. A
   smooth encoder can't do both monotonically, so the far branch turns back over the near branch.
   - Weak skew: little or no tail is in the pull-back band, so the code stays monotone, apart from tiny far-end turns.
   - Very strong skew: the thin tail is pushed beyond the kernel's reach and ejected, so the code stays monotone.
5. In this geometry the near cluster is nearly a point in f0 (f0 spans 0.04 over d∈[1,2) versus 0.18 over [2,16]). This
   is why near mass and skew were confounded in every earlier study.

The tested feature is the skewness γ of f0(d) under P (standardized third moment), with f0 the
initialization-131 encoder. Retrospectively, across 276 audited earlier fits, the junction fold (backward fraction > 0.1)
has 8 window-rule errors with γ, against 9 with near mass. Those cells were used to form the hypothesis. They are not
counted as confirmation.

## Part A: dissociate skewness from near mass (38 fits)

The atoms are all actual endpoint observations of sources 130 and 136. Within each region, weights are exponentially
tilted in f0:

- near region: w ∝ exp(λn·f0)
- far region: w ∝ exp(λf·f0)

The near mass below distance 2 is fixed at m. λf is solved by bisection to hit a target skewness. If that cannot reach the target, λn is used instead, and if neither alone can, λn = −λf moves both regions toward the junction.
Effective atom count must be at least 200 per region, and each law's full specification is saved.

| m | target γ values |
| ---: | --- |
| 0.25 | −0.8, 0.7 |
| 0.5 | 0.1, 1.1 |
| 0.6 | 0.45, 1.4 |
| 0.75 | 0.0, 1.2, 2.0 |
| 0.875 | 0.3, 1.8, 3.1 |
| 0.9375 | 0.3, 2.15, 3.0, 4.2 |
| 0.96875 | 2.4, 3.0, 5.5 |

Targets were set from the feasible range under the effective-atom constraint, before any training. The lowest
targets at m ≥ 0.75 use the joint junction tilt, which concentrates both regions near distance 2.

Each law is trained with initializations 131 and 133 (alignment plan always source 131) and new uniform plan 9600131.
Training is otherwise identical to the dose-study SIGReg fit: 0.9 alignment + 0.1 SIGReg, 4,000 updates, Adam, and the
original schedule.

Pre-registered rule, using γ measured in each initializer's own f0:

| γ (own f0) | Prediction |
| --- | --- |
| 0.75 ≤ γ ≤ 2.6 | junction fold (backward fraction > 0.1) |
| γ ≤ 0.45 or γ ≥ 3.8 | no junction fold |
| otherwise | no prediction (boundary zone) |

Decisive cells:

- m = 0.5 with γ ≈ 1.1 must fold. No 50%-mass law has folded before.
- m = 0.9375 with γ ≈ 2.15 and m = 0.96875 with γ ≈ 2.4 must fold. No law at those masses has folded before.
- m = 0.75 with γ ≈ 0 and m = 0.875 with γ ≈ 0.3 must not fold. Every earlier law at m = 0.75 folded.
- Mass-only accounts predict the opposite for all of these cells.

## Part B: which part of the objective responds to asymmetry (30 fits, initialization 131)

Five laws are used: the native n130/f130 templates at m = 0.5, 0.75, 0.9375, and the Part A laws (0.5, γ 1.1) and
(0.9375, γ 2.15).

| Objective | Prediction |
| --- | --- |
| B1. 0.9 alignment + 0.1 × even-only SIGReg (sine term removed) | no junction fold in any law |
| B2. Dose-study variance control + 1 × skew² of the standardized batch code | no prediction |
| B3. Dose-study variance control + 10 × skew² of the standardized batch code | junction fold at the native m = 0.75 template |
| B4. Raw/own-code InfoNCE + κ·skew², κ ∈ {1, 10} | junction fold at the native m = 0.75 template for κ = 10; no prediction for κ = 1 |
| B5. Plain variance control and plain raw InfoNCE on the two Part A laws | no junction fold |

B1's even-only SIGReg still has the variance-fixing even moments, so it isolates the asymmetry pathway. In B4,
InfoNCE's critic and normalization are the dose-study ones, and κ·skew² is computed on the same 256 endpoint codes.

## Reporting

- Every cell is reported with backward fraction, roots, tiny-turn status (0 < backward ≤ 0.1) and junction-fold status.
- Failures of any prediction are retained and reported as falsifications.
- Final geometry comes from an independent dense reload at 20,001 points.
- Predictions are written to `output/skew/PREDICTIONS.json` and hashed before training.
- These are finite-training empirical checks with a common encoder class, not analytic guarantees.
- γ uses this encoder's initial input geometry. It is not claimed as a coordinate-free property of distance.

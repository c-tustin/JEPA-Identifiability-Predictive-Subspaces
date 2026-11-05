# What could explain ICM's folds?

## The new hypothesis

**The original ICM observation distribution may be sufficient to create folds under our existing finite network and training procedure. Transition alignment may change their severity without being necessary for their existence.**

This is a testable hypothesis, not a conclusion from the matched-coverage result. That result used corrected stationary policies; correcting ICM changed its behavior. It cannot establish that the original transition law is irrelevant.

Our new test uses the exact original ICM, random, count, low-pass OU-action and RND datasets. It keeps every training endpoint, minibatch, initialization and SIGReg setting fixed and removes alignment. This is a mechanism intervention, not a new behavior policy or regularizer-design study.

- **ICM folds persist:** alignment is not necessary for those intervention fits. Observation sampling, encoder bias and optimization remain candidate mechanisms.
- **ICM folds disappear:** the original alignment interactions matter.
- **Removing alignment creates folds:** alignment was protective in those runs.
- **ICM differs from random/count after removal:** the original sampled observation sets still matter even without transition coupling. This does not identify which observation statistic causes the difference.

These outcomes can coexist across seeds; we report every run.

## A useful exact restriction on the explanation

Our encoder loss has the form

$$
L(h)=0.9\,\frac1B\sum_{i=1}^B
\left[h(x_i)-h(x_i')\right]^2
+0.1\,\operatorname{SIGReg}
\left(\{h(x_i),h(x_i')\}_{i=1}^B\right).
$$

Swapping either direction of a pair does not change its squared difference or the pooled endpoints. Therefore **this loss cannot distinguish a fixed pair dataset from its time reversal**. This invariance holds for every encoder; exact-arithmetic gradients also match. Numerical checks verify both loss and gradients.

Consequently, an explanation in terms of directed approach bias alone is incomplete. The bias can change which observations and undirected connections get sampled, but the encoder does not receive the direction, curiosity reward or action label. The later action-conditioned predictor is different: it does receive actions.

When alignment is removed, within-batch endpoint pairing also becomes irrelevant. The remaining loss depends only on the endpoint multiset. This is why the intervention tests necessity of the explicit alignment term while retaining the original sampled endpoints. Paired endpoints still co-occur within each finite minibatch, so their correlations can affect the variance of the regularizer estimate. Removing alignment does not remove every effect of the pair law. [Interpretation clarification](CLARIFICATIONS.md).

## Identifiability has several levels

For our scalar code, the paper's orthogonal ambiguity reduces to a sign after normalization. Finite affine R² measures how nearly the code is a straight-line function of log-distance; it does not prove the theorem's guarantee.

A monotone nonlinear code can preserve all distance information with mediocre affine R². Rank correlation helps distinguish that warping from loss of order. A fold can create two distinct distances with the same code. We find numerical witnesses using branch crossings and refine them on the actual network, rather than treating low R² as proof of information loss.

If exactly equal goal codes correspond to distances $g_1$ and $g_2$, a deterministic controller that only sees the code chooses the same trajectory for both goals from the same start. If it ends at $D$,

$$
\frac{|D-g_1|+|D-g_2|}{2}
\geq \frac{|g_1-g_2|}{2}.
$$

This is a task-level consequence of exact aliasing. Our numerical witnesses are not formal proofs of equality in deployed floating-point planning. They give refined evidence of the ambiguity; actual goal-reaching is evaluated separately.

## Why the stationary surrogate was too narrow

For a stationary nearest-neighbor chain on a finite line, there is no sustained probability current through an endpoint. Stationarity fixes the forward/backward edge ratio once the marginal is fixed. The corrected ICM surrogate therefore retained crossing strengths but changed important original biases.

Its null result motivates returning to actual policies. It does not show that all policies are interchangeable, that occupancy alone causes folding, or that the original zoo effects disappear under a valid intervention on the original datasets.

## What the completed tests now show

All 100 alignment-removal fits retain their original fold status. Earlier collection with the same frozen ICM actors then repairs all ten old ICM folds, with no new folds across 20 seeds; mean affine recovery rises from 0.6035 to 0.9457 and shared-data goal error falls from 0.8316 to 0.0516 across ten fixed seeds. Random changes little; count loses affine recovery. See [collection results](output/EARLY_COLLECTION_RESULTS.md).

This motivates a more specific hypothesis: late ICM collection concentrates the training sample, and finite-network generalization outside the dominant region produces the fold. The intervention supports a role for collection phase; it changes both occupancy and pair statistics, so a purely marginal causal explanation remains unproven. It also does not show that folds are necessary to Gaussianize a non-Gaussian distribution: monotone transformations can do that.

## What would come next

First replicate early/late collection with fresh ICM actors and seeds. Then test intermediate collection ages with fixed actors and equal budgets to locate the onset of concentration, folds and planning failure. Keep random and count controls. Separately, independently sampled pooled-endpoint controls could distinguish residual finite-batch endpoint correlations from the marginal sample distribution. These are proposed follow-ups, not completed results.

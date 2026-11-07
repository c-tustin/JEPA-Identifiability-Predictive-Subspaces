# What nonlinear planning adds to the identification question

## Prediction error needs a coordinate scale

Let a scalar encoder $h$ satisfy, on the relevant interval,

$$
|h(d)-h(g)|\ge m|d-g|,\qquad m>0.
$$

Then physical goal error is bounded by latent goal error divided by $m$. A strongly compressed interval has a small $m$; a folded encoder has no such global positive lower bound. No-reversal checks on a finite grid do not certify this condition.

For a deterministic transition, suppose the true encoded map $F$ is $K$-Lipschitz in its state input and a fitted map $\widehat F$ has uniform error at most $\varepsilon$ for the states/actions visited by either rollout. Starting from the same state and applying the same length-$H$ action sequence gives

$$
|y_H-\widehat y_H|\le\varepsilon\sum_{j=0}^{H-1}K^j.
$$

Therefore

$$
\boxed{
|d_H-g|\le\frac{|\widehat y_H-h(g)|+\varepsilon\sum_{j=0}^{H-1}K^j}{m}.
}
$$

This is a conditional deterministic bound, not a guarantee for the present learned models. We do not certify $m$, uniform prediction error, or coverage of every planned state/action. A dataset-averaged MSE cannot be substituted for $\varepsilon$. Stochastic execution adds noise terms. The bound explains why latent accuracy alone is insufficient and identifies quantities a stronger theorem would need to control.

## The experiment exposes that distinction

For the persistent-action encoder, with a shared predictor dataset:

| SIGReg data | Shared held-out latent MSE | Closest physical error, noisy execution | Hit rate within 0.1 |
| --- | ---: | ---: | ---: |
| Raw | 0.001408 | 0.0346 | 97.4% |
| Balanced | **0.001152** | **0.1930** | **60.9%** |

The balanced model has lower MSE on its own normalized coordinate but poorer goal reaching. Its compressed coordinate makes that comparison unsuitable as a physical accuracy ranking. This is consistent with the bound's role for inverse sensitivity; it does not measure or prove which factor caused every failure.

Using native predictor data improves the raw persistent encoder's closest error to 0.0204 and hit rate to 100% on the tested goals, despite a worse global shared-test MSE of 0.002884. Thus broader uniform training is not automatically better for a particular local goal distribution and a fixed-capacity predictor. This differs from the earlier ICM-native failure and does not invalidate it: both the world and goal distribution changed.

## Interpret the references carefully

The known-dynamics nominal MPC uses the exact deterministic Euler transition, the same candidate bank and the same cost formula. Its cost is physical-distance based; learned controllers use their own latent coordinate. The bank is finite and the cost emphasizes terminal distance rather than the first-hit metric. Learned controllers can therefore outperform this reference on hit rate without outperforming an optimal physical controller. Stochastic noise adds another distinction. The saved two-action witnesses establish deterministic reachability separately from MPC performance.

There are only three representation/data seeds. The second predictor fit and 32 goals per seed test sensitivity within each seed; they do not create additional independent representation replications. The 100% hit rate is a result on these local tasks, not a general control guarantee.

[Full planning table](output-control/RESULTS.md) · [Coverage control](output-coverage/RESULTS.md) · [Why the drift theorem needs shorter-time targets](DRIFT_ESTIMATION.md)

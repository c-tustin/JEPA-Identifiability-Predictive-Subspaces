# A stronger Predictive Closure test within one distance world

**Status:** an idealized continuous-variable derivation. Completed numerical checks are reported in [RESULTS.md](RESULTS.md); the learned neural encoders did not show feature-selection separation. This strengthens the cross-world comparison by holding the transition law fixed.

## One world, two encoders

Let the only physical coordinate be distance, rendered by the same triangle:

$$
D=4\exp(0.7Z),\qquad X=V/D.
$$

Use the stationary random-sign transition

$$
Z'=SrZ+\sqrt{1-r^2}\epsilon,
\qquad Z,\epsilon\sim\mathcal N(0,1),\quad S\sim\operatorname{Uniform}\{-1,+1\},
$$

with independent variables and $0<r<1$. The sign is an independently chosen input, not a persistent extra world coordinate. It is not supplied to the predictor in the comparison below. This is a synthetic positive-distance transition.

Two candidate scalar encoders both have standard Gaussian output marginals:

$$
h_{\rm full}(Z)=Z,
\qquad
h_{\rm mag}(Z)=\Phi^{-1}\!\left(2\Phi(|Z|)-1\right).
$$

The second expression Gaussianizes the half-normal magnitude. It is defined almost surely; its value diverges at $Z=0$, a zero-probability point. Numerical implementations must clip quantiles and report the resulting approximation. The full encoder preserves physical distance. The magnitude encoder folds $Z$ and $-Z$ together, confusing the two physical distances $d$ and $16/d$.

## Equal predictive information

Write $M=|Z|$, $M'=|Z'|$, and let $B,B'$ be the signs. The conditional law of $Z'$ is invariant to changing the sign of either $Z$ or $Z'$. Consequently, $(B,B')$ are independent uniform signs, independent of $(M,M')$. Thus

$$
I(Z;Z')=I((M,B);(M',B'))=I(M;M').
$$

Both encoders are invertible functions of their respective arguments, so

$$
I(h_{\rm full}(Z);h_{\rm full}(Z'))
=I(h_{\rm mag}(Z);h_{\rm mag}(Z')).
$$

The full encoder already retains all state information, so data processing makes its predictive information maximal over deterministic observation encoders. The folded magnitude encoder attains the same value: the discarded sign is irrelevant to this uncontrolled transition distribution. That sign can still matter for choosing a physical goal.

## Different optimized mean-prediction errors

For the full encoder, symmetry gives $E[Z'\mid Z]=0$, so optimal squared prediction error is exactly 1.

For the magnitude encoder, the next magnitude distribution depends nontrivially on the current magnitude. In fact, for $t>0$ and $m\ge0$, writing $s=\sqrt{1-r^2}$,

$$
P(M'\le t\mid M=m)
=\Phi((t-rm)/s)-\Phi((-t-rm)/s).
$$

Its derivative in $m$ is negative for $m,t>0$ because the standard Gaussian density at $(t-rm)/s$ exceeds that at $(t+rm)/s$. The conditional magnitude distribution therefore shifts strictly upward with $m$. Since Gaussianization is strictly increasing, the conditional mean of $h_{\rm mag}(Z')$ is nonconstant. All relevant second moments exist because the output marginal is Gaussian.

The total-variance identity then gives

$$
\min_g E[(h_{\rm mag}(Z')-g(h_{\rm mag}(Z)))^2]
=1-\operatorname{Var}\!\left(E[h_{\rm mag}(Z')\mid |Z|]\right)<1.
$$

Thus squared-error prediction strictly favors a folded candidate over the fully informative coordinate, while predictive information ties them. This is stronger than comparing two transition laws: it shows different preferences among encoders in one fixed world.

## Scope: this does not refute the paper's theorem

[Predictive Closure](https://arxiv.org/abs/2610.27364v1), Appendix A.1–A.2, states conditional Gaussian, shared-marginal, fixed-precision and capacity assumptions for its squared-error/information equivalence. The random-sign transition violates its conditional Gaussian assumption without the sign input. The construction here is continuous, assumes exact Gaussianization and ideal predictors, and does not establish its finite-precision capacity or neural representability conditions. It diagnoses what changes when those assumptions are relaxed.

At finite precision, the two quantizers can allocate different resolution to magnitude, so exact information equality must be tested rather than asserted. Under a finite neural model and approximate SIGReg, neither Gaussianity nor the proposed encoder preference is automatic.

## Numerical validation checklist

1. Verify the equal-information identity analytically and with Gaussian-mixture log densities; avoid unstable generic mutual-information estimators.
2. Compare the two explicit encoders using independently sampled train/calibration/test data. Fit mean predictors and report held-out error. Apply the same clipping and precision policy to both.
3. Sweep quantization/clipping to expose numerical effects. For magnitude-only mapping, report ambiguity of the physical distance readout, not just latent prediction.
4. Train the original flexible scalar encoder with mean-only, action-conditioned and distributional prediction. Report actual head/trunk parameter counts and update budgets; additional distributional output parameters must not be hidden as an exactly matched model.
5. Check whether conditioning on the random sign preserves useful physical-distance information. Do not assume that this sign-conditioned synthetic transition establishes performance for the original physical action system.

This checklist describes the study protocol. Completed checks and their limits are recorded in [RESULTS.md](RESULTS.md) and [RUN_STATUS.md](RUN_STATUS.md); the derivation alone is not experimental evidence.

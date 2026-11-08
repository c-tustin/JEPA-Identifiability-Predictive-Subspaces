# Why the nonlinear theorem needs a different training target

The [identification lemma](PROOF.md) concerns **instantaneous conditional drift**. Our planner predicts a whole one-second transition. Those are different objects, even if the learned coordinate is exactly physical distance.

Consider the interior diffusion, holding action $a$ fixed for an interval $\delta$:

$$
dD_t=[b(D_t)+ca]dt+\sigma(D_t)dW_t.
$$

Assume enough smoothness and bounded generator derivatives for the following small-time expansion, uniformly on an interior region. Boundary effects must be treated separately. With $v=\sigma^2$ and generator

$$
L_af=(b+ca)f'+\tfrac12vf'',
$$

the conditional mean of physical distance satisfies

$$
E[D_\delta\mid d,a]
=d+\delta[b(d)+ca]
+\frac{\delta^2}{2}\left[(b(d)+ca)b'(d)+\tfrac12v(d)b''(d)\right]
+O(\delta^3).
$$

Subtract the two conditional means for $a=+1$ and $a=-1$:

$$
\boxed{
\frac{E[D_\delta\mid d,+1]-E[D_\delta\mid d,-1]}{2\delta}
=c+\frac{\delta}{2}cb'(d)+O(\delta^2).
}
$$

**Even physical distance has a state-dependent finite-time action effect.** For our drift, $b'(d)=0.3\cos d$. At a one-second interval, imposing a constant action coefficient in an ordinary next-state predictor is not the exact assumption used in the theorem. The expansion is local in time; its leading term is not an accuracy guarantee at one second.

## Independent noise creates a real sampling cost

At a fixed source state, collect $N$ independent endpoints for each action. Let $\overline D_+$ and $\overline D_-$ be the two sample means. The estimated drift contrast is

$$
\widehat C_\delta(d)=\frac{\overline D_+-\overline D_-}{2\delta}.
$$

For small $\delta$ and independent noise across both actions,

$$
\operatorname{Var}(\widehat C_\delta(d))
=\frac{\sigma(d)^2}{2N\delta}+O(N^{-1}).
$$

The leading squared bias is of order $\delta^2$, while variance grows as $1/(N\delta)$. Balancing those terms suggests $\delta$ of order $N^{-1/3}$ and pointwise MSE of order $N^{-2/3}$ when the leading bias is nonzero. These rates apply to this local contrast estimator under reset access, not to a neural encoder, a temporally dependent exploration dataset, or the full representation-identification problem.

The earlier shared-noise prototype largely bypassed this cost. Its near-perfect recovery is therefore not evidence that independent trajectory samples suffice.

## Numerical check

[Code](check_drift_sampling.py) checks the finite-time action-effect bias with a high-accuracy deterministic integrator and measures contrast variance using **independent noisy endpoints**, 64 microsteps per interval. The deterministic check isolates the same leading bias term; it does not estimate the full stochastic conditional mean. The noisy check uses the smooth interior coefficient extension without reflecting boundaries, four source states, 256 independent groups and 128 endpoints per action per group. It is a simulator diagnostic, not additional representation training.

[Results](output-drift/RESULTS.md) retain empirical variance relative to the leading formula at intervals 0.02, 0.05, 0.1 and 0.2. Finite-step bias, Monte Carlo error and discretization remain separate from the proved instantaneous-drift lemma.

The next theorem-oriented experiment should use independently noisy action samples at several observation intervals, check stability as the interval shrinks, and train an action-structured drift model while explicitly preserving information. Forcing a constant coefficient on the existing one-second predictor would test an incompatible model assumption.

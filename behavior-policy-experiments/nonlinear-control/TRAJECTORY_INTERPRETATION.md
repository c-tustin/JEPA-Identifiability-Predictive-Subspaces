# What would nonlinear recovery mean?

The original Gaussian OU experiment singles out an affine version of its Gaussian latent under the paper's assumptions. Changing the world removes that reference point. In particular, neither physical distance nor log-distance is automatically identified by passive alignment and SIGReg.

For a smooth change of coordinate $Y=T(D)$, the interior dynamics become

$$
dY_t=\left[T'(D_t)(b(D_t)+cA_t)+\tfrac12T''(D_t)\sigma(D_t)^2\right]dt
       +T'(D_t)\sigma(D_t)dW_t.
$$

An unrestricted nonlinear dynamics model can describe this transformed world too. Predicting well therefore does not by itself select physical distance. This is a coordinate ambiguity; an action coefficient constrained to be constant removes it under the assumptions in [the lemma](PROOF.md).

The Gaussian regularizer adds a second issue. If a scalar encoder is increasing and its output is exactly standard Gaussian under a continuous distance distribution with CDF $F_\nu$, it must satisfy

$$
T(d)=\Phi^{-1}(F_\nu(d)).
$$

This follows by equating $P(T(D)\le T(d))$ with $F_\nu(d)$. A decreasing encoder gives the opposite sign. The formula concerns interior points; Gaussian quantiles diverge at the endpoints of bounded support. A finite neural encoder and finite-frequency SIGReg only approximate this condition.

Thus a policy-dependent sampling distribution can favor a policy-dependent nonlinear coordinate even without folding. Source-bin balancing changes the distribution used by SIGReg; it does not make the target physical distance or guarantee uniform endpoint coverage.

Read the new trajectory results as follows:

- High distance or log-distance $R^2$ measures **affine recovery of that specific target** on a common grid.
- Low affine recovery with no reversals can be a smooth coordinate warp. It is not sufficient evidence of lost information.
- A grid reversal is evidence of a folded mapping at the evaluated resolution. Distinct distances can then share an embedding, which matters for control. A finite grid cannot certify global injectivity.
- Compare nonlinear and zero drift within the same policy, sampler, seed, observation interval, and noise model. These are finite-training interventions, not proofs that drift alone determines the outcome.

The sinusoidal drift has attracting regions. After burn-in, the trajectories can leave many sensor bins unvisited even though initial distances were uniform. Consequently this comparison changes both transition structure and coverage. A follow-up with matched source states is needed to separate them. High grid recovery across an unvisited interval is encoder generalization, not direct evidence that the collected data identify that interval.

The trajectory study does not yet train an action-conditioned predictor. Next, compare the same observations with a predictor constrained to constant action effect, estimate drift from independent action samples rather than shared-noise pairs, and measure physical goal error. Those steps would test whether the proved mechanism survives realistic sampling.

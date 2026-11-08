# Why changing the observed distribution can change geometry

## The target distribution allocates latent space

Write $q=\log d$ and let the regularizer's input distribution have continuous density $p_R(q)$ and CDF $F_R(q)$. Suppose, only for this argument, that an increasing code $h$ transforms that distribution into an exact standard Gaussian. Then

$$
\Phi(h(q))=F_R(q),
\qquad
h(q)=\Phi^{-1}(F_R(q)).
$$

Differentiating where densities are positive gives

$$
h'(q)=\frac{p_R(q)}{\varphi(h(q))},
$$

where $\Phi$ and $\varphi$ are the standard Gaussian CDF and density. Thus even a **monotone** code changes its spacing when the sampled distribution changes. A densely sampled physical interval can occupy a large fraction of code space; sparsely sampled intervals can be compressed.

This does **not** require a fold. The monotone quantile map is a counterexample to the claim that non-Gaussian observations inherently force folding. The question is why our finite, unconstrained network and optimizer choose a folded map instead of an order-preserving one. Our finite SIGReg, alignment term and architecture also mean the quantile map is a diagnostic comparator, not the exact trained optimum. Empirical plots use rank interpolation with clipped unobserved tails, not this continuous-density identity as a literal finite-data theorem.

## Separate which data each loss sees

The new factorial uses

$$
L_{a,r}(\theta)=0.9\,\mathbb E_{(x,x')\sim D_a}
\left[(h_\theta(x)-h_\theta(x'))^2\right]
+0.1\,\mathbb E\left[\mathrm{SIGReg}_B
(h_\theta(X_1),\ldots,h_\theta(X_B))\right],
$$

with $X_i$ sampled independently from the pooled endpoint measure of $D_r$, and $B=1024$. This isolates the regularizer's **marginal exposure** in this intervention. The original loss sampled correlated pairs, so we also compare both matching-source cells with the original coupled continuations.

For fixed encoder and empirical pool, let $\phi(t)=\mathbb E[e^{it h(X)}]$, $\phi_0(t)=e^{-t^2/2}$ and integration weights $w_t$. The exact expected implemented regularizer for iid observations is

$$
\mathbb E[\mathrm{SIGReg}_B]
=\sum_t w_t\left[B|\phi(t)-\phi_0(t)|^2+1-|\phi(t)|^2\right].
$$

We calculate this on the full fixed pools, rather than interpreting a noisy minibatch score as the population loss. Paired observations have a different finite-batch covariance term; the iid formula is not applied silently to the old coupled objective.

For the original sampler with $B/2$ correlated pairs, define

$$
\psi(t)=\mathbb E[\cos(t(h(X)-h(X'))) ].
$$

Its expected finite-batch penalty differs from the iid expression by

$$
\sum_t w_t\left[\psi(t)-|\phi(t)|^2\right].
$$

This is why identical pooled marginals need not give identical finite-batch training when endpoints co-occur. Sharing batches between alignment and SIGReg also changes gradient-noise correlations. Our original-versus-independent comparisons are sampling-protocol comparisons, not proof that this one covariance term alone explains every difference. [Earlier exact paired calculation](../mechanism-tests/MATH.md).

## Nearby examples can move distant slopes

Let $J_\theta(q)=\partial h_\theta(q)/\partial\theta$. For a small actual parameter update,

$$
\Delta h(q)\approx J_\theta(q)\,\Delta\theta,
\qquad
\Delta h'(q)\approx \partial_qJ_\theta(q)\,\Delta\theta.
$$

All distances use the same weights. A gradient computed from nearby observations can therefore change far-distance slopes even when no far observations were sampled. This is a statement about parameter sharing; its sign and magnitude must be measured, not assumed. A declining far slope can become negative, introducing a fold.

We split Adam's first moment into carried early-training history, post-switch alignment history and post-switch regularizer history. With a common actual second-moment denominator $v$ and step $t$,

$$
\Delta\theta_k=-\eta\,
\frac{m_k/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\epsilon},
\qquad
k\in\{\mathrm{carry},\mathrm{alignment},\mathrm{regularizer}\}.
$$

These parameter directions add to the actual update up to floating-point error. They use the **same combined denominator**; they are not three separately trained Adam optimizers. Functional JVPs predict their local curve/slope changes; actual component-only parameter interventions and full training replays check them. Removing a component at one update is a local counterfactual, not a guarantee about training with that loss removed for thousands of updates.

The source-region diagnostic uses the exact chain rule

$$
\nabla_\theta R=\sum_b\sum_{i\in b}
\left(\frac{\partial R}{\partial y_i}\right)\nabla_\theta h_\theta(x_i).
$$

The coefficients $\partial R/\partial y_i$ are computed from the **entire actual batch**, then grouped by source distance. Recomputing SIGReg independently within bins would change the objective and is not what we do. Each current-gradient contribution is projected through the same actual Adam denominator and network Jacobian to measure its effect on slopes at other distances. This regional diagnostic allocates the current impulse, not all accumulated optimizer history. The five preselected initial-state comparisons include both later-folding and nonfolding actors.

## What is matched in the zoo control

Every policy receives the same starting observation pool $x_i$, and every regularizer update receives the same indexed multiset of those observations. Policies still choose their original actions, producing actual successors $x_i'$. Consequently

$$
p(x)\ \text{and regularizer exposure match, while}\quad
p_\pi(x'\mid x)\ \text{can differ}.
$$

The successor marginal and stationarity are not forced to match. Requiring a common stationary marginal while retaining arbitrary original nearest-neighbor action probabilities would generally be inconsistent. The new reset intervention avoids correcting/symmetrizing ICM's action rule, but it changes collection and supplies broad real observations beyond native gaps. It does not isolate coverage merely by renaming a different policy.

# Why equivalent recovery scores need not mean equivalent planning

This is an explanatory lemma for the existing one-distance world, not a new novelty claim or a certified numerical bound for the trained networks.

Let distance be d in [1,16], and let h(d) be the scalar learned representation. If h is differentiable and monotone, with

$$
0<m\leq |h'(d)|\leq M,
$$

then it has an inverse whose sensitivity is at most 1/m. A latent goal error e can correspond to a physical-distance error as large as e/m. Thus a nearly flat portion of h can make accurate physical planning require much more accurate latent predictions.

For an action a, define the true encoded transition

$$
G_a(y)=h\!\left(T_a(h^{-1}(y))\right),
$$

where T is the existing reflected physical move. Suppose the learned predictor has uniform error at most epsilon, and the true encoded transitions are L-Lipschitz over the states being compared. For the **same executed H-action sequence**, recursive prediction gives

$$
|y_H-\hat y_H|\leq\epsilon\sum_{j=0}^{H-1}L^j.
$$

The triangle inequality and the inverse bound then give

$$
|d_H-d_{\mathrm{goal}}|
\leq\frac{|\hat y_H-h(d_{\mathrm{goal}})|+
\epsilon\sum_{j=0}^{H-1}L^j}{m}.
$$

Proof: each step satisfies e_(t+1) <= L e_t + epsilon, starting at e_0=0. Sum the recurrence, add predicted terminal goal error, then apply the inverse sensitivity bound. Predictions must remain in the domain on which the uniform bounds hold. This is a sequence-level statement, not a convergence theorem for receding-horizon replanning.

## What behavior can change

- **Geometry:** policy-dependent exposure can change the smallest slope m or create folds, invalidating a single-valued inverse.
- **Transition learning:** data may cover distances yet omit important actions at those distances. That does not control the uniform predictor error epsilon.
- **Error propagation:** even an injective nonlinear coordinate can have a poorly conditioned encoded transition. Reflected moves are nonexpansive in physical distance, but the coarse encoded Lipschitz bound M/m can be large.

R² averages linear-recovery error. It does not bound the smallest slope, rule out a narrow fold, or bound action-conditioned prediction error. Testing equivalence of R² therefore addresses only one part of this chain.

None of m, L or uniform epsilon has been certified for the saved networks. Dense-grid derivatives and held-out errors can diagnose them but are not global certificates. A stronger result would establish or bound these quantities and show that intervening on the relevant state/action data changes actual reaching as predicted.

If two distinct goals share exactly the same encoding and the planner receives only that encoded goal, they are indistinguishable to it. From the same start and randomness the terminal distance is the same, so their average absolute goal error is at least half their separation. This simple ambiguity bound is stronger than a low R² alone as a concrete failure witness; it is already used by the earlier alias diagnostic.

# What can and cannot determine a fold?

There are three different questions: **which code has low loss, which code optimization reaches, and where a particular finite run ends.** They are not interchangeable.

## Data define a loss; they do not specify the optimizer's answer

For a policy's distribution of observation pairs $P(x,x')$, the implemented objective is the expected finite-batch quantity

$$
J_P(\theta)=0.9\,\mathbb E_P[(h_\theta(x)-h_\theta(x'))^2]
+0.1\,\mathbb E_{P^B}[R(h_\theta(X_1),h_\theta(X'_1),\ldots)].
$$

Different policies change **both the visited views and how they are paired**. Coverage describes only part of the first ingredient. The network is trained on shuffled pairs, so it does not read a policy's entire action history; persistence changes the collected pair distribution and the dependence among stored samples.

Given the actual finite data, initial weights, Adam state, learning rates and batch indices, the training algorithm defines the final weights exactly by iteration. The data distribution and initialization *alone* are insufficient inputs when those other choices are unspecified. Even when everything is specified, an exact replay is different from an analytic theory predicting roots without simulating the trajectory.

For a simple turn, with $q=\log d$ and $s=\partial h/\partial q$,

$$
\Delta q_*\approx-\frac{\nabla_\theta s(q_*,\theta)^\mathsf T\Delta\theta}
{\partial s(q_*,\theta)/\partial q}.
$$

This predicts a **local** change using the current weights and actual update. Each update changes both numerator and denominator. Roots can appear, disappear or become degenerate, so simply summing one initial local prediction cannot determine the final root. Turning the learning rate to zero freezes the root without requiring a stationary objective; [the continuation experiment](../../qualitative-study/fold-location/training-mechanism/stopping-point/README.md) verifies this directly.

The separate dataset/initialization matrices test whether failures are mainly associated with one dataset, one initial function, or their interaction. They do not establish that a nonconvex network has reached any global optimum.

## A framework beyond Gaussian OU

For a stationary policy-induced transition operator $P$ and visitation distribution $\pi$, a centered, unit-variance scalar code satisfies

$$
\mathbb E[(h(S')-h(S))^2]
=2\langle h,(I-P)h\rangle_\pi.
$$

For a real-valued quadratic form, only the self-adjoint part $(P+P^*)/2$ contributes. Under a variance constraint, minimizing alignment selects slowly changing functions of this operator; it does not automatically select physical distance. Gaussian OU is a tractable special case with Hermite eigenvalues $\rho^k$. A policy changes both $\pi$ (which outputs SIGReg weights) and $P$ (which distinctions alignment encourages the code to erase). A histogram estimates the first and cannot determine the second.

This gives a principled direction for a nonlinear-dynamics extension: state conditions on the policy's transition operator that make the physical coordinate the unique admissible slow code, then bound errors from finite sampling and restricted encoders. Exact Gaussian-output constraints impose a smaller nonlinear admissible set than centering/variance alone; a spectral variance argument must not silently replace that constraint. For finite stored pairs, a symmetrized edge measure gives an analogous graph-Dirichlet identity, but properties of that empirical graph alone do not determine the smooth neural encoder learned between sampled observations.

## A Gaussian example where a good readout is not identification

Suppose $(Z,Z')$ is a stationary standard-Gaussian pair with correlation $\rho$, and $h(Z)$ has mean zero and variance one. Expand $h$ in orthonormal Hermite functions with coefficients $c_k$. The identity

$$
\mathbb E[h(Z)h(Z')]=\sum_{k\ge1}c_k^2\rho^k
$$

follows by expanding the Gaussian generating-function identity $\mathbb E[e^{tZ-t^2/2}e^{uZ'-u^2/2}]=e^{\rho tu}$. Alignment is therefore $2(1-\sum_k c_k^2\rho^k)$.

For $0<\rho<1$, the largest nonconstant eigenvalue is $\rho$, belonging to the linear code. This statement assumes the population distribution and exact centering/variance conditions. It is not a theorem about the soft finite-batch SIGReg loss or finite optimization.

For $\rho<0$, the linear code has *negative* cross-correlation. An even code can have positive cross-correlation and lower alignment. The normalized second Hermite function is a variance-constrained optimum, but it is **not Gaussian distributed**, so it is not automatically feasible under an exact Gaussian-output constraint.

A feasible Gaussian-output counterexample is

$$
h_F(z)=\Phi^{-1}\!\left(2\Phi(|z|)-1\right).
$$

It is even, maps a standard Gaussian to a standard Gaussian, and loses the sign of $z$. Its Hermite expansion contains only even orders, so its cross-correlation is positive for nonzero negative $\rho$. Thus it has lower population alignment than the linear Gaussian code despite losing information. This is a **comparator**, not a claim that it is the optimal exact-Gaussian code or is exactly representable by our fixed finite tanh network.

## Finite paired batches also affect SIGReg

Let $N=2B$ endpoints come from $B$ independent stationary pairs. Define $\varphi(t)=\mathbb E[e^{itY}]$, $\psi(t)=\mathbb E[\cos(t(Y-Y'))]$ and the Gaussian target $T(t)=e^{-t^2/2}$. Expanding the variance of the empirical characteristic function gives

$$
\mathbb E[R]=\sum_t w_t\left[
N|\varphi(t)-T(t)|^2+1+\psi(t)-2|\varphi(t)|^2
\right].
$$

The two endpoints of a pair are correlated: treating all $N$ outputs as independent would drop a covariance term. Even if the population output marginal is exactly Gaussian, the implemented batch-size-scaled loss has a nonzero finite-batch expectation that depends on pairing.

The same expression holds exactly for sampling pairs with replacement from a finite stored pool, even with unequal source/successor marginals: define $\varphi$ as the **pooled endpoint** characteristic function. This follows from $\operatorname{Var}(A+A')=2+2\psi-4|\varphi|^2$ for $A=e^{itY}$. `pair_objective.py` checks that identity against actual sampled minibatches and uses it to evaluate oracle candidate codes without training a network.

For $Y=aZ$, $\varphi=e^{-a^2t^2/2}$ and $\psi=e^{-a^2t^2(1-\rho)}$. This gives an analytic expected loss for any rescaled linear code. `loss_theory.py` checks the formula against 512 independently sampled batches at each of four settings, then compares its numerical minimum against a Monte Carlo integral for $h_F$:

| Correlation | Best rescaled linear expected loss | Gaussianized fold expected loss |
| ---: | ---: | ---: |
| +0.900 | 0.368 | 0.723 |
| +0.299 | 1.350 | 1.796 |
| −0.073 | 1.959 | 1.899 |
| −0.452 | 2.568 | 1.648 |

The fold estimates use 1,048,576 pairs per setting; Monte Carlo standard errors are approximately 0.0009–0.0024. These compare population sampling of independent pairs with the implemented finite batch, **not** the exact empirical objective of each stored trajectory dataset. The scaled-linear search has a finite search bound from the nonnegative penalty and alignment term, and brackets all detected grid minima.

The zoo's nearly linear negative-correlation encoders therefore do not establish identification by the objective. The population comparator and trained-network result differ. Distinguishing finite optimization, restricted architecture and finite data would require further controls; these results alone do not prove which one causes that difference.

Both exact-Gaussian benchmark functions have unbounded output tails, whereas a finite-weight tanh encoder has bounded outputs. A close fit over the evaluated distance interval is therefore not exact global implementation of either benchmark. The numerical loss comparison is not a proof of feasibility within this fixed network class.

## What the policy association means

If the sampled log-distances are very non-Gaussian, an affine log-distance code cannot also have an exactly Gaussian sampled output distribution. SIGReg then supplies pressure to change that code. This is a concrete incompatibility, not a proof that the resulting change must be a fold: a monotone quantile transform can also Gaussianize a continuous distribution.

Pair dynamics, network response and optimization decide which distortion is reached. RND/disagreement performing worse than ICM despite smaller mean Gaussian-CDF discrepancy is already a counterexample to a one-variable ranking. The within-seed association tests measure how consistently the marginal shape tracks performance; they do not establish that it is the sole cause.

## A concrete reflection-induced ambiguity

For an approach action of size $a=0.25$ that overshoots the near boundary $d_{\min}=1$, the environment reflects the proposal:

$$
d'=2d_{\min}+a-d=2.25-d.
$$

With $c=d_{\min}+a/2=1.125$, the folded function $g(d)=(d-c)^2$ obeys

$$
g(d')=g(d)
$$

on **every such reflection pair**. Any function of $g$, including its monotone quantile transform, inherits this invariance. Alignment can therefore reward erasing the distinction between the two sides of a boundary bounce. This is a concrete pairing effect, rather than a lack of coverage alone.

In the five fixed datasets, near reflections make up approximately **48.5% of ICM pairs**, **0.51% of random pairs**, and **4.29% of count-policy pairs**. The [diagnostic](output/reflection-diagnostic.json) verifies the squared-distance identity numerically. The rank-code scan prefers a log-distance fold near 1.11 for all five ICM datasets, but the trained network's folds generally occur farther out and one of those datasets trains to an unfolded curve. The proxy therefore **does not predict the learned root**. Its candidate family, representation smoothness and optimization history differ from the neural training problem.

The proxy also has a broader secondary low-loss region near distance 2, closer to the learned turns. That makes it a useful qualitative landscape, but still not a predictive equation: it neither selects the observed neural root as its best solution nor predicts the absence of a fold on dataset 8.

A lower-loss mirror comparator alone does not establish that reflection alignment causes the network's observed fold. The separate [alignment-removal control](REFLECTION_PROTOCOL.md) changes only that pressure while keeping all SIGReg endpoints fixed, and compares it with removing an equal amount of random alignment.

The control observes **20/25 folds in all three conditions**. Removing reflection alignment raises mean R² from 0.6214 to 0.6927, but equal random removal raises it to 0.6892. Reflection-specific improvement is much smaller than the shared effect of lowering alignment and is not significant in the five-dataset sign test (p = 0.0625). This is evidence against a sufficient reflection-alignment-only explanation, not proof that reflections have no effect.

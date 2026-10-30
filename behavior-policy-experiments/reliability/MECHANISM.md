# What we can explain rigorously

## A policy changes the objective

Let $q_\pi(x,x')$ be the collected pair distribution and $\nu_\pi$ the average of its two endpoint distributions. Symmetrize the pairs and let $S_\pi$ be conditional expectation of the other endpoint. For a scalar output with zero mean and unit variance under $\nu_\pi$,

$$
\mathbb E_{q_\pi}[(h(x)-h(x'))^2]
=2-2\langle h,S_\pi h\rangle_{\nu_\pi}.
$$

This is an exact identity, including nonstationary collections. A policy changes both the measure $\nu_\pi$ and which features remain correlated through $S_\pi$. Coverage alone does not determine the objective. Under covariance constraints, leading eigenfunctions describe the optimum when the necessary spectral assumptions hold; an exact Gaussian-output constraint is stronger and may exclude those eigenfunctions.

For Gaussian OU, linear modes have eigenvalue $\rho$ and higher-order modes at most $\rho^2$. The ideal separation is $\rho-\rho^2$. The bounded ICM distributions do not satisfy that theorem's assumptions. Indeed an affine transform of bounded log-distance cannot be exactly Gaussian.

An uneven distribution can require a nonlinear monotone Gaussianizing map. It does **not** require a fold. The folds are an additional information-loss/finite-training failure; they are not a consequence that follows from non-Gaussianity alone. Finite-frequency, finite-weight SIGReg is not an exact Gaussian constraint or an injectivity certificate.

## Which experimental effects survive controls?

1. The earlier successes and failures persist after resetting Adam and restarting the learning rate with identical final OU batches. They are not explained by inherited optimizer memory alone.
2. With each saved dataset frozen, both problematic ICM sources (3 and 7) fail across all three new encoder initializations (100–102). The other six sources succeed in all three. Matching OU controls have no observed folds. This points to the data condition, rather than a single unlucky encoder seed, but each data seed also changes its OU pool and collection randomness.
3. Balanced replay uses the **same stored ICM transitions** and removes the folds in the two historical failures and all five new validation seeds. This identifies a concrete sampling intervention. It changes both alignment and regularization, so it does not by itself identify which term was responsible.
4. [The loss ablation](LOSS_ABLATION.md) separately changes those two sampling measures on additional new seeds. Its control is retrained with two independent batches, rather than compared unfairly with a shared-batch old model.

This evidence does not prove the existence of a specific local minimum, or show that maximizing curiosity universally causes failure. RND and disagreement also fail in this implementation; count coverage and action-balanced sampling are more reliable here. These results are conditional on reward scales, model capacity and training budgets.

## The loss-term ablation narrows the explanation

Five additional new seeds, 18–22, with independent alignment and regularizer batches in every cell:

| Alignment sampling | SIGReg sampling | Shared-range $R^2$ | Observed folded runs |
| --- | --- | ---: | ---: |
| Ordinary | Ordinary | 0.9100 | 1/5 |
| Ordinary | Balanced | 0.9940 | 0/5 |
| Balanced | Ordinary | 0.9119 | 1/5 |
| Balanced | Balanced | 0.9941 | 0/5 |

The two historical failures show the same pattern: balancing SIGReg restores roughly 0.994 recovery, whereas balancing alignment alone leaves the mean near 0.337. These results isolate a sampling intervention on the regularizer within this protocol. They do not claim every possible failure is regularizer-driven.

## What “balanced SIGReg” means here

We keep the SIGReg formula and its loss coefficient unchanged. We change which stored observation pairs enter its batch:

1. Put each pair into one of **32 bins of log apparent triangle height**, using its source observation. The bin range comes from the known sensor geometry; no true distance labels are used.
2. Resample stored pairs with replacement, giving each pair weight inversely proportional to the number of pairs in its bin. If bin $b(i)$ contains $n_{b(i)}$ pairs and $M$ bins are occupied, the ideal sampling probability is

$$
p_i=\frac{1}{M\,n_{b(i)}}.
$$

3. Apply ordinary SIGReg to the encoded source **and** next observations of those resampled pairs. In the “ordinary alignment / balanced SIGReg” condition, only the regularizer uses this replay pool; alignment keeps its ordinary pool.

Thus a bin with 100 stored pairs and a bin with 10 stored pairs receive the same total sampling probability, rather than the first contributing ten times as much. The code draws a fixed replay pool, so equal bin contributions hold **in expectation**, not exactly in every batch. Empty bins stay empty; source-bin balancing does not make the next-observation distribution uniform. The ICM-to-OU mixing schedule and final OU data stay unchanged.

The code implements this by resampling; SIGReg still averages its actual batch uniformly. The weighted expressions below describe the resulting replay measure.

This intervention is implemented in [`balanced_views`](ablate_losses.py), using [`size_bin`](policies.py). It uses a known observation ordering; it is not a general guarantee of uniform physical-state coverage.

## Why the sampling changes SIGReg gradients

Write $h_i$ for a sampled scalar encoder output, $v_i$ for its normalized endpoint weight, and $t_k$ for an integration frequency. The endpoint-weighted characteristic-function estimates are

$$
c_k=\sum_i v_i\cos(t_k h_i),
\qquad
s_k=\sum_i v_i\sin(t_k h_i),
\qquad
\sum_i v_i=1.
$$

For a standard Gaussian target, $c_k^*=\exp(-t_k^2/2)$ and the target sine average is zero. With fixed integration weights $w_k$, define

$$
R=\sum_k w_k\left[(c_k-c_k^*)^2+s_k^2\right].
$$

Its derivative at a sampled output is

$$
\frac{\partial R}{\partial h_i}
=2v_i\sum_k w_k t_k\left[
-(c_k-c_k^*)\sin(t_k h_i)
+s_k\cos(t_k h_i)
\right].
$$

The implementation adds a common batch-size multiplier. The explicit factor $v_i$ shows that the regularizer's gradient weights the visited distribution, not an abstract uniform physical state space. Repeated near-object observations dominate; rare regions contribute less. Balanced batches change this weighting and the target distribution the encoder is asked to Gaussianize. Shared neural parameters couple outputs, so this derivative alone is not a complete optimization proof.

The strongest supported account is therefore **a data-dependent interaction between Gaussian regularization and finite optimization**. Reweighting the regularizer prevents the observed folds; a changed OU spectral gap or optimizer memory alone does not explain the controls. A proof of convergence, or a characterization of every nonlinear optimum, remains open.

## Action coverage has a separate, elementary guarantee

For physical actions $a\in\{-1,0,1\}$, an $\eta$-uniform mixture obeys

$$
\pi(a\mid x)\ge\eta/3,\qquad
\operatorname{Var}(A\mid x)\ge2\eta/3.
$$

For any nonnegative action-prediction error $e(x,a)$, at the **same state distribution**,

$$
\mathbb E_x\left[\frac13\sum_a e(x,a)\right]
\le\frac1\eta\,\mathbb E_{x,a\sim\pi}[e(x,a)].
$$

Proof: multiply each $e(x,a)$ by $1/3\le\pi(a\mid x)/\eta$ and sum. This says why a conditional action floor protects prediction at alternative discrete actions. It gives no guarantee at unvisited states, does not identify continuous-action responses between the three choices, and does not prevent a folded encoder. It is a supporting observation, not a claim of a new theorem.

The [planning comparison](../planning/SUMMARY.md) separates shared predictor data from policy-native data. Native ICM performs much worse. This is consistent with coverage limitations, but state coverage, action coverage and predictor fitting have not been individually isolated. The measured action entropy alone cannot establish which is limiting.

## Nonlinear extension

The [action-anchor lemma](../nonlinear-control/PROOF.md) gives an exact and approximate result for arbitrary nonlinear autonomous drift under constant physical actuation, state coverage and conditional action variation. It explicitly assumes informative smooth coordinates and uses conditional-drift error, not ordinary finite-step JEPA loss. The remaining proof obligations are written beside the bound; a successful simulator check is not substituted for them.

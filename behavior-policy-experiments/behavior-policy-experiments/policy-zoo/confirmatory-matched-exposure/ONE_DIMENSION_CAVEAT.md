# Why one-dimensional policy equivalence is not itself a new guarantee

Suppose the distance distribution has a strictly increasing continuous CDF F, and a scalar representation h is strictly increasing and maps that distribution exactly to a standard Gaussian. Then

$$
\Phi(h(d))=F(d),\qquad h(d)=\Phi^{-1}(F(d)).
$$

For decreasing h the sign is reversed. This follows directly by matching cumulative probabilities. Conditional on monotonicity, the observation marginal already determines the Gaussianizing coordinate; the transition policy does not appear. The argument does not establish that optimization finds a monotone map, exclude folded solutions, or apply an exact Gaussian constraint to finite SIGReg.

Thus finding similar unfolded codes after fixing the exposure distribution is informative but not automatically surprising. The useful question is which behavior/exposure/optimization conditions produce or avoid folds, and how remaining differences affect actual control.

## A concrete warning about the recovery target

Our common-reset pool is uniform in log-distance. Let

$$
U=\frac{\log d}{\log16}\sim\mathrm{Uniform}(0,1),\qquad Z=\Phi^{-1}(U).
$$

The exact monotone Gaussianizing map is nonlinear in log-distance. In fact, its squared linear correlation with log-distance is

$$
R^2(U,Z)=\frac3\pi\approx0.95493.
$$

To check: Var(U)=1/12 and Var(Z)=1. Integration by parts gives

$$
\mathrm{Cov}(U,Z)
=\int z\Phi(z)\phi(z)\,dz
=\int\phi(z)^2\,dz
=\frac1{2\sqrt\pi},
$$

which yields the expression above. Gaussianizing the endpoints requires divergent output limits; finite bounded neural networks can only approximate this ideal.

Consequently, recovery near R²=0.998 under log-uniform exposure is **not evidence of exact Gaussianization** or a realization of the paper's Gaussian-world optimum. It can reflect a finite-loss compromise that remains nearly linear in log-distance. The previous exploratory models have approximately this recovery; the fresh cohort must be assessed on its own outcomes.

The prospective tests intentionally target practical recovery and physical reaching, not exact satisfaction of the theorem's representation constraint. Stronger scientific value comes from isolating the failure mechanism and transition-learning requirements, rather than relabeling a high R² as a new identifiability guarantee.

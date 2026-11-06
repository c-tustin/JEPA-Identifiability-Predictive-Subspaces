# What a significant equivalence test does not guarantee

The fresh-seed study uses a prespecified paired-mean TOST with a t approximation. Its sign-based check asks a different question about typical seed differences. Neither is a distribution-free guarantee about population means when arbitrarily rare large failures are allowed.

## Why that distinction matters here

Consider the difference D between two policies' scores. Compare two hypothetical populations:

- Every run has D=0.
- Most runs have D=0, but a fraction delta/B have D=B. Its mean difference is delta, exactly the equivalence boundary.

With n independent runs, the second population produces an all-zero sample with probability

$$
\left(1-\frac{\delta}{B}\right)^n.
$$

If that probability exceeds 0.05, declaring equivalence whenever all differences are zero cannot be a universally valid 5% mean-equivalence test over this class of populations. This is a simple rare-event obstruction, not a criticism specific to TOST.

For recovery differences, B=1 is the general score bound and delta=0.001. For goal-error differences, B=5 is a conservative bound: goals are at most 2 units from the start and 12 actions move at most 3 additional units; delta=0.005. Both examples have delta/B=0.001. At n=60, the all-zero probability is about **94.2%**. Getting it below 5% requires **2,995** independent runs in this illustrative worst-case family.

This is not a power calculation for our actual neural networks, nor proof that those worst-case outcomes occur in our environment. It shows why strong conclusions from 60 runs require assumptions beyond score bounds. Fresh seeds protect against retrospective tuning; they do not remove all distributional assumptions.

For the separate binary fold-risk diagnostic at a five-percentage-point boundary, the corresponding probability is 0.95^60, about **4.61%**. That explains why zero disagreements in 60 paired runs can support a much coarser claim about rare fold probability, while not certifying an extremely small mean-error difference against arbitrary tail behavior.

Report the prespecified tests, effect sizes, all failures, the sign-based sensitivity, and these limitations together. A significant prospective result is evidence conditional on its model and design, not an unconditional guarantee that the policies will never differ.

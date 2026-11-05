# A finite-capacity test without adding world coordinates

**Status:** proposed experiment with an analytic reference, not a completed cloud run. The current neural pilot has one scalar world variable and one scalar latent, so lack of a feature-selection difference may simply mean there is too little capacity pressure. This design makes the information budget explicit.

## Same sensor, a different synthetic transition

Use 16 possible physical distances. Four binary *features of that distance* are its binary digits:

$$
D=1+8U+4V+2C+N,\qquad X=V_{\rm triangle}/D.
$$

The simulator stores only the integer distance. The binary notation defines observable features, not additional physical coordinates or visual channels. This transition can jump between distances; it is a synthetic feature-selection benchmark, not the original physical actuator.

Set the next distance's digits using

$$
U'\sim\operatorname{Bernoulli}(1/2),\quad
V'=U\oplus E_q,\quad
C'=C\oplus E_p,\quad
N'\sim\operatorname{Bernoulli}(1/2),
$$

where all fresh noise draws are independent, $E_q\sim\operatorname{Bernoulli}(q)$, and $E_p\sim\operatorname{Bernoulli}(p)$. Uniform distance is stationary. Fix $p=0.1$ and vary $q$.

The feature $U$ has no self-predictability. Neither does $V$. Nevertheless, the pair $(U,V)$ carries $1-H_b(q)$ bits of predictive information because the current $U$ predicts the next $V$. A pair containing $C$ and an otherwise irrelevant digit carries $1-H_b(p)$ bits. Here $H_b$ is binary entropy in bits.

Give the representation exactly four symbols: two bits, not an unconstrained real number. Compare the six encoders selecting two named digits, all of which have the same uniform four-symbol marginal. Then:

| Copy noise $q$ | Predictive information of $(U,V)$ | Competitor containing $C$ |
| ---: | ---: | ---: |
| 0 | 1.0000 | 0.5310 |
| 0.05 | 0.7136 | 0.5310 |
| 0.10 | 0.5310 | 0.5310 |
| 0.20 | 0.2781 | 0.5310 |
| 0.50 | 0 | 0.5310 |

This makes a set-level prediction: for $q<p$, retaining two individually unpredictable features beats retaining the individually predictable competitor, within this six-encoder candidate class. It is not a global optimum claim over every balanced four-symbol partition of the 16 observations.

## Side information provides a second test

Supply $U$ to the predictor without requiring the encoder to retain it. The latent $(V,C)$ then has conditional next-code entropy $H_b(q)+H_b(p)$ and joint-input predictive information $2-H_b(q)-H_b(p)$. At $q=0.05$, that is about 1.2446 bits, exceeding the 0.7136 bits of $(U,V)$ even with $U$ supplied. This predicts a change in the preferred feature set.

Supplying a distance digit is a known-geometry side-information privilege. State it explicitly; it is not an additional label-free discovery. Probe labels should otherwise be used only for evaluation.

## Objective and scientific limits

With a categorical next-code predictor and a fixed uniform code marginal, optimized cross-entropy is exactly conditional entropy. That tests predictive closure under a proper distributional objective without a conditional Gaussian approximation. It is an objective extension, not a reproduction of the paper's squared-error/Gaussian theorem.

For squared prediction error, numeric code assignment can affect the ranking. Compare it separately with categorical prediction, including permutations of the four numeric codes. Do not claim mutual-information equivalence merely because code entropy is fixed.

First enumerate the exact transition matrix and six candidate encoders. Then train balanced four-symbol encoders from the original six triangle coordinates and compare their feature probes and likelihoods with the analytic reference. Arbitrary learned partitions may differ from named digit subsets; inspect rather than relabel them. This would test actual capacity-dependent feature competition while preserving the one-distance scene.

[Predictive Closure directions](PREDICTIVE_CLOSURE.md) · [Action-information controls](ACTION_INFORMATION.md)

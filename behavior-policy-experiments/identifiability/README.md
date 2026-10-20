# Identifiability from an object that shrinks with distance

This is a one-dimensional adaptation of **Theorem 1** (page 5; proof in Appendix A) of [*When Does LeJEPA Learn a World Model?*](https://arxiv.org/abs/2605.26379), Klindt, LeCun and Balestriero. It includes a derivation and a small unsupervised numerical experiment. It does not reproduce the paper's converse theorem or its full experimental suite.

## 1. The environment

Put a fixed triangle at the origin. An agent looks straight at it while moving along a ray. Its distance is positive, and the object stays frontoparallel, with fixed physical size, orientation and illumination.

Let the triangle's three physical vertices be the rows of a fixed matrix V. With unit focal length, the ideal observation is

\[
 x(d)=V/d,\qquad s(d)=1/d.
\]

Here s means a **linear image dimension**, such as radius or side length. Image area instead scales as 1/d². Scaling changes the object's apparent size and preserves its shape.

The experiment feeds the six continuous projected vertex coordinates to an MLP. This is a minimal visual sensor, not a CNN trained on raster images. It avoids finite-resolution and visibility issues while preserving the inverse-distance observation law. Training never receives distance, log-distance or their labels.

## 2. Choose coordinates and movements that satisfy the theorem

Physical distance cannot be an exactly Gaussian latent: it is positive. Instead define

\[
 z\sim\mathcal N(0,1),\qquad
 d=d_0\exp(\sigma z),\qquad \sigma>0.
\]

Thus distance is log-normal, and the renderer is a nonlinear, injective map of a Gaussian latent:

\[
 x=g(z)=\frac{V}{d_0}\exp(-\sigma z).
\]

Generate positive pairs with the explicit stationary OU transition used by the paper:

\[
 z'=\rho z+\sqrt{1-\rho^2}\eta,
 \quad \eta\sim\mathcal N(0,1),\quad \eta\perp z,
 \quad 0<\rho<1.
\]

Both z and z' have the same standard Gaussian marginal. In physical coordinates the movement rule is

\[
 d'=d_0(d/d_0)^\rho
       \exp\!\left(\sigma\sqrt{1-\rho^2}\eta\right).
\]

An agent can implement this as a random move to the next positive distance along its viewing ray. This is deliberately a stationary exploration policy in **log-distance**; ordinary constant-step walking in physical distance does not satisfy these assumptions. The code draws independent stationary pairs, rather than a single correlated trajectory.

Defaults: d₀ = 4, σ = 0.7, ρ = 0.9. A central 95% of distances lies approximately between 1 and 15.8 in arbitrary units. No distances are clipped.

## 3. The result to recreate

Let y=f(x) be a scalar embedding. The paper's ideal population objective specializes to

\[
 \min_f\ \mathbb E[(f(x')-f(x))^2]
 \quad\text{subject to }f(x)\sim\mathcal N(0,1).
\]

Its global optima satisfy, almost surely,

\[
 \boxed{f(x(d))=\pm\frac{\log(d/d_0)}{\sigma}},
 \qquad L_*=2(1-\rho).
\]

In one dimension the orthogonal ambiguity is a sign. Equivalently, if s₀=1/d₀ is the apparent size at the reference distance, an ideal encoder is

\[
 f(x)=\pm\frac{\log(s_0/s(x))}{\sigma}.
\]

This is a nonlinear inverse of the observation process, followed by a possible reflection. The network is not supplied this formula.

At the optimum, the learned dynamics also have the correct form:

\[
 y'\mid y\sim\mathcal N(\rho y,1-\rho^2).
\]

**What is identified?** The Gaussian log-distance coordinate, up to sign. Physical distance is available through the nonlinear decoder d=d₀ exp(±σy). The result does not make d a linear function of y or give the agent an absolute physical scale without calibration.

## 4. A short proof

Write h(z)=f(g(z)). Expand it in normalized probabilists' Hermite polynomials:

\[
 h(z)=\sum_{k\geq1}a_k\psi_k(z),\qquad
 \sum_{k\geq1}a_k^2=1.
\]

The constant coefficient vanishes because the embedding has mean zero; its variance is one. Under the explicit OU pairing,

\[
 \mathbb E[\psi_j(z)\psi_k(z')]=\mathbf1_{j=k}\rho^k.
\]

Therefore

\[
 L(h)=2-2\sum_{k\geq1}a_k^2\rho^k
       \geq 2(1-\rho).
\]

Since 0<ρ<1, equality requires aₖ=0 for every k≥2. Thus h(z)=a₁z with a₁²=1, giving h(z)=±z. Such an f exists because the ideal observation is injective in z. Substitution gives the stated transition for y.

Notice that this forward proof needs only zero mean and unit variance; full Gaussianity is sufficient but stronger than necessary. This is also noted in the paper's Section 5.2. No claim about the converse for arbitrary physical environments is needed here.

## 5. Numerical reproduction

Run from the repository root with the dependencies in `requirements.txt` installed:

```sh
python3 behavior-policy-experiments/identifiability/reproduce.py
```

The implementation uses a 6 → 48 → 48 → 1 MLP. It receives fixed, bounded transforms of the observed vertex coordinates. Each batch contains paired observations. The two training terms are:

1. Squared distance between paired **raw** embeddings.
2. **SIGReg**, using the squared discrepancy between the empirical characteristic function and the standard Gaussian characteristic function.

`SIGReg1D` matches the scalar specialization of the [authors' SIGReg implementation](https://github.com/klindtlab/lejepa-identifiability/blob/main/experiments/lejepa_id/losses.py). With M=2B embeddings, it computes

\[
 L_{\mathrm{SIG}}=M\sum_{j=0}^{16}w_j e^{-t_j^2/2}
 \left[\left(\frac1M\sum_m\cos(t_jy_m)-e^{-t_j^2/2}\right)^2
       +\left(\frac1M\sum_m\sin(t_jy_m)\right)^2\right],
\]

where tⱼ are 17 equally spaced knots from 0 to 3, and wⱼ are twice the trapezoidal integration weights. In one dimension, normalized random projections are ±1; both signs have identical squared error, so computing duplicate random slices is unnecessary. The numerical implementation is checked against the authors' full 256-slice expression.

Training minimizes (1−λ) L_alignment + λ L_SIG, with λ=0.1 by default (`--sigreg-weight`). There is **no batch whitening during training**: SIGReg itself constrains mean, variance and higher-order distributional structure. The toy run uses 10,000 steps, batch size 512 pairs, Adam at 0.001, with constant learning rate for the first half and cosine decay to zero for the second half. These toy settings differ from the paper's larger experiments; the implementation does not claim to reproduce the entire authors' pipeline. A finite minibatch objective and neural optimization do not inherit the exact population guarantee automatically.

Evaluation uses new samples, and normalization is calibrated on a separate sample. Raw embedding mean, variance, alignment and sign-aligned error are also reported, so evaluation normalization cannot hide training collapse or an incorrect scale. The remaining embedding metrics use the calibrated representation. A linear probe is fitted on half of the evaluation set and scored on the other half. Latent values are used only for reporting metrics and resolving the unidentifiable sign. Three training seeds are run by default. The output directory contains a figure, metrics and each trained model, including its training history.

The useful checks are:

- R² for a linear probe from the embedding to log-distance should approach one.
- Pair alignment should approach 2(1−ρ)=0.2, with embedding variance near one.
- Transition residual variance should approach 1−ρ²=0.19.
- A linear probe from raw apparent size to log-distance is substantially worse.
- A linear probe from the embedding to **physical** distance need not approach one.

There is also an analytic Gaussian-preserving control: h(z)=z for |z|≤1 and h(z)=−z otherwise. It has exactly Gaussian marginals, yet scrambles the coordinate and has worse alignment. This illustrates why matching the marginal alone does not establish identifiability over arbitrary measurable encoders. This control uses the known latent only for evaluation; it is not trained.

The script reports the paper's approximate-identifiability expression as an additional diagnostic. All ingredients are Monte Carlo estimates and the output is only approximately centered; this is **not** a certified verification of Theorem 3. Small sampling errors can dominate the gap near the optimum.

### Observed results

The default SIGReg run was executed locally with Python 3.12.3, PyTorch 2.2.2 and Matplotlib 3.9.0. The final held-out scores were:

| Seed | Embedding → log-distance R² | Apparent size → log-distance R² | Raw embedding variance | Calibrated pair alignment |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.9890 | 0.7661 | 0.9250 | 0.2052 |
| 1 | 0.9905 | 0.7737 | 0.9401 | 0.2052 |
| 2 | 0.9906 | 0.7776 | 0.9360 | 0.2028 |

The theoretical alignment optimum is 0.2 under the exact standard-Gaussian constraint. The raw learned representations have variance slightly below one and raw alignment 0.1888–0.1919; this is not a violation of the theorem because that constraint has not been met exactly. The Gaussian-preserving scrambled control had alignment 0.787–0.802, despite exactly Gaussian population marginals. Calibrated transition residual variances were 0.1925–0.1949, versus the ideal 0.19.

For comparison, the earlier quantile-penalty version attained R² around 0.996. Switching to SIGReg and removing training whitening changes the finite-sample optimization problem; the current result is a closer implementation of the requested regularizer, but is not a claim of improved numerical accuracy.

Recovery is approximate, especially in the sparsely sampled distance tails, as the plot shows. These three seeds support this toy experiment; they do not establish performance for other sensors or exploration policies. The raw values and configuration are saved in `output/metrics.json`.

## 6. Boundaries of the claim

- **One fixed object:** This recovers a one-dimensional coordinate; it does not demonstrate disentangling shape identity from distance. Discrete shape classes are not independent Gaussian latent coordinates.
- **Variable object size:** If physical size S varies and the only measurement is S/d, then (S,d) and (cS,cd) give identical observations. Neither state can be recovered from a single image without additional information or assumptions. Fixing S removes this ambiguity for our toy world.
- **Ideal visibility:** Clipping, occlusion, size normalization of each image, or a finite raster can make different distances indistinguishable. Exact recovery requires observations that retain distance information.
- **Specified exploration:** The theorem applies to the explicit Gaussian OU process in z. Its conclusions are not guaranteed under an arbitrary agent policy or a uniform distribution of physical distances.
- **Population versus experiment:** The proof establishes the ideal global optimum. High test R² supports successful optimization in this example; it is not a proof of global convergence.
- **One-dimensional simplicity:** Any continuous monotone map taking this scalar observation to an exact Gaussian is already forced to ±z. Richer non-monotone maps motivate the alignment argument. A two-dimensional environment would make a stronger representation-learning experiment.

An appropriate next experiment is to add a second independent continuous factor, such as horizontal object position, give both factors the same OU correlation, render pixels without losing injectivity, and test recovery up to a 2×2 orthogonal matrix. Arbitrary categorical shape variation or ordinary walking would be separate tests outside this exact theorem.

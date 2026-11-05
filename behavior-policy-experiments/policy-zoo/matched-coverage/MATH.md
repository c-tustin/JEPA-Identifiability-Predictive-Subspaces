# How we calculate a policy's slowest features

A feature is simply a number assigned to each distance. For example, physical distance, log-distance, or an indicator of being on the far side of a region. “Preferred predictive coordinate” meant **the varying feature that changes least between paired observations**. It was not a new physical coordinate or an action-conditioned predictor.

## 1. Write the transition probabilities

For our 61 reachable distances, define

$$
P_{ij}=\Pr(d_{t+1}=d_j\mid d_t=d_i).
$$

Each row sums to one. The policies in this experiment move only to an adjacent distance or stay still. Their conditional action probabilities are the three entries in that row. The policy can locate its grid state from apparent height; this uses the same known scene geometry as earlier hand-coded controllers. Distance labels do not enter encoder fitting or planning.

A stationary distribution is a vector $\pi$ satisfying

$$
\pi^\top P=\pi^\top.
$$

We fix $\pi$ first. We then change the probability of crossing particular edges while preserving this equation. This is stronger than matching the number of occupied bins.

## 2. Preserve the marginal exactly

Let $M_i$ be the shared integer number of appearances of state $i$, and $N=\sum_i M_i$. Assign each adjacent edge an integer crossing count $c_i$. Enumerate that edge equally in both directions:

$$
C_{i,i+1}=C_{i+1,i}=c_i,
\qquad
C_{ii}=M_i-c_{i-1}-c_i.
$$

Missing boundary edges have count zero. The transition matrix is

$$
P_{ij}=\frac{C_{ij}}{M_i},
\qquad
\pi_i=\frac{M_i}{N}.
$$

Symmetry of $C$ gives detailed balance:

$$
\pi_iP_{ij}=\pi_jP_{ji}.
$$

Both endpoint counts equal $M_i$ exactly. Every policy also shares the same $\sum_i c_i$, so global approach/stay/retreat counts and physical step moments match. Changing where the crossings occur changes the transition rule without changing the marginal distribution. It need not preserve log-distance step moments or mixing time.

The dataset enumerates this stationary joint law. It is deliberately not collected by waiting for a single trajectory to visit every state equally often.

### Why correcting coverage must change the original policy

On this finite line with adjacent moves and reflecting endpoints, a stationary policy cannot sustain a net probability current through an edge. Stationarity therefore forces

$$
\pi_i P_{i,i+1}=\pi_{i+1}P_{i+1,i}.
$$

Once the common $\pi$ is fixed, the forward/backward ratio across each edge is fixed too. The remaining freedom is **how frequently that edge is crossed in either direction**. Our policies vary precisely that freedom. We cannot retain an arbitrary original ICM drift while independently prescribing its stationary occupancy. This is why the corrected ICM policy is explicitly labeled as a new controlled policy.

## 3. Find slowly changing functions

For a feature vector $f$, the expected next feature is $Pf$. An eigenvector satisfies

$$
Pf=\lambda f.
$$

The constant function has eigenvalue one and carries no distance information. After removing it, the largest eigenvalue identifies the slowest nonconstant feature. To compute it stably, use the symmetric matrix

$$
S=\operatorname{diag}(\sqrt\pi)\,
P\,\operatorname{diag}(\pi^{-1/2}).
$$

If $Su=\lambda u$, convert back using $f_i=u_i/\sqrt{\pi_i}$. Normalize with stationary weights:

$$
\sum_i\pi_i f_i=0,
\qquad
\sum_i\pi_i f_i^2=1.
$$

We save the first three nonconstant modes. Their signs are arbitrary; the figures choose positive correlation with log-distance. These chains are deliberately lazy, so their eigenvalues are nonnegative. “Slow” here means high **positive** autocorrelation, not large absolute eigenvalue.

## 4. Why this connects to the encoder

For any stationary, centered, unit-variance feature,

$$
\mathbb E[(f(d_t)-f(d_{t+1}))^2]
=2\left(1-\langle f,Pf\rangle_\pi\right).
$$

For an eigenfunction this equals $2(1-\lambda)$. Thus the first mode minimizes alignment under these normalization constraints. This is an exact finite-state calculation; it does not require continuous-time assumptions or Gaussian latents.

**Our trained encoder also uses unchanged SIGReg.** The eigenvector therefore benchmarks the alignment part of its objective; it is not claimed to be the exact minimizer of the full implemented loss. Holding the marginal fixed makes the population marginal regularization functional the same across policies for any fixed encoder. Paired minibatch correlations can nevertheless change its finite-batch estimates.

These connected reversible nearest-neighbor chains have a monotone first nonconstant mode. A bottleneck can concentrate its change near a particular distance, compressing other regions, without creating a fold. A learned fold would need explanation beyond this first-mode benchmark.

## 5. Connect recovery to planning without conflating them

A nonlinear but one-to-one representation retains distance information while failing affine recovery. A nonlinear action-conditioned predictor could still plan successfully. Conversely, a high global R² can hide local compression and prediction errors that matter for a particular goal.

We therefore report both affine recovery and actual goal-reaching. Shared predictor data holds action training data fixed across encoders. Native predictor data adds the effect of each behavior policy's state-dependent action support. Neither low latent MSE nor a high R² substitutes for reaching the physical goal.

For old learned policies with unknown transition rules, one can estimate joint state counts and a transition matrix from held-out transitions. Stationarity, reversibility, finite-bin uncertainty and policy memory then need separate checks. We do not silently symmetrize the original ICM data and interpret the resulting eigenvectors as exact ICM dynamics. Here the controlled kernels are known, so no estimation is needed.

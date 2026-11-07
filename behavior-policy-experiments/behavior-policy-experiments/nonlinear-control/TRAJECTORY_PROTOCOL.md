# Nonlinear trajectories: fixed comparison

This study removes the paired interventions and ordered encoder used in the first prototype. There is still only one world coordinate, distance $D$, and the observation is the same triangle $X=V/D$.

Compare a zero-drift control with nonlinear drift:

$$
dD_t=[b(D_t)+0.25A_t]dt+0.05(1+0.1\cos D_t)dW_t,
\qquad b(d)\in\{0,\;0.3\sin d\}.
$$

Euler steps are 0.02, with 50 steps per decision and reflection at 1 and 16. Each action is held for those 50 steps. Both worlds have the same state-dependent noise. The zero-drift control is **not** the original Gaussian OU experiment. These bounded, policy-dependent trajectory distributions need not be Gaussian or stationary.

Three behavior policies use actions $-1,0,1$:

- **Random:** independent uniform actions.
- **Persistent:** a stationary Gaussian OU action score with correlation 0.9, thresholded at $\pm\Phi^{-1}(2/3)$ so its marginal actions are uniform. Its internal memory belongs to the policy, not the world.
- **Action-balanced:** the existing heuristic that favors less-used actions within 32 bins of observed log triangle height, with a 10% uniform floor. This uses known sensor bounds, an explicit geometry prior.

Collect 256 episodes of 64 pairs after 32 burn-in decisions. Simulator starts are uniform physical distances. Independent simulator noise and action random streams are matched across drift conditions. No paired actions share noise, and distance labels never enter encoder training.

Use the original flexible 6→48→48→1 encoder, 4,000 updates, batch 512, Adam 0.001 followed by cosine decay, and $0.9$ alignment $+0.1$ SIGReg. Every update uses separate alignment and regularizer batches. Cross raw versus source-bin-balanced **SIGReg** data; alignment always uses raw pairs. Balance by resampling whole pairs with inverse observed source-bin counts. This cannot fill unvisited bins or guarantee a uniform endpoint law. All cells use identical encoder initialization and batch random streams for each seed.

Fix seeds **23, 24, 25** before running the comparison: 2 dynamics × 3 policies × 2 regularizer samplers × 3 seeds = **36 encoders**. This is an exploratory finite-sample comparison, not a power analysis. No tuning based on these seeds.

Evaluate independent affine calibration on uniform-distance probes, then score a shared interior grid 1.2–15.8. Report both physical-distance and log-distance $R^2$, reversed grid steps after orienting by covariance, and representation standard deviation. Neither target is automatically identified by nonlinear passive alignment plus Gaussian regularization. Also report training occupancy and action variance within observed bins; these coarse diagnostics do not establish a pointwise excitation bound.

The earlier action-anchor lemma does not apply to this objective: it requires an informative encoder and a constant action coefficient in a conditional-drift model. A failure here would not contradict that lemma. A success here would be empirical evidence, not a nonlinear identification theorem. Planning and retrained ICM are follow-up comparisons.

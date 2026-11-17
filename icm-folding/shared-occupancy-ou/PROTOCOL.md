# Shared-input multimodal OU experiment

November 16, 2026. Exploratory protocol written before running these fits.

The earlier multimodal study changed only SIGReg's inputs. This experiment instead generates temporal observation pairs and gives exactly the same encoded endpoints to alignment and ordinary SIGReg. Historical results are retained under their original interpretation.

## Generator

Use a stationary latent OU skeleton: `z_next = exp(-dt)*z + sqrt(1-exp(-2*dt))*epsilon`, with independent standard-normal initial states and innovations. Transform each state by `q = F_target_inverse(Phi(z))`, then `d = exp(q)` and the existing six-dimensional sensor. The target is the earlier 64-bin piecewise-uniform log-distance distribution: equal log-Gaussian peaks of width 0.14 at (2,8) or (2,5,12), plus 10% uniform-log background on [1,16]. Include a uniform-log control. This preserves the *specified discrete-bin target* in the population for every positive dt. It is not an exact continuous Gaussian mixture.

This is a passive simulator-coordinate process, not a trained actor or the original three-action environment. Distance increments can exceed the original 0.25 action bound. Stationary initialization is privileged; no claim of rapid convergence from arbitrary initial distances. The transform yields state-dependent distance increments: moves are small in dense modes and larger in valleys. Smaller dt increases temporal correlation; it does not change the invariant distribution. Ordinary OU step-size changes alone do not create multiple stationary modes.

Generate 256 independent length-64 trajectories (16,384 adjacent pairs) per condition and seed. No pair reweighting, endpoint replacement, per-loss sampling, or old ICM pairs. Pair boundaries never cross trajectories. Match normal draws across distributions and dt. Compare dt=0.01 and 0.1 on seeds 130,131,136. These are diagnostic seeds, not a fresh confirmatory cohort.

## Encoder

Retain the existing gain-4 scalar MLP, seed-specific initialization, 4,000 updates, batch 512, historical scheduled Adam rate and `0.9*alignment + 0.1*SIGReg`. Uniformly sample pair indices with replacement using one seed-specific index stream shared across all six conditions. Each update calls the encoder exactly once on `cat(x[ix],xp[ix])`. Alignment compares its two halves; ordinary SIGReg consumes that same entire tensor. There are no occupancy weights. Both endpoints receive alignment gradients.

## Checks and reporting

Audit inverse-CDF round trips, independent-sample target occupancy, exact OU replay, adjacent-pair continuity and sensor reconstruction. Audit one forward call and shared loss tensor with a spy regularizer; check both loss gradients reach both endpoints. Retain source/protocol hashes, complete trajectories, indices and initial/final weights. Reload each final checkpoint and independently recompute float64 slopes on 20,001 log-distance points. Report sign-changing roots as finite-grid fold checks, never a continuous injectivity guarantee.

Plot prescribed versus measured occupancy, each trajectory example, transition matrices between peak basins, and learned encoders. Report endpoint histogram total variation, per-endpoint marginals, latent lag correlation, mode switching and distance increment quantiles. Do not treat correlated endpoints as independent observations. Population stationarity is an analytic property of the specified generator; finite generated histograms and fitted fold counts are empirical. Interpret changes with dt as changes in transition dynamics at fixed target occupancy, subject to finite dataset differences. This does not isolate occupancy from dynamics across different target distributions.

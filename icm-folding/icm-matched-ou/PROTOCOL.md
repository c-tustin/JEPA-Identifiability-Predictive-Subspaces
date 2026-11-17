# ICM-occupancy-matched OU, shared losses

November 16, 2026. Fixed exploratory protocol written before these fits. Four gallery sources: seeds 130,131,134,136. Existing studies/checkpoints remain unchanged.

## Question and arms

Can a passive transformed OU process reproduce the concentrated native ICM occupancy, and does the same gain-4 scalar encoder fold when both alignment and SIGReg see its temporal pairs?

For each seed train three arms: original ICM pairs with shared loss inputs (`native`); OU time step 0.01 (`ou-small`); OU time step 0.1 (`ou-large`). These dt values are inherited from the earlier multimodal study, not selected using new fold outcomes. No inference from the earlier native controls' fold labels to the new controls is assumed: the earlier trainer sampled separate regularizer endpoints, whereas here both losses always share the exact same endpoints and encoder evaluation.

## Occupancy transform

Pool the source's 16,384 current and 16,384 successor observations. Sort the 32,768 endpoints by sensor distance. Run a stationary standard-normal OU chain with `rho=exp(-dt)`, using 256 independent initial normal states and 64 steps per trajectory. At every time compute `u=Phi(z)` and the zero-based rank `floor(N*u)`, clamped at N-1 for numerical saturation. Select the corresponding original observation from the sorted pool. This is the generalized inverse *empirical* CDF: each original endpoint has probability 1/N at every time in the ideal stationary process. No histogram smoothing, uniform background, artificial support, loss-specific resampling or occupancy weights. Native empty bins stay empty. Repeated original endpoints retain their empirical multiplicity.

The two OU arms share initial normals and innovations. Their sampled finite trajectories need not have exactly the empirical occupancy, because samples are correlated. Report actual 32- and 64-log-bin endpoint histograms, total variation from the native histogram, unbinned KS distance, fraction below distance 2, and step quantiles. Measure current and successor occupancies separately as well as pooled occupancy.

Both OU arms have 16,384 adjacent pairs. The original ICM control retains all original pairs, including its original episode treatment; OU pair boundaries never cross trajectories. This rank transform preserves the empirical observation marginal in the population but changes transition dynamics. Stationary OU and its transform are reversible; the ICM actor may have directional flow. This experiment does not match the transition joint distribution, original action limits, or compute of actor training. All source observations and the geometry used to rank them are privileged simulator data.

## Training and evaluation

Reuse the previous shared-input study's `shared_loss`: one encoder evaluation of concatenated current and successor observations; squared temporal alignment on its two halves; ordinary SIGReg on that same entire output tensor. Keep 0.9/0.1 loss coefficients, original gain-4 MLP, seed-specific initialization, Adam schedule, batch 512 and 4,000 updates. All three arms share initial weights and a uniform pair-index stream seeded as in the previous study. Checkpoint every 100 updates for sampled fold histories; retain full plan and final weights. No new actor is trained and no inverse weighting is used.

Independently regenerate OU states/ranks, reconstruct observations exactly from the source pool, check temporal continuity and empty-bin preservation, and verify native pair identity. Independently reload final encoders and reproduce saved float32 curves/slopes, then compute sign-changing derivative roots on a 20,001-point float64 log-distance grid. Report roots inside the observed distance range separately from roots elsewhere in [1,16]. Sampled folds, including temporal histories, are empirical checks rather than injectivity guarantees. No fresh-policy or general causal conclusion from these four diagnostic sources.

Produce (1) histogram/encoder/slope gallery for all four seeds, (2) training histories with turning-point locations, (3) a compact version of the earlier bimodal/trimodal shared-input gallery, and (4) the OU-to-quantile mapping diagram. Preserve imported source/protocol/checkpoint hashes and new source hashes.

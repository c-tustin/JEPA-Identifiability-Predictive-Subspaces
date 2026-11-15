# Hypotheses and a no-fold guarantee

Exploratory follow-up, fixed before new outcomes, 2026-11-14. Reuse and preserve the completed concentration comparison. No new significance claims: all original sources are historical diagnostics.

## Hypotheses

H1: The concentration effect is an optimization-branch interaction between global SIGReg moment errors, near/far sample forces and shared encoder parameters. There is no intrinsic 87.5% threshold. At the current budget, 87.5% can reverse one branch while 93.75% follows a different ordered path. Test actual optimizer contributions and early/late measure switches. A competitive ordered solution at 87.5% would show that the observed fold is not required to achieve that objective value; a worse tested ordered fit would not prove every monotone function is worse.

H2: ICM's near-boundary preference is partly driven by observation geometry: fixed physical actions make larger visual changes on the near side, and inverse/forward feature learning maintains prediction-error rewards there. Distance 2 is a threshold used in analysis, not an algorithmic attractor. Reverse the sensor geometry, and separately equalize action-response geometry at the actual neural input. These interventions change the observation law and model difficulty, so they do not uniquely isolate visual amplitude from curvature or feature learning.

H3: A mathematical no-fold guarantee requires a hard order constraint or a monotone parameterization with a known ordered sufficient observation coordinate. Weighting or a finite hinge penalty alone cannot guarantee it. Test two constructions using the measured apparent height s=x[:,1]=1/d: positive-slope parameterization and exact isotonic projection after gradient proposals.

## Guaranteed models

Use 65 geometric knots in apparent height [1/16,1]. Both models are piecewise linear between knots and extrapolate with the end slopes. Require all segment slopes >=m=0.05. One parameterizes each slope as m+softplus(a); the other optimizes knot values and projects z−m*s onto the nondecreasing cone with pool-adjacent-violators after each Adam step. The initial function is the same affine height map from -1 to +1. Use double precision, record the analytic segment-slope certificate after every update and independently check the final interval certificate.

Same stored alignment pairs, matched paired regularizer draws, 4,000-step schedule, loss weight and batch sizes as the free encoder. Two methods × ten sources × [native + (a=0.875,0.9375)×(one,two far bumps)] = 100 fits. No ordering hinge is added. The model class and known apparent-height ordering are changed; this is an explicit prior, not unsupervised identification. No additional observations are used.

Compare both full-population loss (SIGReg multiplier fixed at 1,024) and mean loss over 16 fixed independent paired batches against the corresponding free encoders. Report alignment, regularizer discrepancy, readout R², variance and certificates separately. Model classes and optimization geometry differ, so only finding an explicit lower-loss ordered solution is decisive evidence against the necessity of that particular folded solution.

## Encoder mechanism and switches

Reproduce source 131 for a=0.875 and 0.9375, one/two far bumps. Decompose each SIGReg output gradient into contributions from near and far endpoints using the same global moment errors. Accumulate separate alignment/near/far Adam first moments and use the actual shared denominator. Probe fixed near/far slopes and finite component updates at selected checkpoints and the first observed endpoint-slope flip; verify exact final weight reproduction. Component interventions include history and need not have additive finite function changes.

For the one-far-bump pair, switch a in both directions at update 100 and 500, retaining model and optimizer state, matched future batches and the original total 4,000-update schedule. Four switch runs. Early-switch versus late-switch behavior tests path dependence; final no-fold outcome is not assumed.

## Actor geometry controls

For sources 130–139, train 600-update ICM/actor runs with two sensor changes, keeping all other code/settings unchanged. Mirror: sensor(d)=original_sensor(17−d). Equalized: linearly interpolate the two original endpoint observations **after tanh(4*x)**, then invert that preprocessing for the raw sensor output. This preserves endpoint input vectors while making interior fixed-step visual displacements constant at the network input. The equalized observation curve is a line, so curvature as well as amplitude changes.

Twenty new actor fits. Keep native saved actors as controls. Collect fresh 256-episode trajectories using the true physical distance labels saved in `z`, not inverse apparent height (invalid in altered sensors). Measure probability below 2 and above 15, median distance, policy drift, reward map and exact visual response. Preserve diagnostic RNG state. A mirrored or equalized result is conditional on these toy sensors and finite training, not a universal ICM claim.

## Artifacts and checks

Save checkpoints and machine-readable results, clear plots of sensor response and actor occupancy, free-versus-guaranteed curves/objectives, switch histories, and local force contributions. Independently verify certificates, projection correctness, original-run reproduction, altered-sensor geometry, actual optimizer reconstruction, held-out metrics, and source hashes. State which hypotheses are supported or contradicted and distinguish empirical training outcomes from analytic guarantees.

## Sequential optimizer-memory control

After the four planned switch tests showed early switching followed the new measure while late switching retained the earlier branch, add two update-500 switches with Adam reset. Preserve starting weights, remaining schedule and future draws. This distinguishes persistence due solely to optimizer memory from persistence in the learned curve. It is an explicitly post-result diagnostic. Near/far Adam components are accumulated in double precision; float32 summation/EMA roundoff is reconciled into the far component while recording the residual. The original optimizer/model updates remain untouched and must reproduce exactly. Because opposing components can be much larger than their sum, finite isolated component probes use a recorded common fraction limiting the largest coordinate change to 0.0001; linearized slope projections retain the full component direction.

## Sequential tail-code diagnostic

After observing the much larger uniform-grid output variance of the 93.75% free encoders, inspect conditional near/far code statistics for all forty matched one/two-bump models. Then, for source 131, translate its trained 93.75% far-code distribution in output space while keeping its shape and the near codes fixed. Evaluate the exact implemented 17-frequency SIGReg discrepancy and derivative, and the far self-interaction's dispersion derivative at two mixture masses. This is an output-distribution diagnostic, not a realizable smooth encoder intervention or a new training run. It tests a proposed rare-tail parking mechanism; it does not establish a unique cause of the basin selection.

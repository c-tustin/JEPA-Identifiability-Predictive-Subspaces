# Follow the creation of a fold, not only its final shape

Exploratory continuation, fixed before the new pure-ICM balancing results. Same frozen source datasets and diagnostic seeds 130–139. No new physical coordinate or actor learning.

## Sampling intervention

At gains 2, 4 and 8, train pure ICM for 4,000 steps with ordinary alignment pairs and either ordinary or log-size-balanced SIGReg pairs. Balance the same stored complete pairs using the existing 32-bin resampler. There is **no final OU phase** in either condition. Compare the 30 new balanced models with the 30 ordinary models already saved by the gain experiment.

Reproduce ordinary trajectories for seeds 130, 131 and 136 at all gains. They were chosen for a monotone final curve, a fold created after initialization, and a fold already present at initialization. These are mechanism examples, not a favorable-outcome subgroup. Run their matched balanced models as part of the ten-seed cohort. Total: 30 new balanced fits and 9 ordinary trajectory reproductions.

Reuse identical network initialization, training-pair index and independent regularizer-index streams, 0.9/0.1 loss weights, Adam settings and learning-rate schedule. Retain all outcomes. First require the instrumented ordinary seed-131/gain-4 run to reproduce the earlier checkpoint before starting the rest.

## What to record

Every 25 updates, save the scalar function and its signed derivative on 257 log-spaced physical evaluation distances. For the three diagnostic seeds, also record the separate weighted-gradient norms, Adam first-moment components and their first-order effects on the output slope. Save full gradients, moments and model/optimizer states at selected snapshots for independent replay; other frames keep the output-space effects rather than duplicating all parameter vectors. Labels enter only evaluation and differentiation of the saved function.

Adam first moments are linear in the gradient history. Track alignment and SIGReg moments separately, but use the **actual combined second-moment denominator** for both directions. Their parameter-update directions must sum to the actual Adam direction up to floating-point error. This is an attribution conditional on the actual optimizer state, not two independently run optimizers or a causal loss ablation. The denominator depends on both losses.

Obtain slope-change predictions with nested automatic differentiation and compare their sum with the observed before/after slope change. Report the Taylor residual and floating-point moment/update reconstruction error. Never describe a poor local approximation as an explanation of the finite update. Near a simple fold root, use the implicit-root relation to predict movement only when the root derivative is nonzero and the observed root can be matched locally; report failures and unmatched roots.

## Interpretation

The scale landmarks 2, 3.46 and 4 are not asserted to be exact fold roots. A root solves cancellation in the learned full derivative. Frozen-weight rescaling is an exact algebraic identity; retrained-model movement includes changing weights. The interventions test how the chosen sampling measure interacts with the nonlinear input scale. Remaining failures or disagreement between loss attributions and actual interventions must be retained and investigated.

No post-hoc significance test, new planning claim, universal optimizer guarantee, or new validation cohort is claimed from these same-seed diagnostics.

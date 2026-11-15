# Original MLP gradients and alternative regularizers

Written before the main runs on November 15, 2026.

Question: Can an observation/action-only, transition-aware collision gradient reduce folds in the unchanged scalar MLP, and does folding also occur with objectives other than SIGReg?

## Fixed design

- Original encoder: 6 -> 48 -> 48 -> 1 tanh MLP, input `tanh(4*x)`. All encoders start from random initialization, paired across methods by source seed. No ordered knots, distance input, geometry constraints, pretrained encoder, or supervised physical labels.
- Ten existing independent ICM trajectory sources, seeds 130--139. This is a paired diagnostic study on historical sources, not a fresh confirmatory cohort.
- Split by trajectory/episode, not by adjacent transitions: episode IDs are the row index modulo the stored episode count, per original collector. First 80% of episodes train; remaining episodes are held out. Training receives only `x`, `xp`, and `action`.
- 4,000 updates, batch 512, Adam 0.001 with the existing half-constant/half-cosine schedule. All objectives share sampled pair indices and initialization. InfoNCE and collision pair matrices use the first 128 sampled transitions to bound compute. No projector or changed encoder width.
- Four objectives: original scalar SIGReg (`0.9 alignment + 0.1 SIGReg`); scalar VICReg (`25 alignment + 25 variance`, covariance identically zero); symmetric distance-score InfoNCE (temperature 0.1, batch-standardized scalar codes); alignment plus univariate Gaussian quantile matching (`0.9 alignment + 0.1 * 1024 * W2_squared`). This is a controlled toy comparison, not a reproduction of published image benchmarks.
- Each objective: no collision gradient, or collision coefficient 1.0. Total: 80 fits. Loss scales differ and no cross-objective numerical loss ranking is appropriate. Mechanism comparisons are paired within objective.

## Proposed separating gradient

Whiten post-tanh successor observations with training-set coordinate means/stds. For each anchor among 128 transitions, select the eight closest codes among *other transitions with the same action*, using detached code distances. Successor disparity `D_ij` is RMS distance of these whitened observed successors, capped at 2. Define a desired code gap `0.5*D_ij`. Penalize squared hinge violations of this gap using code distances normalized by a stop-gradient batch standard deviation. Selection and normalization are detached deliberately: the new encoder gradient separates conflicting codes without supplying a global orientation. It is piecewise defined and not a hard no-fold constraint. At an exact symmetric collision, the absolute-distance gradient can vanish.

Controls/limits: this encourages preservation of observation detail, including potentially irrelevant details; it is not a guarantee of minimal sufficient state. Deterministic same-action successor differences motivate the loss; stochasticity, aliasing and nuisance information can weaken that reasoning. Targets are observed successors, not predictions from a privileged dynamics oracle. The scalar setup may still be difficult to optimize and no success is assumed.

## Evaluation

- At initialization and updates 100, 250, 500, 1000, 2000, 4000: code and derivative on a log-distance grid, derivative sign reversals, scale-invariant backward variation, affine log-distance R2. Physical labels are used only here. Final independent checks use a denser grid and checkpoint reloads. Report tiny sign reversals separately from material folds (backward variation > 0.001).
- Fit a fresh, matched 4 -> 64 -> 64 -> 6 MLP decoder from frozen scalar code + one-hot action to whitened next observation, for 1,500 updates. Score independent held-out native trajectories and a common uniform-distance/random-action evaluation set. Do not use these evaluations to pick the main loss coefficient. Compare action-only and raw-input prediction baselines as reference limits.
- Retain model checkpoints, history curves, exact configuration, training indices hashes and source hashes. Check loss formula gradients, the absence of labels in the training input dictionary, and paired initialization/batches.

## Scope

This study tests an empirical repair in the original MLP. It does not prove injectivity between sampled states, solve unknown dynamics generally, or show that all versions of VICReg/InfoNCE fold. A scalar VICReg has no off-diagonal covariance, and this InfoNCE score is RBF distance rather than degenerate scalar cosine. Further dimensional/temperature/weight controls will be labeled separately if added after the main runs.

Primary definitions: [VICReg paper](https://arxiv.org/abs/2105.04906), [official VICReg loss](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py), [InfoNCE/CPC paper](https://arxiv.org/abs/1807.03748). Gaussian quantile matching is an additional distribution-matching control, not VICReg or InfoNCE.

## Adaptive follow-up: global collision mining

Added after inspecting the first three source seeds of the running main study: local-batch collision gradients barely change the folded SIGReg/quantile examples. This motivates a stronger search for conflicting transitions, not a redefinition of the original 80-fit comparison.

Run twenty additional SIGReg fits: ten sources x collision coefficients 1 and 10. Every 50 updates, encode all available training observations (no test observations), group by action, sort by the *current learned code*, and compare neighbors at offsets 1, 2, 4, 8, 16 and 32. Rank pairs by desired successor-disparity gap minus current code gap; keep the 512 largest violations. Each training update draws 64 of those pairs for the separating gradient. Normalize by a detached training-code scale. This is adaptive hard-negative sampling in representation space, not sorting by a known physical coordinate.

The original 4,000-update encoder architecture, random initialization, base objective, and base pair batches remain fixed. Mining adds computation and extra observation exposures; it is not a matched-compute comparison. There is no distance/dynamics label in mining. Results are exploratory because this follow-up was chosen after inspecting early main-study runs. Do not pick a coefficient using the evaluation set and then present it as prespecified.

After coefficients 1 and 10 left source 131 folded, an additional coefficient-100 diagnostic on that source produced an ordered model and improved held-out observation prediction. Extend coefficient 100 to all ten historical sources to check whether that result is consistent. This adds ten exploratory fits (thirty global-mining fits total, including the source-131 diagnostic reused once). Coefficient selection was informed by inspected development fold and prediction results. It is not a confirmatory held-out hyperparameter test. Per-update training still uses only observed training transitions.

## Fixed-hyperparameter fresh-source check

Before generating any new policies, fix five new source seeds 140--144, the original 600-update ICM policy configuration, and global-mining coefficient 100. For each fresh policy source, run the four baseline objectives and SIGReg with global mining 100 (25 new encoder fits total). Use the same trajectory split and decoder evaluation. Do not tune on these new results. This is a small within-simulator validation; it does not establish transfer to unknown renderers or stochastic high-dimensional environments. The old ten-source coefficient sweep remains explicitly exploratory. Environment interaction uses the simulator as usual; the encoder/miner receives no analytic dynamics or distance labels.

## Gradient-strength control

Add ten historical-source SIGReg fits using the original batch-local separating gradient at coefficient 100. Compare them with global mining at coefficient 100, so an improvement over local coefficient 1 is not attributed to mining when it could merely reflect a stronger auxiliary loss. This is an exploratory development control; it does not alter the fixed fresh-source experiment or select a new coefficient.

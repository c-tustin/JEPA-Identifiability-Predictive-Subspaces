# Gradient repair and predictor learning curves

Exploratory follow-up fixed before new results, 2026-11-14. The previous goal turn made progress: 60 saved occupancy interventions and verified figures are present. This follow-up preserves those results.

## Gradient experiments

Reuse stored ICM source seeds 130–139 and the existing gain-4 unconstrained encoder. All from-scratch trials retain the same 4,000-step schedule, batch size, initialization and separate alignment/regularizer streams. Test: inverse square-root occupancy weighted moments; full inverse moments with weights capped at 20 times their dataset mean; full inverse moments plus an explicit order penalty; and ordinary SIGReg plus the same order penalty. Existing full inverse-weighted and ordinary controls remain comparisons.

Stress test gain 8, all ten seeds: ordinary, full inverse-weighted moments, and full inverse-weighted moments plus order penalty. This tests a known harder architecture scaling, not independent validation seeds.

Repair existing gain-4 folds in seeds 131,132,133,136,139, starting from the same ordinary update-4,000 model and Adam state. Continue 8,000 updates at constant rate 0.0001. Compare ordinary continuation, weighted moments retaining Adam, weighted moments resetting Adam, and weighted moments plus order penalty retaining Adam. All cells share future pair and endpoint draws. Report all outcomes; preventing a fold and repairing an existing fold are distinct requirements.

Order penalty: fix the output orientation from the starting curve's covariance with log-distance. On 65 equally spaced log-distance probes, penalize the mean square of the negative oriented difference quotient, with coefficient 1. This adds a known scalar ordering prior; it is not an occupancy-only intervention or an identifiability result. SIGReg remains active to discourage collapse. Evaluate folds on 1,025 points throughout and 10,001 points at the end; report variance and readout R² alongside fold counts.

## Predictor error

Plot all ten saved ICM training curves through update 600. Reproduce and extend the three diagnostic actor/ICM runs (130,131,136) through 1,800 updates with the original optimizers intact. Capture the exact update-600 weights and verify equality with the stored originals. Measure training forward MSE (half summed feature error), held-out on-policy prediction error, fixed uniform-log-distance/all-action error, feature variance, inverse accuracy, and occupancy every 100 updates, using diagnostics that preserve training RNG state. The actor and features both keep changing; decreasing training error is not assumed.

As a separate capacity/target-drift check, freeze each original update-600 ICM feature encoder and actor, then train only its forward model for 6,000 minibatches on stored transitions; evaluate independent trajectories under the same frozen actor. This removes moving targets and changing policy, but does not establish exact asymptotic zero error.

Representation alignment error is plotted separately from ICM forward prediction error: the SIGReg experiment has no action-conditioned learned predictor. Fixed evaluation batches measure alignment, ordinary and corrected SIGReg discrepancies, slope violations, variance, and R² through training; final losses alone cannot establish whether a curve is folded.

## Sequential stronger-gradient follow-up

After the first completed gain-8 comparisons showed remaining folds under coefficient-1 order penalties (seeds 130–132), fix a stronger intervention before running it on all ten hard gain-8 seeds and all five late-repair starts. Keep weighted moments and replace the order term by 100 times the smaller of the two orientation-specific mean squared hinge penalties, with a positive slope margin 0.1 per unit log-distance on the same 65 probes. This lets either output sign win and discourages nearly flat reversals as well as large backward slopes. All other settings and matched draws are unchanged. The combined strength/orientation/margin change is an effectiveness test, not an ablation attributing success to a single component. These 15 additional runs are exploratory follow-ups, not untouched validation.

After the first three strong late repairs succeeded, add a control on all five late-repair starts that uses only stored observations for the strong order penalty. Select the stored endpoint closest to each of the 65 uniform-log-distance probe locations, remove duplicate selections, and normalize each adjacent code difference by the actual log-distance gap. The order/margin/coefficient stay the same. This tests whether successful late repair needs newly rendered probe observations, while retaining the explicit ordering prior. Five added runs; no selection based on the remaining two outcomes.

## Review and deliverables

Create simple, separately labeled figures for: crowded-versus-rare weighting; identical distance pairs becoming ambiguous under a fold and ordered after correction; fold size during training; from-scratch/stress/late-repair counts; and training versus held-out predictor errors. Show native training units and log scales for nonzero error, without claiming zero from plotting resolution. Retain numerical tables, source hashes, checkpoints and independent final-geometry/gradient checks.

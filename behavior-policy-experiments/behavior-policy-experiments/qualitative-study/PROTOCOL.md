# Fixed ten-trial qualitative and planning study

Specified before inspecting this cohort's results (2026-10-31). This stays in the existing one-distance, inverse-size triangle environment. **ICM + OU means the existing ICM→OU curriculum**, not a fixed mixture.

## Trials and budgets

Ten fresh paired seeds, **130–139**. Primary arms: pure ICM, passive Gaussian OU, and ICM→OU. Identical encoder initialization, pair-index stream, optimizer and 4,000 updates per seed/arm; batch 512 and SIGReg weight 0.1. ICM actors use the existing 600-update budget and are then frozen. Every seed saves its policy, collected pairs, models, source counts and representation curves. Extra actor interactions are reported; OU is not computationally matched to actor learning.

Each update uses independent 512-pair alignment and SIGReg batches in **all five arms**, matched across methods. This controls the extra independent observations in the replay arms. It differs from the earlier shared-batch study, so earlier scores are not substituted. This detail was fixed before any full-size cohort run.

Two secondary replay arms use the same curriculum and stored ICM pairs: SIGReg-only balancing with 32 log-size bins, or with 32 linear apparent-size bins. Alignment, pair budget and final OU source stay unchanged. Replay equalizes occupied bins only in expectation and cannot fill empty bins. All five arms are run for every seed; no failed model is replaced.

The original curriculum is pure first-source through 25% of training epochs, ramps OU from 25% to 75%, then uses only OU. Pure ICM never switches; pure OU uses OU throughout. Representation learning receives only observations.

## Planning and success screening

Freeze each encoder. Fit the existing action-conditioned predictor for **1,000 updates**, batch 256. All arms first use identical physical random-action training pairs within seed. Replanning uses the existing 128-candidate, horizon-8 planner for 12 decisions. Physical terminal goal error is the primary outcome. Goal distance and simulator state are used only by simulation/scoring, not by the learned predictor or planner.

Use **24 screening goals** and **64 separate held-out goals**, with distinct fixed random streams, matched across methods within seed. A run qualifies as “screen-successful” when ICM→OU has no substantial fold on the evaluation grid and lower screening mean goal error than OU. A fold means over 1% of grid increments reverse the globally sign-aligned coordinate, ignoring changes below 1e-5 calibrated units. This is a descriptive grid check, not a proof of injectivity. Held-out planning scores are not used to select the subgroup.

Additional policy-native predictor fits use the saved ICM physical-action pairs for ICM-containing arms. **OU-native prediction is undefined:** the passive OU representation pairs have no physical action labels. We never fabricate such labels. The native predictor for ICM→OU therefore uses its ICM physical pairs; it cannot use synthetic OU pairs as action data.

## Statistical analysis

The independent unit is a seed, not a goal or trajectory. Primary comparison: ICM→OU minus OU in held-out mean terminal error over **all ten seeds**. Report each pair, mean difference, paired seed-bootstrap 95% interval (20,000 resamples), and an exact two-sided sign test of the run-level win probability against 1/2. Ties within 1e-6 physical-distance units are excluded and counted, to avoid treating floating-point noise as a win. A supplementary paired t-test concerns the mean difference and relies on normal differences with this small sample.

Repeat the sign test in the independently screened subgroup, stating its size and conditional estimand: performance in runs passing this particular screen, not all possible runs. Apply Holm correction to these two sign tests. This does not test a subgroup selected for favorable held-out outcomes. Fewer than six non-tied pairs cannot achieve two-sided sign-test p<.05 even with all wins; conclusions may be inconclusive. No goals are treated as extra independent training runs.

## Coverage and bin importance

Plot 32 log-distance-bin histograms derived from the observed size, actual representation/SIGReg training exposure, fold/sensitivity diagnostics, and planning errors. Show within-range coverage separately from OU observations outside [1,16]. Per-bin correlations across runs are exploratory, not causal or independent tests.

For a controlled planning-data intervention, freeze each seed's **OU encoder**. Retrain its predictor after removing one of **eight contiguous regions (four bins each)** from the shared physical training pairs. Uniformly resample retained pairs to the same 16,384-pair pool size and keep initialization, updates, calibration, goals and candidate sequences fixed. A matched resampled-all-data predictor is the control. This changes available state/action examples without changing the encoder or training budget. Report per-seed increases in held-out physical error, region demand along oracle paths, and effects on tasks whose oracle path crosses the removed region. It identifies effects of these region-removal interventions, not uniquely important individual bins or a universal coverage law. Eight region sign tests are Holm-adjusted separately and labelled secondary.

No main trial is dropped because it fails, folds, has poor coverage, or contradicts the hypothesis. Smoke tests use a separate output folder and are never added to the main results.

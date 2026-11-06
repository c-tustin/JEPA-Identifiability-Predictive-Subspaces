# Secondary usefulness check: the untrained encoder

Added after cohort training started, before complete-cohort inference. This is **not part of the locked confirmatory family** and does not change its margins, sample size, training or decision rule.

For each of the same 60 seeds, recreate the exact initial encoder used in every matched-policy cell, freeze it without any encoder updates, and train the same 2,000-update predictor on shared data. Use identical calibration data, predictor randomness, goals, planner and physical-error evaluation. Sixty extra predictors; no additional trained encoders or exploration actors.

Compare whole-range recovery and shared-data physical goal error with all seven trained matched conditions. Retain all seeds. Report means, paired mean changes and unadjusted bootstrap intervals descriptively, not new confirmatory p-values. The architecture itself may already produce useful geometry; a post-training policy tie would otherwise conceal this.

Recreate every baseline's geometry and physical rollouts, and check that its geometry equals the saved initial geometry in every corresponding trained condition. The baseline is not a true-state oracle and can fail. Reproducing initialization from its seed is not resetting a trained model after seeing its outcome.

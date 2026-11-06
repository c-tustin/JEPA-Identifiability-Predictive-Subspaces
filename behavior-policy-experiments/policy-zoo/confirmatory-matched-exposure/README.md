# Fresh-seed confirmation: does the apparent policy tie hold up?

**Status: verification gates, before fresh-cohort training.** Results will be reported only after all 60 seed blocks finish. The earlier exploratory results are not new confirmation results.

[Fixed plan](PROTOCOL.md) · [Why this is useful beyond the paper](RESEARCH_VALUE.md)

We fix “close enough” before the new runs:

- Recovery differs by less than **0.001 R²**: 0.1% of target variance in normalized linear-reconstruction error.
- Actual goal-reaching error differs by less than **0.005 distance units**: 2% of one action step.

These are explicit task-resolution choices, not unique physical constants. They retain the strict earlier tolerances. Fresh learned actors, datasets, encoder initialization, and planning cases vary across seeds 6000–6059; all seven policies share randomness within each seed where appropriate.

The test requires every policy pair to pass for whole-range, near-object and far-object recovery **and** shared-data planning. We also keep ICM's native transitions fixed and change only SIGReg's observation exposure. This asks whether broad observations repair geometry and reaching without replacing ICM's action rule.

Total: **540 encoders, 1,080 predictor/planning evaluations**, with 240 newly trained actors. No extra environment coordinates, SIGReg formula changes, or dynamics changes. Starting observations and regularizer exposure match; successor distributions and stationarity do not.

From the repository root:

```sh
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/run_confirmation.py --gate
# Commit the protocol/code, then lock it before any cohort training:
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/run_confirmation.py --freeze
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/run_confirmation.py --run --workers 6
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/analyze_confirmation.py --require-complete
```

Repeating `--run` resumes the same cohort. Encoder/Adam checkpoints save every 500 updates. Completed actors and predictors are reused; an interrupted actor/predictor restarts its bounded fit. Dataset caches are regenerable and ignored by Git. No hypothesis tests are run on a partial cohort.

The main four-claim family includes global recovery/planning equivalence and three directional ICM exposure-control tests, all Holm-corrected. Fold-rate and policy-specific predictor-data tests are prespecified secondary diagnostics, not extra confirmatory claims. TOST is an approximate test of population means; the sign check addresses typical differences. Neither gives a distribution-free guarantee against arbitrarily rare failures.

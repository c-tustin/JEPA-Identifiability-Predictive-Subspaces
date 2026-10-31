# Ten-trial representation and planning comparison

This study plots what ICM, OU and ICM→OU learn in the same distance-only world, then tests whether the representations help the agent reach real physical goals. It also compares two replay binnings and removes training-data regions to see where predictor coverage matters.

[Protocol](PROTOCOL.md) fixes the seeds, budgets and tests before the new results. The main comparison uses all ten trials. A separate success screen uses different goals from the final evaluation, so we do not select the runs because their final test scores look good.

From repository root:

```bash
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/run_trials.py
python behavior-policy-experiments/qualitative-study/report.py
```

Use `--quick --output /tmp/qualitative-smoke` for a reduced execution check. Main source/model/predictor checkpoints permit completed stages to resume; partial encoders save every 500 updates. Outputs are written separately from earlier studies. No existing Predictive Closure run is resumed.

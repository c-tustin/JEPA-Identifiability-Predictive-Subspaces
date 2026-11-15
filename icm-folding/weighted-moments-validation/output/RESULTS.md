# Weighted-moment validation

Five recently generated ICM sources (140–144), new to occupancy weighting. All hyperparameters fixed before this experiment. Known distance is used for occupancy weights.

| Condition | Final folds | Material folds | Mean log-distance R² | Minimum output std |
| --- | --- | --- | --- | --- |
| gain 4, ordinary, initialization | 2/5 | 2/5 | 0.6647 | 2.0475 |
| gain 4, weighted, initialization | 0/5 | 0/5 | 0.9829 | 0.8642 |
| gain 8, ordinary, initialization | 5/5 | 5/5 | 0.3392 | 0.7365 |
| gain 8, weighted, initialization | 5/5 | 5/5 | 0.9265 | 0.8084 |
| gain 4, ordinary, existing fold | 2/2 | 2/2 | 0.3936 | 1.5583 |
| gain 4, weighted, existing fold | 2/2 | 2/2 | 0.0108 | 0.8929 |
| gain 4, weighted_reset, existing fold | 2/2 | 2/2 | 0.0087 | 0.8935 |

All new checkpoints passed independent float64 20,001-point curve/derivative recomputation; paired initialization, training plans and direct-repair starting weights verified. No encoder reinitialization is counted as repair.

New direct-repair starts: [141, 143]. Initial training: 4,000 updates; continuations: 8,000 updates at 0.0001. Gain-4 and gain-8 results are not pooled.

## Historical weighted-moment re-audit

| Historical condition | Final folds |
| --- | --- |
| gain 4 prevention | 0/10 |
| gain 8 prevention | 7/10 |
| late repair weighted | 5/5 |
| late repair weighted_reset | 5/5 |

Historical curves were re-evaluated on the same float64 grid, not retrained. On a portable checkout without historical checkpoints, `historical-reaudit.json` is preserved evidence, not a new re-audit. Finite-grid checks are not continuous-domain certificates. Standard-gain prevention does not establish consistent direct repair or robustness to gain changes. No downstream prediction improvement is asserted.

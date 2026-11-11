# Beijing predictive-closure pilot

Exploratory real-data extension, not an original-paper replication or a causal test.

| Latent | Condition | Six-hour PM2.5 RMSE (seed mean ± SD) | PM2.5 readout R² |
| --- | --- | ---: | ---: |
| 2 | none | 71.90 ± 1.89 | 0.737 |
| 2 | weather | 71.84 ± 0.70 | 0.759 |
| 2 | shuffled_weather | 70.16 ± 1.29 | 0.782 |
| 8 | none | 63.43 ± 1.51 | 0.937 |
| 8 | weather | 63.55 ± 0.96 | 0.935 |
| 8 | shuffled_weather | 63.40 ± 1.57 | 0.937 |

Units: μg/m³. SD measures initialization/training variability on one fixed test year; it does not measure uncertainty across datasets or years.

Baselines:
- persistence: RMSE 61.24; R² 0.570.
- pm_only_ridge: RMSE 67.37; R² 0.480.
- raw_ridge: RMSE 66.32; R² 0.496.

See metrics.json for all seeds, validation-selected ridge penalties, per-feature readouts, PCA/random baselines, collapse diagnostics and month-by-month errors.

The held-out comparisons describe this fixed pilot. They do not establish cross-subsidization, mutual-information maximization, or causal weather effects.

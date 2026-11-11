# First real-data predictive-closure pilot — November 11, 2026

**Implemented and executed:** 18 tabular JEPAs on actual hourly Beijing pollution
and weather. Weather supplied to the predictor does **not consistently improve**
the held-out pollution forecast over zero-input and shuffled-weather controls.
This is an exploratory extension, not a replication of the paper's controlled
CIFAR experiments or evidence that its theorem applies to this dataset.

## Fixed protocol and evidence

2010–2012 train; 2013 validation for frozen ridge-readout penalties; 2014 test.
Six-hour exact-calendar pairs; 24,049 / 8,612 / 8,582 complete-endpoint pairs.
Train-only normalization, no imputation, no split-crossing pairs.
Three paired seeds × dimensions {2,8} × {none, weather, shuffled_weather}.
Each model received 1,500 updates at batch 256. Architecture, loss, optimizer,
seeds and data choices are recorded in [README.md](README.md).

All 18 saved encoder/predictor checkpoints were reopened; their frozen downstream
RMSE, MAE, R² and selected penalties reproduced within numerical tolerance.
Seven automated checks passed, including train-only preprocessing, calendar gaps,
both-endpoint gradients, test-independent readout selection, matched SIGReg
values/gradients against the pinned upstream source, and a synthetic
cross-feature predictive-link calibration.

Recorded runtime: Python 3.12.3, PyTorch 2.2.2, NumPy 1.26.4; CPU float32,
two PyTorch threads. Summed per-model training-plus-evaluation time: 182.9 seconds;
this excludes setup, baselines, plotting and the subsequent checkpoint audit.
No paid infrastructure was used. A separate 20-update smoke preceded the pilot.
No budget or hyperparameter was selected using the pilot's test-year results.

## Held-out downstream results

RMSE in μg/m³; lower is better. Mean ± sample SD over the three training seeds.

| Latent dimension | Predictor input | Pollution forecast RMSE | Current log-pollution readout R² |
| --- | --- | ---: | ---: |
| 2 | Zero weather | 71.90 ± 1.89 | .737 |
| 2 | Current weather | 71.84 ± .70 | .759 |
| 2 | Shuffled weather | 70.16 ± 1.29 | .782 |
| 8 | Zero weather | 63.43 ± 1.51 | .937 |
| 8 | Current weather | 63.55 ± .96 | .935 |
| 8 | Shuffled weather | 63.40 ± 1.57 | .937 |

Baselines on the same test pairs:

| Baseline | RMSE |
| --- | ---: |
| Persistence (future pollution = current pollution) | 61.24 |
| Ridge on current raw-unit pollution | 58.08 |
| Ridge on current raw-unit pollution plus weather | **56.51** |
| Ridge on the 11 transformed encoder-input columns | 66.32 |
| PCA, 2 components / 8 components | 86.22 / 66.68 |
| Untrained encoder, 2D / 8D (seed mean) | 90.80 / 75.95 |
| Two individually most self-predictable columns (pressure, dew point) | 94.00 |

The 8D learned models outperform the equal-dimension random encoders and PCA
on this downstream readout. They still underperform persistence and raw-unit
regression. The difference between transformed-input and raw-unit regression
also shows that the downstream readout's function class matters: a linear
reader of log pollution is not a linear reader of physical pollution.
Poor linear readout must not be equated with information loss.

## What the closure diagnostics say

For each frozen encoder, separate linear readouts predict future latent coordinates
from the current latent, with or without current weather. Mean test R² gain from
weather, averaged over latent coordinates and seeds:

| Latent dimension | Zero-weather training | Weather training | Shuffled-weather training |
| --- | ---: | ---: | ---: |
| 2 | .0123 | .0194 | .0117 |
| 8 | .0292 | .0328 | .0294 |

This exposes weather information useful for predicting the retained target set
within a linear readout class. It does not estimate conditional mutual information
and does not distinguish feature selection from nonlinear coordinate changes.
The small differences between training conditions need independent confirmation.

The weather-conditioned models have lower mean neural latent prediction MSE
(.155 vs .172 for 2D; .416 vs .433 for 8D, compared with zero weather), while
their standalone downstream pollution forecasts do not consistently improve.
The target encodings differ between models, so these MSEs are not errors on a
common fixed target and cannot rank representation quality by themselves.

Learned-predictor residual covariance condition numbers average about 6.7–7.2
in 2D and 14.7–16.3 in 8D, rather than 1. Residual coordinate excess kurtosis
ranges from −.75 to 2.22 across runs. These are departures in the fitted model's
unconditional residuals; they do not prove that an optimal predictor could not
satisfy the theorem. They do prevent treating this run as verification of the
ideal isotropic conditional Gaussian-error assumption. SIGReg also does not
certify exact Gaussian marginals or the fixed entropy constraint.

## Next experiments, before a stronger claim

1. Add nonlinear frozen readouts and a matched supervised forecast baseline. This
   separates representation limits from the current ridge reader's limits.
2. Use validation-only selection over lag, capacity, regularization and training
   budget. Treat 2014 as already inspected; seek fresh stations/years or an
   independent dataset for confirmatory testing. Do not tune to this table.
3. Repeat with controlled cross-feature dynamics and unchanged observation
   marginals. The present shuffled-predictor input control does not intervene on
   the environmental dynamics, so it cannot establish cross-subsidization.
4. Use temporal blocks and additional test years for uncertainty. Three seeds on
   one fixed test year are optimization replicates, not three independent datasets.

Local artifacts: `output-pilot/metrics.json`, `diagnostics.json`, `comparison.png`,
18 `.pt` checkpoints and the exact original training-source snapshots. The first pilot is explicitly versioned on main, including its checkpoints and
original training-source snapshots. New runs are ignored by default. Data provenance and licensing: [NOTICE.md](NOTICE.md).

# Predictive closure on real hourly pollution and weather

This is a runnable **real-data extension** of Predictive Closure, separate from the
original CIFAR-collage replication and the earlier one-coordinate distance worlds.
The first fixed-budget run is described in `output-pilot/RESULTS.md` and `FINDINGS.md`.

## Question and theoretical connection

For a fixed feature set S, the paper studies predictive information between its
present and future values. A feature can be worth retaining because it predicts
**another retained feature**, even if its own next value is hard to predict.
With side information U supplied only to the predictor, the relevant objective is
I((Z_t,U_t);Z_future), under the paper's stated assumptions. Weather is a plausible
source of predictive information about future pollution. It is not assumed to be
unpredictable, independent, exogenous, or causal in this dataset.

We implement f(X) as a small MLP and train

    0.9 MSE(g(f(X_t), U_t), f(X_t+6h)) + 0.1 SIGReg

with gradients through both encoder endpoints. SIGReg is evaluated separately at
each endpoint and averaged. No target detach/EMA, forecast-label loss, or feature
reconstruction loss is added. PM2.5 is part of the observed state; current pollution
therefore participates in self-supervised training. Future pollution labels are used
to fit/evaluate frozen downstream readouts, not to optimize the encoder.

Three matched predictor conditions:

- `none`: weather input filled with zeros.
- `weather`: current weather supplied directly to the predictor.
- `shuffled_weather`: the same weather vectors permuted within the current batch,
  independently each update; this preserves the marginal distribution while breaking
  most pair-specific information. Evaluation uses a fixed full-test permutation.

All three encoders see the same complete observation, including weather. Side
information is a bypass around the bottleneck, not an extra measured variable.
Models have identical architecture and parameter count within a bottleneck size,
initial weights and source-pair minibatches are matched by seed. SIGReg random
projections are also matched. Predictor input weights connected to zero inputs
are inactive in `none`; the shuffled condition provides an active-input control.

Default grid: latent dimensions 2 and 8; seeds 0, 1, 2; 1,500 updates each;
batch 256; AdamW lr .001 and weight decay .0001; clip norm 5; CPU float32;
32 SIGReg projections, 17 knots on [0,3]. This is a tabular architecture and a
reduced compute budget, **not the paper's architecture or training budget**.
The two bottlenecks test sensitivity to capacity; finite dimension and approximate
SIGReg do not certify the paper's exact entropy or reader constraints.

## Data and evaluation

[UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
contains hourly pollution at the US Embassy and weather at Beijing airport during
2010–2014. These are different measurement locations. Attribution/license and
upstream SIGReg notices are in [NOTICE.md](NOTICE.md).

Input: pollution, dew point, temperature, pressure, cumulative wind speed,
cumulative snow and rain, and four one-hot wind categories (11 columns). log1p
is applied to four nonnegative variables: pollution, wind, snow and rain.
Weather variables retain their UCI meanings:
Iws is cumulative wind speed, not an instantaneous wind measurement. The timestamp
and row ID are excluded from encoder/predictor input; no future weather is supplied.

- 2010–2012: representation training, normalization and readout fitting.
- 2013: select each ridge readout's penalty from a fixed seven-value grid.
- 2014: fixed final test year; never used for normalization or penalty selection.
- Preserve the hourly calendar, then form exact six-hour pairs. Drop pairs with
  incomplete endpoints; do not interpolate or join across a missing-time gap.
  Missing interior observations are allowed because only endpoints are inputs.
  No pair crosses a split boundary. Complete-case exclusion may bias the sample.
- 43,824 calendar rows; 41,757 complete rows; 24,049 / 8,612 / 8,582 pairs.

Readouts recover each current feature from the frozen latent, and forecast PM2.5
six hours later in physical units. Baselines: persistence, ridge on transformed
inputs, pollution-only ridge, PCA of equal dimension, and untrained encoders of
the same architecture. `audit.py` adds ridge on current **raw-unit** pollution,
raw-unit pollution plus weather, and a reference selecting k columns by individual
self-predictability on validation. Unequal feature entropy and redundant one-hot
columns mean this last reference is not an exact theoretical feasible-set optimizer.

`audit.py` also fits frozen linear predictors of **future latent coordinates** from
current latent, latent plus weather, or the full observation. Improvement from
weather is a finite-readout diagnostic of omitted predictive information about
the learned target set. It is not a conditional-MI estimate; nonlinear readouts
could change the conclusion. Residual covariance eigenvalues and excess kurtosis
check obvious departures from the ideal isotropic conditional Gaussian assumption;
they do not establish conditional Gaussianity or homoscedasticity.

Latent prediction MSE, variances/eigenvalues, feature readouts, readout penalties,
training history and monthly downstream error are saved. Lower raw latent MSE
alone is not evidence of a better representation. Low linear feature-readout R²
does not prove a feature is absent. Random-encoder baselines help distinguish
information already present at initialization from a training benefit.

## Run

From the repository root, using Python 3.12. Dependencies are pinned in
`real-world-closure/requirements.txt`:

```sh
python3 real-world-closure/data.py
python3 -m unittest discover -s real-world-closure -p 'test_*.py' -v
python3 real-world-closure/run.py --output real-world-closure/output-reproduction
python3 real-world-closure/audit.py real-world-closure/output-reproduction
```

The downloader checks both a pinned archive hash and the CSV hash (on cached data,
the CSV hash). The first download needs network access. Tests use the pinned data;
the SIGReg reference test additionally uses the sibling official checkout and
explicitly skips if it is unavailable. No package installation is performed by
these scripts. `run.py` refuses to reuse an existing output directory.
Choose a new `--output` path for every rerun; the completed local pilot is already
in `output-pilot/`.

Separate smoke run:

```sh
python3 real-world-closure/run.py --steps 20 --seeds 0 --latent-dims 2 --output /tmp/closure-smoke
```

Outputs include every encoder/predictor checkpoint, preprocessing parameters,
configuration, Python-source hashes, runtime versions, metrics JSON and a figure.
New data and outputs are Git-ignored. The completed first pilot, including its
checkpoints and source snapshots, is explicitly versioned under `output-pilot/`.
`FINDINGS.md` is the tracked evidence summary. Individual overlapping hourly pairs
are not independent statistical replicates; three seeds only measure optimization
variability on a single year, and are insufficient for a broad significance claim.

## What would count as progress

First compare weather to **both** zero and shuffled weather within each capacity,
inspect shifts in feature readouts, and ask whether retained features support the
external pollution forecast. Compare learned models with random/PCA baselines and
persistence. Report null or adverse results without selecting favorable seeds.
The present pilot does not establish cross-subsidization: that needs a controlled
change in a predictive relationship while holding observation exposure fixed.

Before a stronger claim: freeze a new protocol, use rolling validation/test years
or additional stations, add nonlinear readouts and temporal-block uncertainty,
vary horizon and capacity on validation only, and replicate on an independent
dataset. Keep a synthetic intervention calibration alongside observational tests.
The supplied synthetic unit check verifies an off-diagonal predictive link; it is
a numerical calibration, not a neural reproduction of the paper.

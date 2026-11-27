# Protocol: does SIGReg alias states in multi-dimensional embeddings, and does it matter?

Fixed November 26, 2026, before any fit in this directory. The SHA-256 is recorded in `output/protocol.sha256` before the
first run.

## Question

All fold results so far use a scalar code. LeJEPA's SIGReg regularizes D-dimensional embeddings through random 1-d
projections. A curve in D ≥ 2 can bend without two states sharing a code, so a scalar "fold" has no direct analogue.
This protocol asks whether SIGReg on ICM occupancy still makes **distant states look closer than neighbouring states**,
and whether that hurts downstream use.

## Training (only the output dimension and the projection step change)

- **Sources and seeds.** ICM sources 130–139, with initialization seed = source seed. The minibatch streams are the
  historical alignment indices and the regularizer indices `randint(..., seed 1900000 + source)`, exactly as in
  `initialization-crossing/run.py`. Same 4,000 steps, learning-rate schedule, Adam 0.001 and loss
  0.9 · alignment + 0.1 · regularizer.
- **Encoder.** The original `6 → 48 → 48 → D` tanh MLP with fixed `tanh(4x)` input, D ∈ {1, 2, 4, 8, 16}. Parameters
  are created after `torch.manual_seed(40000 + seed)` in the original layer order, so the two hidden layers start
  identically for every D.
- **Alignment.** Mean squared difference over batch and dimensions (`(z − z').square().mean()`), which equals the
  original at D = 1.
- **SIGReg in D dimensions.** At each step draw 64 Gaussian directions (generator seed 3000000 + source, so both arms
  see the same directions), normalize them, project the embeddings, and average the original 1-d statistic (17 knots,
  t_max 3, batch-size scaling) over directions. At D = 1 the single direction +1 is used, which is identical to the
  original.
- **Arms.**
  - *ordinary*;
  - *weighted*: the occupancy-weighted characteristic-function moments of `occupancy-folding/run_study.py`, with
    inverse 32-bin log-distance endpoint mass. These weights are privileged and serve as a paired contrast.

100 fits. **Implementation check:** at D = 1 the ordinary arm must reproduce `initialization-crossing` `data<s>-init<s>`
and the weighted arm `native-weighted_moments-<s>`, both bit-for-bit.

## Aliasing measure

Evaluate the embedding in float64 on 2,049 distances uniform in log d over [1, 16]. For grid state d, let
ℓ(d) = ‖f(min(d + 0.25, 16)) − f(d)‖, the embedding displacement of one move. d is **aliased** if some grid state d′
with |d′ − d| > 1 (more than four moves away) has ‖f(d′) − f(d)‖ < ℓ(d). Report:

- the aliased fraction of native occupancy (all stored endpoints, assigned to the nearest grid point);
- the aliased fraction of the uniform grid;
- the minimum ratio ‖f(d′) − f(d)‖ / ℓ(d) over far pairs.

A fit is **aliased** if more than 1% of native mass is aliased. At D = 1 this is reported next to the existing fold
diagnostic as a calibration.

## Downstream (Protocol 1 harness, generalized to D-dimensional codes)

The code is tabulated on the 20,001-point log-d grid, column by column, and read through the same interpolating wrapper.
Codes are standardized per dimension. The rest generalizes with D:

- next-observation decoder: (D + 3) → 64 → 64 → 6;
- log-d readout: D → 64 → 64 → 1;
- planning predictor: residual (D + 3) → 64 → 64 → D, with squared Euclidean cost in standardized code space.

Seeds, goals, action bank, budgets and three replicates are those of Protocol 1. At D = 1 the harness is the Protocol 1
harness.

## Pre-registered predictions

- **M1.** The D = 1 bit-exact checks pass for all 20 fits.
- **M2.** Ordinary SIGReg at D = 2 is aliased in at least 5 of 10 sources.
- **M3.** Aliasing decreases with dimension: fewer aliased ordinary fits at D = 16 than at D = 2.
- **M4.** Pooled over ordinary fits with D ≥ 2, aliased native mass has Spearman ρ > 0.3 with native next-observation MSE
  and with near-goal shared planning error.
- **M5.** Weighted SIGReg has lower aliased native mass than ordinary in at least 70% of (source, D ≥ 2) pairs where
  either arm is aliased.

## Limits fixed in advance

- One deterministic world with a one-dimensional state, so for D > 1 the embedding is a curve in a larger space. This
  probes intrinsic dimension below embedding dimension, which is common in practice but is not the full high-dimensional
  image setting.
- 64 directions per step and small D, unlike LeJEPA's scale.
- One initialization per source.
- The aliasing threshold (four moves versus one move) is a planning-motivated choice fixed here, not a unique definition.

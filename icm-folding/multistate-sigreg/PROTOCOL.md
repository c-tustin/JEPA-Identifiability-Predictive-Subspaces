# Protocol: SIGReg aliasing when the state has more than one dimension

Fixed November 27, 2026, before any fit in this directory. The SHA-256 is recorded in `output/protocol.sha256` before the
first run.

## Question

With a one-dimensional state, SIGReg folds only through the scalar bottleneck. At D ≥ 2 it never aliases states, but
skewed occupancy still compresses the code ([multidim-sigreg](../multidim-sigreg/README.md)). Real JEPA states have
several dimensions. This protocol asks whether skewed occupancy makes SIGReg alias distant states when the state is 2-d
and the embedding dimension is below, equal to, or above it, and whether that harms downstream use.

## World (a 2-d extension of the original renderer)

- **State.** Depth d ∈ [1, 16] and lateral offset b ∈ [−3, 3].
- **Observation.** The original triangle (vertices (0, 1), (±0.866, −0.5)), pinhole-projected as ((x + b)/d, y/d),
  giving 6 coordinates. The observation is injective in (d, b), and no state coordinate is given to any learner.
- **Actions.** 0 stay, 1 d − 0.25, 2 d + 0.25, 3 b − 0.25, 4 b + 0.25. Overshoot reflects at the walls.
- **Data.** Each dataset holds 256 episodes, each with a uniform random start, 3,000 burn-in steps (discarded) and
  64 recorded steps, so 16,384 transitions. The datasets come from scripted stochastic policies with action
  probabilities (stay, d−, d+, b−, b+):
  - *balanced*: (.1, .25, .25, .2, .2);
  - *skew-d*: (.1, .31, .19, .2, .2). The stationary depth profile is ∝ exp(−1.92 (d − 1)), giving about 85% of mass
    below d = 2, like ICM's native data.
  - *skew-db*: (.1, .31, .19, .24, .16), which also concentrates mass toward b = −3.

  The shared predictor data uses uniform random actions. Data seeds are 600000 + 1000 · condition index + seed; the shared
  set uses 650000 + seed.
- **Occupancy check.** The realized mass below d = 2 is recorded per dataset. These policies are controlled stand-ins for
  ICM's occupancy, not ICM actors.

## Training

- **Encoder.** The original `6 → 48 → 48 → D` tanh MLP with `tanh(4x)` input, D ∈ {1, 2, 3, 4, 8}.
- **Seeds.** Five seeds 0–4. The initialization uses `manual_seed(40000 + seed)` and is shared across D (trunk),
  conditions and arms. Alignment indices are `randint(16384, (4000, 512))` with generator 700000 + seed; regularizer
  indices use generator 800000 + seed. Both are shared across conditions, D and arms.
- **Optimization.** 4,000 steps, the original learning-rate schedule, Adam, and loss 0.9 · MSE alignment +
  0.1 · SIGReg. SIGReg averages the original 1-d statistic over 64 random unit directions per step (generator
  3000000 + seed); at D = 1 the direction is +1.
- **Arms.** *ordinary*, and *weighted*: occupancy-weighted characteristic-function moments with inverse mass over
  16 log-d × 12 b bins of the endpoints. The weights are privileged.

3 conditions × 5 D × 2 arms × 5 seeds = 150 fits.

## Aliasing measure

The grid is 41 depths uniform in log d over [1, 16] × 25 offsets b ∈ {−3, −2.75, …, 3}.

- **Local scale.** ℓ(s) is the smallest embedding displacement among the valid one-move neighbours of s, each evaluated
  at exactly ±0.25.
- **Aliased state.** s is aliased if some grid state s′ more than four moves away (|Δd| + |Δb| > 1.0) has
  ‖f(s′) − f(s)‖ < ℓ(s).
- **Reported.** Aliased native-occupancy mass (endpoints mapped to the nearest grid point in (log d, b)), aliased grid
  fraction, and the minimum far ratio.
- **Aliased fit.** More than 1% of native mass is aliased.

## Downstream (Protocol 1 harness generalized to a 2-d state and 5 actions; the network is evaluated directly)

- **Next-observation decoder.** (D + 5) → 64 → 64 → 6, whitened `tanh(4x′)` targets, 1,500 updates, episodes < 204
  train and the rest held out.
- **Readout.** D → 64 → 64 → 2, targets log d and b; held-out MSE is reported for each.
- **Planning.** Residual predictor (D + 5) → 64 → 64 → D, 1,000 updates, on shared and on native data.
  - The 256 action sequences of horizon 8 include a direct path to every displacement with |i| + |j| ≤ 8 (d-moves, then
    b-moves, then stay); the rest are random (generator 1500000 + seed).
  - Cost: squared Euclidean distance in standardized code space, plus 0.01 × stage cost.
  - 16 control steps.
  - Goals, 64 per set. *Standard*: start d ∈ U[3, 14]. *Near*: start d ∈ U[1, 3.5]. In both, b ∈ U[−2.5, 2.5] and
    offsets k_d, k_b ∈ {−6…6}, not both 0, redrawn until the goal is inside the box.
  - A greedy known-dynamics oracle must reach every goal.
  - Scores: mean Euclidean (d, b) terminal error, and success (both coordinates within one move).
- **Replicates.** Three, with seed offsets 100000 · r.

## Pre-registered predictions

- **S1.** D = 1 is aliased in every fit; a continuous map from a 2-d state to a line cannot be injective. This is a
  calibration check.
- **S2.** D = 2, ordinary, skew-d: aliased in at least 3 of 5 seeds, and in more seeds than under balanced occupancy.
- **S3.** D ≥ 4, ordinary: no aliased fit in any condition.
- **S4.** D = 2, skew-d: weighted has lower aliased mass than ordinary in at least 4 of 5 seeds.
- **S5.** Skewed conditions, D ≥ 3: weighted beats ordinary on standard-goal shared planning error in at least 70% of
  pairs.
- **S6.** Ordinary fits with D ∈ {2, 3}, pooled over conditions: Spearman ρ(aliased mass, native next-obs MSE) > 0.3.

## Limits fixed in advance

- The policies are scripted, not ICM.
- The state is 2-d and the observation is a clean 6-d projection, not images.
- Small D and five seeds.
- The aliasing threshold is a planning-motivated choice.

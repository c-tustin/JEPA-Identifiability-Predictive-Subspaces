# Multi-dimensional SIGReg on ICM occupancy (completed, Nov 26)

**The fold does not survive a second dimension, but the occupancy problem does.** With D ≥ 2 output dimensions,
ordinary SIGReg aliases no states in any of 40 fits. Folding as studied so far is a property of the scalar bottleneck.
The skewed occupancy still costs performance, now by compressing the sparsely visited far region: standard-goal
planning error is 0.53–0.92 under ordinary SIGReg against 0.14–0.30 under occupancy-weighted SIGReg, and weighted is
better in 36 of 40 paired fits.

[Protocol (hash-locked)](PROTOCOL.md) · [Trainer and evaluation](run.py) · [Exploratory spread diagnostic](spread.py) ·
[Summary](output/summary.json) · [By-dimension figure](output/by_dimension.png) · [D = 2 embeddings](output/d2_embeddings.png) ·
[Aliasing versus downstream](output/aliasing_vs_downstream.png) · [Spread diagnostic results](output/exploratory_spread.json)

## Design

Ten ICM sources (130–139) × D ∈ {1, 2, 4, 8, 16} × {ordinary, occupancy-weighted} SIGReg, for 100 fits. Everything
except the output dimension is the original setup:

- the same trunk and initialization seed;
- the historical minibatch streams;
- the same learning-rate schedule and loss weights.

SIGReg in D dimensions averages the original 1-d statistic over 64 random unit directions per step, as LeJEPA does.

**Aliasing measure.** A state is aliased if a state more than four moves away embeds closer to it than its own one-move
neighbour.

**Downstream.** The Protocol 1 harness, generalized to D-dimensional codes.

## Implementation checks

- **D = 1 training:** both arms reproduce the stored 1-d checkpoints bit-for-bit (`initialization-crossing` diagonal
  and `native-weighted_moments`), with maximum parameter difference 0.0 in all 20 fits.
- **D = 1 downstream harness:** it reproduces the Protocol 1 numbers for the five folded sources exactly (maximum
  difference 0.0).
- **Calibration at D = 1:** the aliasing measure flags exactly the five scalar-folded sources (131, 132, 133, 136,
  139), with aliased native mass 0.53–0.96, and none of the five ordered ones.

## Results (means over 10 sources)

| Arm, D | Aliased fits | Next-obs MSE | Near-goal plan error | Standard-goal plan error | Near success |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary, 1 | 5/10 | 0.195 | 0.64 | 0.79 | 63% |
| ordinary, 2 | 0/10 | 0.0014 | 0.12 | 0.54 | 96% |
| ordinary, 4 | 0/10 | 0.0020 | 0.13 | 0.53 | 94% |
| ordinary, 8 | 0/10 | 0.0031 | 0.14 | 0.64 | 93% |
| ordinary, 16 | 0/10 | 0.0057 | 0.14 | 0.92 | 92% |
| weighted, 1 | 0/10 | 0.0014 | 0.07 | 0.13 | 100% |
| weighted, 2 | 0/10 | 0.0002 | 0.10 | 0.14 | 95% |
| weighted, 4 | 0/10 | 0.0001 | 0.12 | 0.17 | 91% |
| weighted, 8 | 0/10 | 0.0001 | 0.12 | 0.23 | 93% |
| weighted, 16 | 0/10 | 0.0002 | 0.15 | 0.30 | 92% |

No aliasing is borderline at D ≥ 2. The closest far pair is still 1.67–2.8 one-move displacements away, while D = 1
folds reach 0.

Paired ordinary versus weighted at D ≥ 2:

| Measure | Weighted better |
| --- | ---: |
| Standard-goal planning | 36/40 |
| Next-observation error | 37/40 |
| Near-goal planning | 22/40 (no consistent difference) |

## What ordinary SIGReg does instead of folding (exploratory, not pre-registered)

At D = 2 the embedding is a nearly straight line ([figure](output/d2_embeddings.png)). Where the scalar encoder bent its
sparse far tail back into the Gaussian band, which created the fold, the D-dimensional encoder has room to avoid that.
It instead stretches the dense near-object region along one direction and leaves the far region compressed:

- **Far compression.** One move at d = 8–16 moves the code by about 1–1.5% of a move at d = 1–2. Weighted SIGReg gives
  8–9%; a code linear in log-distance would give about 12%.
- **Non-Gaussian spread.** About 11% of native data lies outside 3√D. Code norms reach 22–53, against 2–6 for weighted.
  The final regularizer value at D = 2 is 108, against 43 for weighted.

Compressing the far region hurts the standard goals (d = 3–14) and gets worse with D (0.54 → 0.92). Near-object
planning, where the data is, is fine. The skewed occupancy is still costly, but through compression and a non-Gaussian
code rather than aliasing.

## Pre-registered predictions

| Prediction | Outcome |
| --- | --- |
| M1: D = 1 bit-exact | **Held** (20/20) |
| M2: D = 2 ordinary aliased in ≥ 5/10 | **Failed** (0/10) |
| M3: fewer aliased fits at D = 16 than at D = 2 | **Failed**. Both are 0, so there is no decrease to see. |
| M4: aliasing correlates with downstream error at D ≥ 2 | **Not testable.** Aliased mass is 0 in every fit. |
| M5: weighting reduces aliasing | **Not testable.** No pair is aliased. |

I expected a curve forced toward an isotropic Gaussian to wind and cross itself, at least at D = 2. It did not: the
alignment term and the network's smooth bias keep the curve nearly straight.

## Limits

- **One-dimensional state.** For D > 1 the embedding is a curve in a larger space. Real observations have
  higher-dimensional state, where SIGReg's pressure to fill all directions may interact with occupancy differently.
  Aliasing between different state dimensions is untested.
- **Small scale.** 64 projections, D ≤ 16, one initialization per source, and the original small MLP.
- **Threshold choice.** The four-moves-versus-one-move rule is a planning-motivated choice; the minimum-ratio column
  shows the conclusion is not near its edge.
- **Privileged weights.** The weighted arm uses distance bins.

## Reproduce

From the workspace root, about 20 minutes on 10 CPU workers:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/multidim-sigreg/run.py --workers 10
python3 icm-folding/multidim-sigreg/spread.py
```

Checkpoints in `output/runs/` stay local. The D = 1 checks also need the initialization-crossing and occupancy-folding
checkpoints.

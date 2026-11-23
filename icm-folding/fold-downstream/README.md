# Does the fold itself hurt downstream use?

**Yes, substantially, in this world.** Holding the data, predictor networks, predictor seeds and goals fixed, removing
only the fold from each of the 51 materially folded SIGReg encoders improves every downstream measure for 50 of them. The
exception is a fold at d ∈ [15.1, 16], where there is no data. Native next-observation error falls about 100-fold.
Latent-MPC planning error near the object falls from about 1.1 to about 0.09, with success rising from about 30% to
about 99%.

[Protocol (hash-locked before any run)](PROTOCOL.md) · [Code](run.py) · [Per-dataset and per-encoder summary](output/summary.json) ·
[Scatter figure](output/downstream.png) · [Rearrangement examples](output/rearrangement_examples.png) · [Per-cell records](output/cells)

## Design

The encoders are the 55 initialization-crossing encoders with a final slope reversal; 51 of them are material folds,
spread over datasets 131, 132, 133, 136 and 139 (10 each) and 134 (1).

For each one, the counterfactual is its **monotone rearrangement** over a 20,001-point uniform log-distance grid. The
code values are sorted into the encoder's dominant direction, so the rearranged code uses exactly the same multiset of
values, but every fold is gone (multiset equality was checked for all 55). Both arms go through the same interpolating
wrapper (maximum difference from the network: 4.2e-6).

The wrapper reads d from the observation (exact here, d = 1/x₁). This makes the rearranged arm an
**evaluation-only oracle counterfactual**: it shows what the fold costs, not how to train an encoder without it.

Each arm is scored on three downstream measures. Each fit is repeated with three predictor seeds, identical across arms.

- Next-observation decoder: the `mlp-gradient-study` recipe, trained on native trajectories and scored on held-out
  trajectories.
- A log-distance readout from the code.
- The original latent MPC planner, scored on two goal sets:
  - *standard* goals: starts at d ∈ [3, 14], as in the original study;
  - *near* goals: starts at d ∈ [1, 3.5], where ICM data and the folds are concentrated.

## Results (dataset means over material-fold encoders, original → fold removed)

| Dataset | Native mass moved | Next-obs MSE (native) | log-d readout MSE | Plan error, standard, shared data | Plan error, near, shared data | Near success, shared |
| --- | ---: | --- | --- | --- | --- | --- |
| 131 | 0.97 | 0.670 → 0.007 | 0.216 → 0.001 | 0.83 → 0.15 | 1.08 → 0.07 | 29% → 100% |
| 132 | 0.94 | 0.147 → 0.002 | 0.059 → 0.0005 | 0.29 → 0.10 | 1.03 → 0.09 | 41% → 100% |
| 133 | 0.96 | 0.465 → 0.003 | 0.130 → 0.0005 | 0.26 → 0.08 | 1.26 → 0.06 | 35% → 100% |
| 134 (n=1) | 0.00 | 0.0018 → 0.0018 | 0.0004 → 0.0004 | 1.48 → 1.45 | 0.12 → 0.12 | 92% → 92% |
| 136 | 0.40 | 0.307 → 0.0007 | 0.168 → 0.001 | 1.72 → 0.36 | 0.99 → 0.16 | 30% → 87% |
| 139 | 0.97 | 0.339 → 0.004 | 0.102 → 0.0006 | 0.62 → 0.11 | 1.20 → 0.08 | 24% → 100% |

Encoders improved, out of 51:

| Measure | Encoders improved |
| --- | ---: |
| Native next-observation error | 50 |
| Random-policy next-observation error | 49 |
| Readout | 50 |
| Planning, shared data, standard goals | 51 |
| Planning, shared data, near goals | 50 (one tie) |
| Planning, native data, standard goals | 51 |
| Planning, native data, near goals | 50 |

The replicate noise (seed SD of the original arm) is 0.007 for next-observation MSE and 0.10–0.23 for planning
error, much smaller than these changes. Planning on ICM's own action data remains worse than on the shared random-policy
data even after rearrangement (standard goals 0.49–0.89), which is the separate action-coverage problem documented in
the qualitative study.

The four tiny (non-material) folds change nothing measurable: their rearrangement moves almost no grid points.

## Pre-registered predictions

| Prediction | Outcome |
| --- | --- |
| H1: next-obs MSE falls in ≥ 5/6 datasets | **Held**, 5/6. The miss is 134, where the fold is outside the data. |
| H2: standard-goal planning changes by < 0.05 for 131/132/133/139; 136 improves by > 0.2 | **Failed.** 136 held (−1.35), but the other four improved by 0.19–0.68. |
| H3: near-goal planning improves in ≥ 5/6 datasets | **Held**, 5/6. |
| H4: readout improves in ≥ 5/6 datasets | **Held**, 5/6. |

**Why H2 failed, and what it corrects.** The earlier qualitative report measured a fold by its reversed-slope span
(for example 1.00–2.37 for seed 131, crossed by 1.2% of standard routes). On that basis it concluded the folds sat
outside the task range. That understates the damage. The reversed branch's code values collide with ordinary states much
farther out: removing the fold reassigns codes up to d ≈ 6 for seed 131. For datasets 131, 132, 133 and 139, an average of 25–73% of
standard-route positions lie in the affected set (100% for 136). The earlier cross-seed observation that folded seeds planned better than some ordered
ones does not survive a matched comparison. Seed-level differences other than the fold dominated that comparison.

## Limits

- **Oracle counterfactual.** It shows what the fold costs, not that any trainable method recovers this performance.
  Trained repairs (weighted moments, mining, hard order constraints) remain separate comparisons that change more than the
  fold.
- **Rearrangement changes more than order.** Removing a fold necessarily changes local slopes in the affected range.
  Sometimes it places the folded branch's extreme values at the far end of the range (dataset 139: a steep, ordered
  segment at d ≈ 13–16). It does not separate non-injectivity from local compression.
- **Narrow setting.** One deterministic 1-d world, one renderer, scalar codes, fixed predictor budgets.
- **Dependent encoders.** Encoders within a dataset share their data, so datasets (6) are the independent unit.
- **These are empirical comparisons**, not analytic guarantees.

## Reproduce

From the workspace root, about 7 minutes on 10 CPU workers:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/fold-downstream/run.py --workers 10
python3 icm-folding/fold-downstream/run.py --report-only
```

The runner needs the initialization-crossing checkpoints (kept local, regenerable with that study's `run.py`) and the
qualitative-study sources.

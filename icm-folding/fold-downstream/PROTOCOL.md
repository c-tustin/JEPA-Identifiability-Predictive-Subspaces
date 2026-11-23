# Protocol: does the fold itself hurt downstream use?

Fixed November 23, 2026, before any downstream fit in this directory. The SHA-256 of this file is recorded in
`output/protocol.sha256` before the first run; later changes are listed in an addendum, never edited in place.

## Question

Earlier comparisons never isolated the fold. Across seeds, folded ICM encoders (131–133) planned *better* than some
ordered ones (130, 138), and the mining repair changed the training objective as well as fold status. Here the data,
predictor architecture, predictor seeds and goals are held fixed, and only the fold is removed.

## Encoders

The 55 initialization-crossing encoders (`../initialization-crossing/output/runs/`) with a detected final slope
reversal; 51 meet the material threshold (backward variation fraction > 0.001). Primary analyses use the 51 material
folds; the four tiny folds are reported separately. Dataset is the independent unit: material folds occur in datasets
131, 132, 133, 136, 139 (10 each) and 134 (1).

## Counterfactual: monotone rearrangement

Evaluate the trained network in float64 on a uniform grid of 20,001 points in q = log d over [0, log 16]; call the
codes y. Let s = sign(corr(y, q)). The rearranged code is ỹ = s · sort(s · y), assigned to the grid in increasing q
(the increasing rearrangement of Chernozhukov, Fernández-Val and Galichon, 2009). ỹ takes exactly the same multiset of
grid values as y, is monotone, equals y wherever y was already monotone and not overlapped by a reversed branch, and
removes every fold.

Both arms are evaluated through the same interpolating wrapper: observation x → d = 1/x₁ (exact for this renderer) →
linear interpolation of y or ỹ in q. The original arm's wrapper is checked against the network (maximum absolute
difference reported). The wrapper reads distance from the observation, so the rearranged arm is an **evaluation-only
oracle counterfactual**, not a trainable repair; it answers "what would downstream use look like if this encoder's code
values were ordered", not "how to obtain such an encoder".

Diagnostic: the *moved set* is the grid points where |ỹ − y| > 0.001 · range(y). Report its native-occupancy mass and
its share of oracle planning-route positions.

## Downstream measures (frozen code; no distance label enters any predictor input)

1. **Next-observation prediction** (recipe of `mlp-gradient-study/run.py::downstream`). Decoder 4→64→64→6 (ELU),
   input standardized code + one-hot action, target whitened `tanh(4·x')`, Adam 0.001, 1,500 updates, batch 256.
   Train on native trajectories with episode index < 204; score native held-out episodes (≥ 204) and the dataset's
   `physical-common` random-policy set.
2. **State readout.** Same architecture with a 1-d output, input code only, target log d, trained on native training
   episodes; score native held-out and the uniform 1,024-point log-d grid. Labels are used only for this evaluation.
3. **Latent MPC planning** (`planning/evaluate_planning.py`: residual code predictor, 1,000 updates, 128 × 8 action bank,
   12 steps). Predictor data: shared (`physical-common`) and native ICM pairs. Goal sets, 64 cases each:
   - *standard*: the original `goals(c, 10000 + seed, 64)` (start ∈ [3, 14], ±1–8 moves);
   - *near*: start ∈ [1, 3.5] uniform, goal = start + 0.25 k with k uniform in {−8…8}\{0}, redrawn until goal ≥ 1
     (generator seed 1600000 + seed). The known-dynamics oracle must reach every goal.
   Score: mean terminal |distance − goal| and success within one move.

Each measure is fitted with three predictor/decoder seed replicates r = 0, 1, 2 (seed offsets 100000 · r), identical
across arms. Replicate spread is the noise yardstick.

## Pre-registered predictions

- **H1.** Rearrangement lowers native held-out next-observation MSE: dataset-mean ratio (rearranged / original) < 1 in
  at least 5 of the 6 datasets with material folds.
- **H2.** Standard-goal shared-data planning: |dataset-mean change| < 0.05 for 131, 132, 133 and 139 (folds sit below
  the task range); 136 improves by more than 0.2.
- **H3.** Near-goal shared-data planning: rearrangement lowers dataset-mean error in at least 5 of 6 datasets.
- **H4.** State-readout native held-out MSE is lower after rearrangement in at least 5 of 6 datasets.

Encoder-level results are reported with all values; inference uses datasets. A null on H1/H3/H4 would mean the fold is
mostly cosmetic for these uses; that outcome is reported the same way.

## Limits fixed in advance

One deterministic 1-d world, one renderer, scalar codes, frozen encoders, finite predictor budgets. The rearrangement
also changes local slopes inside the moved set (it must, to remove the fold); it does not separate "non-injective" from
"locally compressed". No trained repair is evaluated here.

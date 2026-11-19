# Recovering training and error convergence

Added November 19, 2026 after inspecting the completed November 18 runs. The original training protocol and checkpoints remain unchanged. Original files recorded geometry and loss sparsely, and downstream prediction errors only after the decoder fit. Dense histories are recovered by deterministic replay, not interpolated from six measurements.

`convergence.py` imports the original trainer, source loader, initialization and sampling plan. It replays each 4,000-update encoder/critic fit, records every minibatch objective, and evaluates fixed training and held-out candidate sets every 25 updates. Each evaluation set has 256 objects (128 current and 128 successor observations); candidate identities stay fixed through training. Raw-branch means/stds are fitted on the training endpoints only. Fixed training evaluations are diagnostic subsets, not the optimizer minibatch. Neither evaluation nor curve selection changes optimization.

Encoder-training panels show InfoNCE cross-entropy and the scorer's observation reconstruction MSE in standardized raw-sensor units. The latter is a diagnostic of `g(f(x))`, not an additional optimized reconstruction loss and not next-observation prediction error. Both metrics use the same fixed candidate objects at each measurement. Error bars or bands across seeds represent descriptive between-run variation, not uncertainty from independent new datasets.

Next, freeze the final encoder and replay its separate 1,500-update next-observation decoder fit. Evaluate complete training, held-out trajectory and coverage-stress datasets every 50 decoder updates. These MSE targets are whitened post-tanh successor observations, as in the original downstream protocol. Decoder updates have their own axis and are not encoder training updates. Prediction error during intermediate encoder checkpoints is not measured by this procedure.

Each replay must reproduce the saved final encoder, critic and downstream decoder tensors bit-for-bit, the six original recorded training losses exactly, and the original final full-dataset downstream errors exactly. Store compressed numeric histories and JSON provenance/hash audits separately under `output/convergence/`; do not overwrite original checkpoints. Fail if a deterministic identity check fails.

Representative sources are chosen mechanically by final held-out next-observation prediction MSE among all 15 raw/code runs: minimum, median (8th sorted source), and maximum, ties broken by seed. Show their saved initial/intermediate/final scalar code curves. Also show final SIGReg, scalar VICReg, temporal InfoNCE and raw/code curves for those same sources. A separate comparison panel uses seed 131, the previously discussed folded SIGReg example. Center/scale and sign-orient codes only for the cross-objective shape comparison, label this transformation, and retain original code units in the training snapshots. Physical distance is an evaluation coordinate only.

Run from the workspace/repository root after the original checkpoints and source data are available:

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/raw-embedding-infonce/convergence.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/raw-embedding-infonce/plot_convergence.py
```

Completed all 15 replays: final encoder, scorer and downstream predictor tensors match the original saved states bit-for-bit, and all original recorded losses/final errors match exactly. [Measurements and interpretation](output/convergence/RESULTS.md), [numeric summary](output/convergence/summary.json), [replay verification](output/convergence/verification.json).

The scorer's raw MSE falls early and then grows. This is not a reconstruction objective: the observations lie on a one-dimensional affine line, and a scorer component perpendicular to that line contributes only a common candidate-independent score shift, which cancels in the contrastive softmax. The final scorer audit decomposes error parallel/perpendicular to the observed line and checks the effect of removing the perpendicular component on candidate log probabilities. Downstream prediction errors use a separate calibrated decoder and do fall during its fit.

Figures: [encoder training](output/training_convergence.png), [prediction convergence](output/prediction_convergence.png), [representative curves](output/representative_representations.png), [seed-131 comparison](output/seed131_comparison.png). PNGs and vector SVGs are provided. Each source has a compressed numeric history and JSON provenance audit under `output/convergence/`. Finite-grid curves and derivative checks are empirical and do not prove injectivity.

# InfoNCE: raw observation versus its own embedding

This study trains the original scalar encoder with positive pairs `(x, f(x))` and mismatched raw observations as negatives. It is separate from the earlier temporal InfoNCE study, whose positives were `(f(x_t), f(x_{t+1}))`.

A learned 1→48→48→6 network maps the scalar code into standardized raw-sensor units. Its negative squared distance to each candidate raw observation defines the compatibility score. The raw observation branch is fixed and unembedded. The objective is contrastive cross-entropy, without an additional reconstruction loss, temporal alignment or anti-collapse regularizer. This choice is a restricted critic; the experiment does not cover all raw/code InfoNCE critics.

[Fixed protocol](PROTOCOL.md) · [Training and formula checks](run.py) · [Independent verification/report](report.py)

Run from the workspace/repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/raw-embedding-infonce/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/raw-embedding-infonce/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/raw-embedding-infonce/report.py
```

The trainer reuses the source loader, encoder, geometry and downstream prediction code from [the original alternative-objective study](../mlp-gradient-study/README.md). It imports those sources without duplicating checkpoints or modifying their output paths. The earlier study's baseline checkpoints and five follow-up datasets must exist before reporting. `--seeds` selects disjoint subsets for concurrent processes; `--steps` is for explicitly separate smoke runs. Main results always use 4,000 updates.

All 15 fits completed: **0/10 historical and 0/5 reused follow-up sources have a detected final fold**. Minimum output standard deviation is 1.525, so these encoders have not collapsed. Mean held-out next-observation prediction MSE is 0.001102 and 0.001468 respectively (temporal InfoNCE: 0.002071 and 0.002560). Coverage-stress errors are mixed across objectives; this is not a uniform performance ranking.

[Results and per-seed measurements](output/RESULTS.md) · [Summary](output/summary.json) · [Independent verification](output/verification.json) · [Formula checks](output/formula-checks.json) · [All final curves](output/curves.png)

All 15 final encoders were independently reloaded and checked on 20,001-point float64 derivative grids; source/batch hashes, initial weights, raw statistics and saved decoder errors were verified. New checkpoints are included in this study's output folder. The five follow-up datasets are reused, not newly collected validation. Differences in critic, candidate count and compute mean comparisons with the temporal run are descriptive. All fold checks are empirical, not analytic guarantees.

## November 19: Training and error convergence

[Encoder-training curves](output/training_convergence.png) · [Prediction-error curves](output/prediction_convergence.png) · [Representative representations](output/representative_representations.png) · [Seed-131 comparison](output/seed131_comparison.png)

[Method and reproduction](CONVERGENCE.md) · [Measurements](output/convergence/RESULTS.md) · [Replay verification](output/convergence/verification.json). Dense histories were recovered by replay: all 15 encoders, scorers and downstream predictors reproduce the saved weights bit-for-bit. Encoder/critic evaluations occur every 25 updates; separate frozen-encoder decoder errors are evaluated on whole datasets every 50 decoder updates. The mean final held-out decoder MSE across all 15 sources is 0.001224 (median 0.001148).

The scorer's raw MSE falls early and then increases; this is not a reconstruction-trained network. At least 99.927% of final held-out scorer MSE is perpendicular to the observed raw-data line in every source. Removing that component changes log probabilities by at most 1.02e-6, consistent with cancellation of candidate-independent score shifts. The prediction-error plots use a separate calibrated decoder. Representative seeds 130, 144 and 141 have the lowest, median and highest final held-out prediction errors, respectively. Cross-objective shape comparisons are centered, scaled and sign oriented for display; learning snapshots retain their original units. PNG and vector SVG versions are included.

## Reading the distance-representation graphs

Teal in objective-comparison plots denotes the latest raw-observation/own-code InfoNCE method. The horizontal axis is physical distance, used only for evaluation; the vertical axis is the learned scalar representation of that distance, not a distance estimate in physical units. Original code values have unconstrained scale, offset and sign (the final encoder layer is linear). Values around −6 or +6 are learned code units, not a prescribed range. Comparison panels explicitly center, scale to unit standard deviation and sign-orient each curve for display. Such transformations preserve folds. The flattening of a curve indicates compressed resolution at those distances. Figure labels were clarified November 19; model weights and numerical results are unchanged.

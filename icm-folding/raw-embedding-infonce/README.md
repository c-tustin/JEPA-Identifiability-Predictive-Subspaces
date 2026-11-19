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

## Latest InfoNCE only: distance-representation figures

These figures contain only the raw-observation/own-code InfoNCE experiment, with no alternative objectives. All representations use original code units without centering, rescaling or sign flips. Physical distance is an evaluation coordinate, not a training label.

- [All 15 final distance representations](output/latest_distance_representations.png).
- [Seed 131: distance representation and its physical-distance derivative](output/latest_seed131_distance.png).
- [Distance representations during learning](output/latest_distance_learning.png), for the lowest/median/highest saved final held-out decoder errors (seeds 130/144/141).
- [Plotting source](plot_latest.py) and [checkpoint/derivative audit](output/latest_plots_verification.json). Matching SVG files are available beside each PNG.

Reproduce with `OPENBLAS_NUM_THREADS=1 python3 icm-folding/raw-embedding-infonce/plot_latest.py` from the workspace root. This reloads only this method's saved checkpoints, checks their hashes against the existing independent report, recomputes all final geometries on 20,001 float64 points, and checks the chain rule against finite differences. The stored derivative is with respect to log-distance; the displayed physical-distance derivative divides it by distance. No training is performed.

## Can InfoNCE guarantee no folding?

Not for the current unconstrained MLP and finite-sample training procedure. InfoNCE classifies the matching observation among candidate observations; it does not impose a derivative-sign constraint over the physical interval. Identical codes for distinct co-occurring observations give identical candidate-score rows and impede their joint identification, but discouraging such collisions is not a proof that none exist. Finite sampled observations also leave intervals without direct checks. The current result remains zero detected final folds in 15 runs, not an all-input or all-training-step guarantee. The original [InfoNCE formulation](https://arxiv.org/abs/1807.03748) defines the contrastive classification objective; the missing interval constraint follows directly from our implementation.

For this scalar distance representation, a sufficient guarantee is a certified fixed sign with positive margin for the derivative of `h(d)=f(sensor(d))` over the entire stated interval: `h'(d) >= m > 0`, or `h'(d) <= -m < 0`. This would require a validated bound, not a sampled minimum. Alternatively, train InfoNCE with a strictly monotone encoder using a known ordered coordinate, such as the [existing constrained constructions](../no-fold-proof/README.md); the guarantee would come from that constraint, not from InfoNCE. No such InfoNCE fit or certificate has been implemented here. Exact reconstruction of every distinguishable observation in an interval would also imply injectivity, but our contrastive scorer is not trained or verified to provide it.

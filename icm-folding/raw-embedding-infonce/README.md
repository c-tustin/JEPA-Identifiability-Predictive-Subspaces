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

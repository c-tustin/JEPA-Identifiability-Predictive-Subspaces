# Historical policy data × encoder initialization

Status: all 100 fits completed. All ten diagonal replays exactly reproduce the historical weights. Final folds: 55/100; material folds: 51/100. See [completed results](RESULTS.md). See the [fixed protocol](PROTOCOL.md) and [trainer](run.py).

The full design is ten saved policy datasets × ten original encoder initializations, with both minibatch streams fixed within dataset. Diagonal fits replay the original models exactly as a required implementation check. This separates initialization sensitivity from the bundled dataset/sampling condition; it does not separate actor training from rollout collection or minibatch variation.

The trainer writes resumable partials, completed checkpoints, 100-update geometry histories, independently reloaded dense final geometry and an incremental machine-readable summary. That summary distinguishes completed cells from the planned 100. The completed summary contains all 100 cells.

From the workspace root, using existing numpy/torch/matplotlib dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/initialization-crossing/run.py
python3 icm-folding/initialization-crossing/run.py --report-only
```

Original source datasets remain in the [linked qualitative study](../earlier-qualitative-study/output/sources/); no source checkpoint copies are created.

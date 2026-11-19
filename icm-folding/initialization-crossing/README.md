# Historical policy data × encoder initialization

Status, November 19, 2026: batch launched; complete results pending. The first full diagonal replay (130,130) reproduced the historical final weights bit-for-bit. See the [fixed protocol](PROTOCOL.md) and [trainer](run.py).

The full design is ten saved policy datasets × ten original encoder initializations, with both minibatch streams fixed within dataset. Diagonal fits replay the original models exactly as a required implementation check. This separates initialization sensitivity from the bundled dataset/sampling condition; it does not separate actor training from rollout collection or minibatch variation.

The trainer writes resumable partials, completed checkpoints, 100-update geometry histories, independently reloaded dense final geometry and an incremental machine-readable summary. That summary distinguishes completed cells from the planned 100. No claim about the crossed outcomes is made before completion.

From the workspace root, using existing numpy/torch/matplotlib dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/initialization-crossing/run.py
python3 icm-folding/initialization-crossing/run.py --report-only
```

Original source datasets remain in the [linked qualitative study](../earlier-qualitative-study/output/sources/); no source checkpoint copies are created.

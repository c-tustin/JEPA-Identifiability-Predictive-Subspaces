# Isolating saved data from minibatch sampling

Status, November 19, 2026: validation passed and 500-fit batch launched; complete outcomes pending. Both original full-run replay checks and exact model/Adam resume checks passed.

[Fixed protocol](PROTOCOL.md): ten saved ICM datasets × ten original scalar-encoder initializations × five independently generated batch plans (500 fits). For a given plan, every dataset and initialization receives identical alignment and regularizer index streams. This removes the old association between dataset identity and its batch-plan seed.

[Trainer](run.py) imports the canonical encoder, optimizer schedule and fold diagnostics through the [earlier initialization crossing](../initialization-crossing/README.md), preserving original datasets and checkpoints. It records hashes and resumable checkpoints. [Report](report.py) audits reloaded geometry and pairing and summarizes dataset, initialization, sampling and interaction patterns without treating crossed cells as independent policy sources.

From the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/sampling-crossing/run.py --validate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/sampling-crossing/launch.py --workers 4
python3 icm-folding/sampling-crossing/report.py
```

The launcher runs bounded local worker processes, writes progress after completed fits, and runs the full audit/report after successful completion. Each worker uses one torch thread. No new policy training or rollout collection is included; those remain distinct sources of dataset variation.

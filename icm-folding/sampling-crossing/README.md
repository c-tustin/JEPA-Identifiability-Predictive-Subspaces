# Isolating saved data from minibatch sampling

Status, November 19, 2026: all 500 fits completed and independently audited. Final turns: 270/500, of which 256/500 are material. See [completed results](RESULTS.md), [fold matrices](output/sampling_crossing.png), and [verification](output/verification.json). Across the five fresh streams, 0/100 fixed dataset/initialization pairs change binary fold status; 1/100 changes material-fold status.

[Fixed protocol](PROTOCOL.md): ten saved ICM datasets × ten original scalar-encoder initializations × five independently generated batch plans (500 fits). For a given plan, every dataset and initialization receives identical alignment and regularizer index streams. This removes the old association between dataset identity and its batch-plan seed.

[Trainer](run.py) imports the canonical encoder, optimizer schedule and fold diagnostics through the [earlier initialization crossing](../initialization-crossing/README.md), preserving original datasets and checkpoints. It records hashes and resumable checkpoints. [Report](report.py) audits reloaded geometry and pairing and summarizes dataset, initialization, sampling and interaction patterns without treating crossed cells as independent policy sources.

From the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/sampling-crossing/run.py --validate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/sampling-crossing/launch.py --workers 4
python3 icm-folding/sampling-crossing/report.py
```

The launcher runs bounded local worker processes, writes progress after completed fits, and runs the full audit/report after successful completion. Each worker uses one torch thread. No new policy training or rollout collection is included; those remain distinct sources of dataset variation.

The 500 derived checkpoints remain local; the published summary and audit include complete cell measurements and checkpoint hashes. Use the commands above to regenerate them. Published audits describe the original local run; rerun `--validate` to record dependency hashes for the portable source layout before regeneration. The historical extra tiny turn is discussed separately in the results.

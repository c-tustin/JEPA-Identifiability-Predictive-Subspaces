# Reproduction run status

2026-10-26: **blocked before execution; zero original-paper models trained.**

The selected cloud environment reports environment_offline (HTTP 409). Initial pwd failed with "No usable execution environment is available." After independent source auditing, one retry of pwd and python --version failed identically. No further shell retries were made. CPU/RAM/disk/GPU, current packages and data availability are unknown. Earlier workspace existence is not current compute access.

## Completed

- Retrieved pinned upstream code/configuration text and tracked Git tree through the GitHub app.
- Audited runner, downloader, configs, models/probes/training/corpus/artifact/device code and plots against public paper v1 Appendix B–E. Local paper.txt is inaccessible; no new local PDF read is claimed.
- Checked static accounting of 77 distinct main settings and fixed the seed/baseline policy. The Python planner has not run.
- Found no released trained metrics/checkpoints in the pinned tree or the empty GitHub releases list.
- Prepared source/data hash checks, isolated setup, official-trainer smoke, exact full-budget scheduling, aggregation and standard-library protocol tests.
- Kept published Table 1 targets solely in AUTHOR_REFERENCE.json, not as generated results.

## Not completed

No hardware audit, environment creation, package installation, CIFAR download/hash verification, smoke/timing, training, probes, checkpoints, figures, Python syntax check or unit-test execution has occurred. README commands are prepared instructions, not an execution log.

run-manifest.json contains 77 pending runs, empty checkpoint/log lists and null timings. Expected data hashes have not been checked against downloaded bytes. No hours/GPU requirements estimate is fabricated.

## Decisions and gaps

Use a new isolated Python 3.12/Torch 2.11.0/NumPy 2.4.3/Matplotlib 3.10.8 environment. Do not modify PR #4's earlier runtime. Official profile is CUDA 12.8 BF16; CPU FP32 is an explicit recorded variant.

Seed 303 is upstream default/primary; 304/305 are fixed reproduction choices. Original additional optimization-seed IDs are not specified in Appendix B. Data/pool/probe seeds stay upstream. Budget: 39+19+19 settings; one reconstruction seed.

Probe architecture matches the paper despite metadata [16]. Appendix C evaluator remains unimplemented. Main mean/SEM aggregation is additional presentation code, unexecuted and untested here. Default regularization sensitivity is outside the first main-only suite.

## Resume

Reconnect compute; run lightweight planner/tests and hardware audit; fetch/hash CIFAR; run the labelled four-update smoke; inspect actual timing/memory/failures; launch a full-budget primary traffic sweep and then the remaining main suite as practical. No paid infrastructure is authorized. Verify completion receipts, aggregate full seeds, compare paper curves/seed spread and implement Appendix C separately.

This branch is preparation plus a blocker record, not reproduced results. PR #4 and its distance-world outputs are untouched.

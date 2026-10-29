# Reproduction run status

2026-10-26: **local training paused; no completed paper result yet.**

The trainer and both launchers were confirmed stopped with SIGSTOP. Logs/configs and hashes are archived in `runs-local/pause-logs-and-configs.tar.gz`. The existing full model's weights are still **only in memory** and will not survive a computer restart. See [pause evidence](pause-evidence.json).

Restart-safe checkpointing has now been added and tested: a separate-process resume test matched uninterrupted weights, optimizer state, RNG states and histories exactly in a small test. Seven planner/runtime checks pass. The new adapter cannot attach to the old suspended process. A checkpoint-enabled full model would need to start again; none has been started, honoring the pause request. [Instructions and limits](RESTART.md).

The cloud outage was bypassed after the user authorized local execution. Source verification and six planner/runtime-guard tests pass. CIFAR-10 passed the official archive MD5 and all extracted-file SHA-256 checks. Python files compile.

The background worker started at **16:32:15 UTC**. The four-update official-trainer timing check completed successfully in **86.0 seconds**, including startup, evaluation and checkpoint I/O. Architecture, batch 512 and 1,024 SIGReg projections were retained. This short check is not a scientific result or a steady-state timing estimate.

**Full training started at 16:33:45 UTC and is now paused.** The q=0, seed-303 model logged its first finite-loss update toward the full 25 epochs/4,900 updates. Later progress was between log intervals, so its exact update is unknown. The queued q=1 and remaining traffic runs are paused too. The other 62 main-paper encoders remain pending. No full model is complete yet.

**Local variant:** Intel Mac, 6 physical cores / 12 logical CPUs, 16 GB RAM, no available PyTorch GPU backend. Python 3.12.3, Torch 2.2.2, NumPy 1.26.4, Matplotlib 3.9.0; CPU FP32. Explicit flags distinguish it from the official CUDA/BF16 environment. Model/loss/data source and full training/probe budgets stay unchanged. The virtual environment reuses installed packages without upgrading them.

Live evidence is in ignored `runs-local/`: `local-process.json` (stage/PID), `runtime-audit.json` (runtime/source hashes), `data-verification.json`, and separate smoke/full configs and logs. Completed stages will have official receipts and checkpoint/result hashes. `local-evidence.json` is a dated public snapshot, not a live feed. No paper accuracy or full-run timing estimate is claimed yet.

## Remaining scope

The full main protocol contains 77 distinct encoders. Seed 303 is upstream's default; 304/305 are our fixed reproduction choices. Reconstruction uses one seed and reuses the beta=0 controls. Other studies and the regularization sensitivity sweeps have not started.

Appendix C's incompatible-alternative prediction evaluator remains unimplemented. Traffic interventions and latent prediction MSE are different diagnostics. Published values in AUTHOR_REFERENCE.json are author reference only. No original paper accuracy has been reproduced yet.

This branch uses the original CIFAR collage environment.

## Standalone branch layout

This branch is now based directly on main. Code lives in the top-level predictive-closure-reproduction/ folder. Earlier paused processes and their runtime files remain in the original local checkout; none were moved, restarted or deleted during this reorganization. Recorded execution hashes describe those earlier runs.

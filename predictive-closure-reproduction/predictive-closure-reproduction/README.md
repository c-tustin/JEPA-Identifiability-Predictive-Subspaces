# Original Predictive Closure reproduction

**Local training is paused at the user's request; no paper result is complete yet.** Its current weights remain in memory only. Restart-safe checkpointing is now implemented and tested, but using it requires restarting this model. [Pause/restart instructions](RESTART.md) · [Status](RUN_STATUS.md) · [Earlier execution evidence](local-evidence.json) · [Coverage](figure-manifest.json).

This branch contains the original-paper CIFAR collage reproduction only. It uses unmodified original CIFAR collage code at **2964fa8f737fa43871d47928edc9aac16fa9298f**. All tracked upstream files have pinned Git blob hashes in [source-manifest.json](source-manifest.json); runtime audit will record SHA-256 hashes. No source is vendored. Upstream is MIT licensed.

## Main protocol

| Study | Grid | Optimization seeds | Encoders |
| --- | --- | --- | ---: |
| Traffic | q = 0, .25, .5, .75, 1 | 303, 304, 305 | 15 |
| Pair switching | pD = .60, .75, .85, .95, 1 | 303, 304, 305 | 15 |
| Group size | N = 2, 4, 8, 16 | 303, 304, 305 | 12 |
| Reconstruction | Five pD × beta = .0425, .105, .135, .36 | 303 | 20 |
| Side information | Five pD | 303, 304, 305 | 15 |

Total: **77**, or 553,700 encoder updates before probes. Reconstruction beta=0 reuses pair-switching controls. Seed 303 is upstream's default and our primary run. Seeds 304/305 are fixed reproduction choices, not claimed to be the authors' undisclosed exact seeds. No tuning after viewing replications; source data-generation/pool/probe seeds stay unchanged.

Unfiltered upstream --main-only gives 39 encoders per seed and needlessly repeats reconstruction across replications. Default upstream also adds regularization r=.05 and .20. This harness selects main r=.10 and calls official jobs/save_plan/execute without replacing models, loss, optimizers, probes or corpus code.

Full budgets remain batch 512, eight-frame sequences, base channels 32, projection width 1024, latent 3 and SIGReg 1024 projections. Traffic/pairs/steering use 25 epochs or 4,900 updates; group size uses 100 epochs/19,600 updates. Image probes retain 1,000 updates; traffic probes retain 1,200.

## Run

### Local Intel Mac

The local machine has 6 physical cores, 12 logical CPUs, 16 GB RAM and no available PyTorch GPU backend. Its recorded runtime is Python 3.12.3, Torch 2.2.2, NumPy 1.26.4 and Matplotlib 3.9.0. This is an explicit **CPU/FP32 runtime variant**, not the authors' CUDA/BF16 environment. Architecture, batch size, losses, training steps and probe budgets stay unchanged. These results will be reported separately from an exact-environment reproduction.

With the pinned checkout available, run from this folder:

~~~bash
/usr/local/bin/python3 -m venv --system-site-packages .venv-intel
.venv-intel/bin/python download_parallel.py --upstream /absolute/path/to/predictive-closure-upstream
.venv-intel/bin/python run_local.py --upstream /absolute/path/to/predictive-closure-upstream
~~~

The environment reuses installed packages, and the launcher checks the versions in `requirements-intel.txt`. Downloading uses official-server byte ranges and verifies the original archive MD5 plus all extracted-file SHA-256 hashes. A four-update smoke test must succeed before full training starts. The queue runs q=0 and q=1 for seed 303 first, then completes all five traffic-light values for seeds 303–305, reusing completed stages. Other paper experiments remain pending.

`runs-local/local-process.json` records the active stage and child PID. Launcher logs, upstream training logs, timing, checkpoints and verification receipts stay in `runs-local/`. Upstream saves the training checkpoint only when a model finishes; an interrupted incomplete model cannot resume from its last update. No completed reproduction is inferred from an active process.

### Authors' CUDA environment

From repository root. Bootstrap creates a separate checkout; to reuse the prior one, override PC_UPSTREAM with /workspace/scratch/339c04b91302/predictive-closure-upstream.

~~~bash
export PC_REPRO="$PWD/predictive-closure-reproduction"
export PC_UPSTREAM="$PC_REPRO/upstream"
bash "$PC_REPRO/bootstrap.sh" "$PC_UPSTREAM"
export PC_PYTHON="$PC_REPRO/.venv/bin/python"
"$PC_PYTHON" "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode plan
"$PC_PYTHON" "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode audit
"$PC_PYTHON" "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode fetch-data
"$PC_PYTHON" "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode smoke
~~~

Bootstrap uses an **isolated Python 3.12 environment**, official Torch 2.11.0 CUDA 12.8, NumPy 2.4.3 and Matplotlib 3.10.8. Earlier experiment environments are untouched. A compatible existing GPU is required; nothing purchases infrastructure.

Data fetching calls the official Toronto downloader: archive MD5 c58f30108f718f92721af3b95e74349a, then seven extracted-file SHA-256 hashes. These are expected hashes, not evidence that downloading happened.

Smoke keeps architecture, batch, views and SIGReg projections but uses **four updates**, shortened schedules and short corpora. Its configs/logs go under smoke-only/. It is never a paper result. Its default 180-second timeout is a cap, not measured performance. Process wall time includes startup, loading, evaluation and checkpoint I/O; it is not steady-state update time.

After inspecting actual hardware and smoke evidence:

~~~bash
"$PC_PYTHON" "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode full
"$PC_PYTHON" "$PC_REPRO/aggregate.py" --upstream "$PC_UPSTREAM"
~~~

Use --study traffic --seeds 303 for a first substantive five-model full-budget sweep. Reusing the output directory verifies/reuses completed official stages. Incomplete stages are not silently overwritten: preserve failure logs and use a fresh output directory if needed.

Ignored runs/ contains planned configs/hashes, hardware/data audit, official logs, checkpoints/result hashes, completion receipts, timing journals, per-seed figures and aggregates. Preserve actual large artifacts durably when runs become possible; no such artifacts exist yet.

CPU is an explicit precision variant:

~~~bash
TORCH_INDEX=https://download.pytorch.org/whl/cpu bash "$PC_REPRO/bootstrap.sh" "$PC_UPSTREAM"
"$PC_PYTHON" "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode smoke --device cpu --cpu-fp32 --output "$PC_REPRO/runs-cpu"
~~~

Upstream --device cpu alone fails because its configs retain bf16=true. The opt-in CPU profile changes only device/BF16 in all stages, saving modified configs. CPU feasibility is unmeasured.

## Audit and comparisons

The executed configs use 32 base channels/1024 projection width; smaller class defaults are irrelevant. Probe metadata [16] is misleading: IndependentMLPBank actually executes input→16→16→output with two GELUs, matching Appendix B. No probe architecture patch is needed.

Official plots are per seed. Our aggregation requires all 77 verified encoders, reports raw seeds and mean/SEM, and retains primary-seed reconstruction heatmaps. Its presentation is not claimed pixel-identical to the paper. Group-size extrema reuse existing probes.

**Appendix C is not implemented by this harness or scheduled by the pinned runner.** Actual-versus-incompatible-alternative and shuffled-prediction checks require a separate audited evaluator. Traffic intervention/loss diagnostics are not substitutes. [AUTHOR_REFERENCE.json](AUTHOR_REFERENCE.json) contains published Table 1 numbers only as author reference. No released trained results/checkpoints were found in the pinned tree; the releases endpoint returned no releases/assets.

Compare complete curves/grids and seed spread:

- Traffic: current-light decoding as q increases, plus chance/future-light controls.
- Pair switching: A/B versus C/D preference throughout pD.
- Group size: averages and worst/best positions, endpoint reversal and seed variation.
- Reconstruction: all four beta/pD heatmaps, including nonmonotonic cells.
- Side information: each feature against its paired baseline; do not assume a uniform pair switch.

Report effect signs and mean/SEM without inventing numeric paper acceptance tolerances. Smoke trends, one seed or partial grids do not establish full reproduction.

## Lightweight checks

These imports/planning tests use only the standard library and pinned source, without data or Torch:

~~~bash
PC_UPSTREAM="$PC_UPSTREAM" python -m unittest discover -s "$PC_REPRO" -p test_launcher.py
python "$PC_REPRO/launch.py" --upstream "$PC_UPSTREAM" --mode plan --output /tmp/pc-plan-check
~~~

**Seven planner/source/runtime-guard checks and a separate-process checkpoint/resume test passed locally.** These check execution mechanics, not scientific results. Missing source explicitly skips planner tests rather than passing them.

Sources: [paper v1](https://arxiv.org/abs/2610.27364v1), [official repository](https://github.com/Mosh0110/predictive-closure), [pinned runner](https://github.com/Mosh0110/predictive-closure/blob/2964fa8f737fa43871d47928edc9aac16fa9298f/run.py).

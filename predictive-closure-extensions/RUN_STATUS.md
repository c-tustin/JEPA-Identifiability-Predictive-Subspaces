# Predictive Closure extension status

The original research runs finished in the cloud on 2026-10-25. This is a completed toy extension of Predictive Closure, not a reproduction of its CIFAR experiments or a proof of a new identification theorem.

## What was run

- Signed-OU controls: five independent seeds per law, 200,000 pairs each, with 20 fitted predictor controls. Saved in `output/controls.json`.
- Learned representations: four prediction objectives times five seeds, 2,000 updates per model, batch 256 and 3,988 parameters. All 20 neural encoders retained distance; no feature-selection separation was observed. Checkpoints, settings and metrics are in `output/`.
- Explicit full versus folded coordinates: 120 spline fits and 18 quadrature checks. The folded coordinate predicts its own future better but confuses two physical distances. Continuous and quantized information are reported separately in `output/explicit-encoders.json`.
- Nonlinear drift/planning pilots: ten initial fits and 400 rollouts, followed by 15 fits and 560 rollouts on fresh seeds. Multiple observation intervals increased error here. Results remain in `output/drift.json` and `output/drift-weighted.json`.

Original cloud validation reproduced all 20 checkpoint evaluations exactly and replayed all 960 executed trajectories exactly. Every executed Euler substep stayed at positive distance; no reflecting boundary was used. `output/checkpoint-verification.json` and `output/trajectory-verification.json` preserve those original records.

## Standalone code and current checks

On 2026-10-28, this branch was rebuilt directly on main. Everything needed is in the top-level `predictive-closure-extensions/` folder, including the encoder (`models.py`) and scalar SIGReg (`sigreg.py`). The research equations and saved data did not change.

All 29 saved output files are byte-for-byte unchanged. All 20 checkpoints load through the local encoder and produce exactly the same test outputs as the original encoder. The encoder definitions are structurally identical, and SIGReg is byte-identical. The ten standalone scientific tests pass. See [layout verification](LAYOUT_CHECK.md).

The original cloud runtime was Python 3.12.14, Torch 2.2.2+cpu, NumPy 1.26.4, SciPy 1.17.0 and Matplotlib 3.10.8. Current relocation checks ran locally with Torch 2.2.2, NumPy 1.26.4 and SciPy 1.13.1; they do not replace the original numerical results. No neural training was repeated during reorganization.

## Run commands

From repository root after installing `requirements-cpu.txt`:

```bash
export OPENBLAS_NUM_THREADS=1
python predictive-closure-extensions/closure_controls.py
python predictive-closure-extensions/check_explicit_encoders.py
python predictive-closure-extensions/run_learned.py
python predictive-closure-extensions/run_drift.py
python predictive-closure-extensions/run_drift.py --weighted-followup
python predictive-closure-extensions/recheck_checkpoints.py
python predictive-closure-extensions/verify_trajectories.py
python predictive-closure-extensions/summarize.py
python -m pytest predictive-closure-extensions -q
```

## Limits

The signed-OU world has non-Gaussian conditional distributions, so it lies outside the paper's conditional-Gaussian equivalence assumptions. The folded-coordinate comparison uses known geometry rather than learned neural encoders. Finite precision information is separated from ideal continuous information. Drift fitting uses calibrated observations, simulator resets and a favorable model family; the planning rule is a local heuristic. These limits and the negative results are retained in [RESULTS.md](RESULTS.md).

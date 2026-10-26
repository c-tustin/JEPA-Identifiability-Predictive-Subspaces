# Cloud execution status

Completed a reviewable research increment on 2026-10-25 (America/New_York). This is not completion of the broader JEPA identification research program.

## Access and provenance

Cloud workspace `/workspace/scratch/339c04b91302`, Python 3.12.14, CPU execution. Git clone, fetch and Python numerical compute worked. Initial branch fetched at 937018a; fetched again and fast-forwarded to **586d801** after the local research handoff arrived. Read that handoff and the existing EXTENSION_SUMMARY, DRIFT_ESTIMATION, CONTROL_INTERPRETATION, EXTENSION_PROTOCOL and planning summary. No edits to planning/SUMMARY.md or PR #3. Work branch: `codex/predictive-closure-extensions`, based on `behavior-policy-experiments`.

Public PDF downloaded and text extracted; Appendix A.1–A.2 read in full, with A.6 and experimental setup. Upstream cloned, pinned and license inspected; see README for revision/hash and inspected files. No upstream code adapted.

Installed CPU Torch 2.2.2+cpu after inspecting the environment. Initial environment had NumPy 2.3.5, SciPy 1.17.0, no Torch/pytest. The first pip call mistakenly included pytest on the Torch-only index and failed; split installation succeeded. A first neural run failed during evaluation because Torch 2.2.2's NumPy bridge needs NumPy 1.x; no checkpoint/result from that attempt was retained. Pinned NumPy 1.26.4 and reran all 20 neural models successfully. Recorded dependencies are in requirements-cpu.txt (Matplotlib 3.10.8, pytest 9.1.1). No computation ran on the user's local machine.

## Commands actually run

From repository root, with `OPENBLAS_NUM_THREADS=1` for reproducibility:

```bash
python behavior-policy-experiments/predictive-closure/closure_controls.py
python behavior-policy-experiments/predictive-closure/run_learned.py
python behavior-policy-experiments/predictive-closure/run_drift.py
python behavior-policy-experiments/predictive-closure/run_drift.py --weighted-followup
python behavior-policy-experiments/predictive-closure/recheck_checkpoints.py
python behavior-policy-experiments/predictive-closure/verify_trajectories.py
python behavior-policy-experiments/predictive-closure/summarize.py
python -m pytest behavior-policy-experiments -q
```

The control script was initially named run_controls.py and renamed to closure_controls.py to avoid colliding with an existing test module. A new test file was similarly renamed before the full suite passed. Final scientific outputs use the final filenames above.

## Saved evidence

- `output/controls.json`: five independent seeds per law, 200,000 pairs each; 20 fitted two-parameter predictors, settings and optimizer status. All converged. Common-coordinate NLL, MSE, conditional second-moment error and marginal/information diagnostics.
- `output/learned.json`, `learned-settings.json`, and 20 `.pt` files: four arms × five encoder seeds, 2,000 updates each. Saved marginal/geometry/probe/prediction metrics and loss histories. All evaluation random streams are separate from training.
- `output/checkpoint-verification.json`: all 20 final checkpoints reproduce every saved evaluation metric, maximum absolute difference 0.
- `output/drift.json`: initial five seeds, ten independent-noise fits, 400 task/controller rollouts.
- `output/drift-weighted.json`: prospective variance-weighted follow-up on five fresh seeds, 15 fits and 560 task/controller rollouts. The follow-up protocol was added after inspecting the initial negative result, before running fresh seeds.
- `RESULTS.md` and `output/comparison.png`: generated from saved JSON; figure visually inspected.
- Full repository suite: **68 tests passed** (58 existing plus 10 new checks). New checks cover mixture normalization/conditional moments, non-Gaussianity, marginal/information relationships, stable tails, sensor equivalence, instantaneous drift design, simulation variance/convergence and matched active output gradients/parameter counts.

## Remaining scientific work

The original signed-OU control compares two laws. The later explicit-encoder control is within one world, outside the conditional-Gaussian assumptions; it is not a learned feature-selection or theorem test. The neural experiment found no feature-selection separation. Finite precision entropy equivalence is not verified; continuous MI is reported as such. Scalar drift regression uses exact sensor calibration, simulator resets and a favorable drift/diffusion family. Risk-adjusted planning is a local heuristic and has no robust-control certificate. Next pursue misspecified noise/drift, sample allocation, shuffled/noisy action controls, and information-preserving learned coordinates with physical control checks. The existing nonlinear world's joint encoder action/shuffled-action experiment remains unrun.

Final delivery verification: regenerated report/figure from final JSON, visually checked the figure, verified all 20 checkpoints exactly, and reran the 68-test suite successfully. All learned arms have **3,988 total parameters**, 2,000 updates and batch 256; functional redundancies mean this is a nominal parameter budget, not identical effective capacity. Trajectory replay checks every executed Euler substep for positivity and reproduces saved decision states; see output/trajectory-verification.json. No reflecting boundary is used.

After the initial local commit, retrieved base changes through **fa546cb99c769c7a69f20a17dc307b87816b7a1d** using the GitHub app because direct GitHub network access became blocked. Imported the two missing base commits only after matching their Git object hashes exactly, then merged normally. Both README sections and planning/SUMMARY.md are preserved.

Prospective same-world follow-up: ran `OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/predictive-closure/check_explicit_encoders.py` after fixing its protocol. Saved output/explicit-encoders.json: 120 spline fits (5 fresh seeds × 3 clipping levels × 4 rounding choices × 2 explicit encoders), 23 coefficients each, and 18 known-law quantized-information checks, each at 64/128 quadrature nodes. The folded explicit coordinate gives smaller latent mean error while losing physical sign; the learned neural arms still retained distance. Continuous information equality and finite-resolution differences are reported separately. No neural retraining was performed during delivery verification.

Executed Euler substep ranges from trajectory replay: initial study [2.4353, 14.8213], weighted follow-up [2.4958, 14.8683]. Every replayed decision state matched exactly. These positivity checks concern executed paths, not an invariant-region guarantee for every candidate rollout.

Immediately before publication the base advanced again to **03f0f0ae7cb72e43f3110230e16cc49a753b2bd8**. Imported and merged its two verified commits, preserving ACTION_INFORMATION.md, FINITE_CAPACITY_DISTANCE.md and their handoff links. These propose actuator-off and finite-capacity controls; neither new benchmark has been run here. Publication uses the GitHub app to upload hash-verified Git blobs and create a commit with the exact verified working-tree contents on the latest base, because direct Git transport is blocked. Local incremental commits are retained separately.

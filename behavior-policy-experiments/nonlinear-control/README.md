# Nonlinear dynamics without adding state coordinates

**Latest:** [coverage, nonlinear planning and the drift-estimation result](EXTENSION_SUMMARY.md). This extends the first trajectory comparison with 12 paired coverage controls and 84 predictors evaluated in deterministic/noisy execution.

```sh
python behavior-policy-experiments/nonlinear-control/run_coverage.py
python behavior-policy-experiments/nonlinear-control/evaluate_control.py
python behavior-policy-experiments/nonlinear-control/check_drift_sampling.py
```

The first two commands support `--quick --output /tmp/a-new-folder`, `--resume` and `--evaluate-only`. Full planning requires the completed trajectory and coverage encoders. The drift diagnostic supports `--quick` and `--output`. [Exact protocol](EXTENSION_PROTOCOL.md) · [Why finite-time prediction differs from drift fitting](DRIFT_ESTIMATION.md).

Two experiments answer different questions:

1. **Ordinary trajectories:** does the original flexible encoder + SIGReg still recover distance when the dynamics change? [Short findings](SUMMARY.md) · [Fixed comparison](TRAJECTORY_PROTOCOL.md) · [Full results](output-trajectories/RESULTS.md).
2. **Known action effects:** can stronger structural assumptions identify distance despite nonlinear drift? The proof and first prototype below address this question.

## Ordinary trajectories

Keep the same triangle, one distance, and original encoder. Compare zero versus nonlinear drift, random versus persistent versus action-balanced policies, and raw versus balanced SIGReg sampling. The fixed protocol uses three new seeds and 36 encoders. No ordered encoder or paired interventions are used here.

```sh
python behavior-policy-experiments/nonlinear-control/run_trajectories.py
python behavior-policy-experiments/nonlinear-control/run_trajectories.py --evaluate-only
```

Use `--quick --output /tmp/nonlinear-trajectories-smoke` for an execution check; use `--resume` to continue an interrupted run. An existing output is never silently overwritten. [How to interpret recovery and folding](TRAJECTORY_INTERPRETATION.md).

## Known action effects

The triangle still changes as $x=V/d$. We allow nonlinear drift and state-dependent noise in the same distance variable:

$$
dD_t=[0.3\sin D_t+0.25A_t]dt+0.05(1+0.1\cos D_t)dW_t.
$$

The [proof](PROOF.md) shows that constant physical action effects can identify distance up to an affine map among smooth invertible coordinates. It also bounds coordinate error using drift-fit error, state coverage and conditional action variance. **It does not prove that unconstrained JEPA finds such coordinates.**

The numerical prototype trains a smooth ordered encoder (positive combinations of softplus hinges) from paired approach/retreat interventions, using common simulator noise. It uses 4,000 updates of 512 paired queries per seed, Euler interval 0.02, and interior distances 1.2–15.8. Its target is the known action coefficient, not distance labels. These are simulator intervention queries, not ordinary policy trajectories. It avoids boundaries and uses physical-distance recovery only for evaluation. Output Gaussianity is not imposed on bounded distance. These additional structural assumptions are explicit.

```sh
python behavior-policy-experiments/nonlinear-control/check_actuation.py
```

Use `--quick --output /tmp/nonlinear-smoke` for an execution check. [Results](output/RESULTS.md) include three fixed seeds and a separate analytic check of the approximate bound. A future finite-sample JEPA theorem must still connect its objective to the conditional-drift risk, establish information preservation, and control sampling and discretization errors.

All three smooth encoders recover physical distance with $R^2>0.9999999$ in this restricted prototype. This illustrates the anchored model class; it does not validate a general nonlinear-JEPA guarantee.

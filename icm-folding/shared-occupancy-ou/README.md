# Multimodal OU data shared by alignment and SIGReg

This corrects the scope of the earlier multimodal test: **both losses now train on the same bimodal or trimodal temporal data**. The earlier SIGReg-only experiments remain available and retain their original interpretation.

An exact latent OU step generates `z_next = rho*z + sqrt(1-rho²)*epsilon`, with `rho=exp(-dt)` and standard-normal initial states. Map `u=Phi(z)` through the target log-distance inverse CDF, then render the distance with the existing sensor. Since each latent marginal is standard normal, `u` is uniform and every distance marginal has the specified target occupancy. This holds for either tested dt in the population. Smaller dt produces more correlated pairs; the nonlinear coordinate map makes physical step sizes depend on position. Peaks are at (2,8) or (2,5,12), each of log-width 0.14, with 10% uniform-log background. The implementation preserves the earlier 64-bin target distribution exactly as a piecewise-uniform density in log-distance.

A conventional OU process stays unimodal when its exact time step changes. The inverse-CDF transform supplies multimodality here. The process is passive simulated behavior, like the earlier OU pair baseline; it is **not a learned policy in the three-action environment**. It uses privileged distance coordinates and stationary initialization, and its distance increments can exceed 0.25. See the [fixed exploratory protocol](PROTOCOL.md) for scope and limitations.

At every update, uniformly sample pair indices and compute:

```python
y = encoder(torch.cat([x[ix], xp[ix]]))  # exactly one encoder call
alignment = (y[:batch] - y[batch:]).square().mean()
loss = 0.9 * alignment + 0.1 * sigreg(y)
```

No separate regularizer dataset, occupancy weighting, endpoint substitution or historical ICM pairs. This changes the transition measure as well as the endpoint occupancy relative to the earlier ICM-based fits. It tests representation learning on multimodal behavior data rather than isolating a regularizer-only mechanism.

## Reproduce and inspect

All 18 exploratory fits completed. No final folds were detected in the 12 multimodal fits or six uniform-log controls, with slopes independently checked on a 20,001-point float64 grid after checkpoint reload. Measured multimodal histogram TV from target averages 0.053–0.054 at dt=0.01 and 0.026–0.027 at dt=0.1. These finite correlated datasets approximate the prescribed occupancy; the result does not establish that arbitrary multimodal policies avoid folds.

The [ICM-occupancy-matched OU follow-up](../icm-matched-ou/README.md) instead targets each stored ICM empirical endpoint distribution and does reproduce folds: 2/4 at dt=0.01 and 3/4 at dt=0.1, versus 2/4 for new shared-input native ICM controls. It includes [a compact view of these original multimodal histograms and encoder curves](../icm-matched-ou/output/bimodal_trimodal_shared_inputs.png) and [an explanation diagram of the transformation](../icm-matched-ou/output/ou_quantile_transform.png).

From the workspace root, with NumPy, SciPy, PyTorch and Matplotlib installed (recorded versions in [requirements.txt](requirements.txt)):

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/shared-occupancy-ou/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/shared-occupancy-ou/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/shared-occupancy-ou/report.py
```

See [results](output/RESULTS.md), [measured occupancy, trajectories and encoders](output/shared_occupancy_encoders.png), [transition matrices](output/transition_matrices.png), [per-run metrics](output/summary.json), [input/generator audit](output/audit.json), [checkpoint verification](output/verification.json), and [source hashes and runtime versions](output/provenance.json).

Reuses the existing sensor, gain-4 scalar MLP, scheduled Adam rate, SIGReg and target-bin construction through [the canonical occupancy study](../sources/explanations/occupancy-folding/run_study.py), without copying source folders or checkpoints. New seeds reproduce the historical encoder initialization stream; paired conditions share normal innovations, initial weights and pair-index streams. Fold checks are finite-grid empirical diagnostics. Remaining work includes a bounded-action multimodal actor and fresh-seed replication.

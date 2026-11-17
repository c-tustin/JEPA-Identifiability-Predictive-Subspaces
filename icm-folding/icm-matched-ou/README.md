# Recreate native ICM occupancy with a transformed OU process

Requested follow-up to [the shared-input multimodal OU study](../shared-occupancy-ou/README.md). Both temporal alignment and ordinary SIGReg consume exactly the same encoded endpoints. [Protocol](PROTOCOL.md) fixes four gallery seeds, two inherited OU time steps, and a fresh shared-input native ICM control per seed.

All twelve 4,000-update fits and final checkpoint verifications completed. Native shared-input pairs fold in 2/4 runs; ICM-matched OU folds in 2/4 at dt=0.01 and 3/4 at dt=0.1. Seeds 131 and 136 fold in all arms; seed 130 gains a fold at dt=0.1. All final roots lie inside their actual training distance range. Mean 64-log-bin occupancy TV from the source is 0.0534/0.0221 for dt=0.01/0.1. This recreates folding under concentrated ICM-like occupancy with OU-generated temporal pairs, without establishing that occupancy alone determines a fold.

See [histograms, encoder curves and slopes](output/icm_matched_histograms_encoders.png), [per-seed fold histories](output/fold_histories.png), [compact bimodal/trimodal gallery](output/bimodal_trimodal_shared_inputs.png), [transformation diagram](output/ou_quantile_transform.png), [results](output/RESULTS.md), [full metrics](output/summary.json), [input/generator audits](output/generator-audit.json), [shared-loss audit](output/audit.json), [checkpoint verification](output/verification.json), and [source/runtime provenance](output/provenance.json).

## What the generator does

First generate a standard-normal stationary OU trajectory:

`z_next = exp(-dt) * z + sqrt(1 - exp(-2*dt)) * epsilon`.

Then transform each value to its Gaussian percentile `u=Phi(z)`. Thus z=0 gives u=0.5, and z≈1.28 gives u≈0.9. Finally select the target observation at that percentile. For this study, sort all current and successor ICM observations by distance and select rank `floor(N*u)`. The 50th-percentile Gaussian state becomes the median ICM observation, and the 90th-percentile state becomes its 90th-percentile observation. Each source endpoint has equal probability in the stationary population; its empirical multiplicity is preserved. No new distances, smoothed bins, or uniform background are introduced.

The earlier synthetic targets used `q=F_target_inverse(u)` and `d=exp(q)`, with a piecewise-uniform density in 64 log-distance bins. Here the inverse empirical CDF selects an actual original sensor vector. Both are fixed, increasing rank maps before representation learning. The rank map itself cannot reverse distance order; any detected fold is in the subsequently trained encoder. Plateaus can occur when an empirical quantile selects the same observation at adjacent times.

Because OU values are temporally correlated, the percentile-transformed observations also form a trajectory. Smaller dt generally produces more local and correlated transitions. The transformation sets the stationary marginal, while dt changes the coupling between consecutive observations. This is a passive simulator-based process, not a newly learned ICM actor. Its transitions need not obey the original actor's action limit or reproduce ICM's directional drift. Finite correlated trajectories approximate the exact empirical marginal targeted in the population.

## Reproduce

Run from the workspace root with the [previous study's dependencies](../shared-occupancy-ou/requirements.txt):

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/icm-matched-ou/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/icm-matched-ou/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/icm-matched-ou/report.py
```

The [trainer](run.py) imports the existing shared-loss implementation and canonical sensor/encoder code; it does not modify the earlier hash-verified runs. Source ICM datasets are reused in their original location and referenced by hash. New fits retain pair indices, initial/final weights, OU trajectories/ranks and source/protocol/code hashes. Occupancy checks use both 32- and 64-log-bin histograms, an unbinned KS distance and near-object mass. Final derivative checks use a 20,001-point float64 grid; they are empirical, not continuous-domain guarantees.

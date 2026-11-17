# Shared-input multimodal OU results

November 16, 2026. All 18 fits completed: three targets × two OU time steps × three diagnostic seeds, 4,000 updates each. Both losses use identical paired observations and a single encoder evaluation. No occupancy weights or ICM transition pairs.

| Target | dt | Final folds | Mean endpoint TV to target | Mean basin-switch rate | Mean fraction of steps > 0.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 0.01 | 0/3 | 0.0529 | 0.0000 | 0.5447 |
| uniform | 0.1 | 0/3 | 0.0295 | 0.0000 | 0.7798 |
| bimodal | 0.01 | 0/3 | 0.0531 | 0.0490 | 0.3085 |
| bimodal | 0.1 | 0/3 | 0.0260 | 0.1409 | 0.5997 |
| trimodal | 0.01 | 0/3 | 0.0536 | 0.0860 | 0.4582 |
| trimodal | 0.1 | 0/3 | 0.0271 | 0.2579 | 0.7276 |

Final encoder standard deviations on the uniform-log evaluation grid range from 0.8114 to 0.8908; affine log-distance R² ranges from 0.9382 to 0.9976. Standardized plots alone do not show these output scales.

TV is total variation of the measured 64-bin endpoint histogram from the prescribed mass; endpoints are temporally correlated. Basin boundaries are log-midpoints between peaks. Switching counts all adjacent basin changes and can include repeated boundary crossings. The uniform control has one basin by convention.

Checkpoint reload reproduces stored final float32 curves/slopes exactly. Fold roots are independently recomputed on a 20,001-point float64 grid; these are empirical sign-reversal checks, not injectivity guarantees. All generated data, shared pair-index streams, initializations and normal innovations were regenerated and checked.

This passive transformed OU process permits distance increments beyond the original three-action environment. Occupancy is stationary in the population by construction, but finite correlated trajectories fluctuate. The plots display actual data occupancy. Time-step comparisons change temporal coupling; cross-target comparisons change both occupancy and pair geometry. No causal occupancy-only or bounded-action actor claim follows.

# Nonlinear goal-reaching evaluation

Frozen encoders; independently trained action predictors. Smaller physical errors are better. Hit rate is the fraction reaching within 0.1 during 12 decisions. Goals have a two-action deterministic reachability witness. [Protocol](../EXTENSION_PROTOCOL.md).

Two predictor seeds are nested inside three representation seeds. Means below average those six fits; they are not six independent representation replicates.

## Deterministic execution

| Encoder data | SIGReg | Predictor data | Fits | Closest error | Hit rate | Final error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| action_balanced | balanced | common | 6 | 0.2510 | 0.542 | 1.4428 |
| action_balanced | balanced | native | 6 | 0.2775 | 0.484 | 1.2458 |
| action_balanced | raw | common | 6 | 0.0419 | 0.865 | 0.2654 |
| action_balanced | raw | native | 6 | 0.0160 | 0.958 | 0.2074 |
| persistent | balanced | common | 6 | 0.1772 | 0.661 | 0.8622 |
| persistent | balanced | native | 6 | 0.1944 | 0.682 | 1.0593 |
| persistent | raw | common | 6 | 0.0259 | 0.932 | 0.2231 |
| persistent | raw | native | 6 | 0.0015 | 1.000 | 0.1799 |
| random | balanced | common | 6 | 0.2349 | 0.693 | 1.3914 |
| random | balanced | native | 6 | 0.2426 | 0.620 | 1.5392 |
| random | raw | common | 6 | 0.0529 | 0.849 | 0.2730 |
| random | raw | native | 6 | 0.0148 | 0.958 | 0.1750 |
| reset_uniform | balanced | common | 6 | 0.0751 | 0.818 | 0.3794 |
| reset_uniform | raw | common | 6 | 0.0376 | 0.927 | 0.4162 |
| known_dynamics | — | — | 3 | 0.0396 | 0.865 | 0.1786 |
| stay | — | — | 3 | 0.4921 | 0.000 | 1.7213 |

## Stochastic execution

| Encoder data | SIGReg | Predictor data | Fits | Closest error | Hit rate | Final error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| action_balanced | balanced | common | 6 | 0.2624 | 0.531 | 1.5031 |
| action_balanced | balanced | native | 6 | 0.2658 | 0.490 | 1.1961 |
| action_balanced | raw | common | 6 | 0.0544 | 0.891 | 0.2776 |
| action_balanced | raw | native | 6 | 0.0304 | 0.943 | 0.2290 |
| persistent | balanced | common | 6 | 0.1930 | 0.609 | 0.9009 |
| persistent | balanced | native | 6 | 0.2256 | 0.589 | 1.1408 |
| persistent | raw | common | 6 | 0.0346 | 0.974 | 0.2819 |
| persistent | raw | native | 6 | 0.0204 | 1.000 | 0.2129 |
| random | balanced | common | 6 | 0.2562 | 0.594 | 1.4189 |
| random | balanced | native | 6 | 0.2654 | 0.542 | 1.5073 |
| random | raw | common | 6 | 0.0667 | 0.839 | 0.3058 |
| random | raw | native | 6 | 0.0265 | 0.953 | 0.2257 |
| reset_uniform | balanced | common | 6 | 0.0945 | 0.812 | 0.3981 |
| reset_uniform | raw | common | 6 | 0.0558 | 0.875 | 0.4319 |
| known_dynamics | — | — | 3 | 0.0355 | 0.927 | 0.1980 |
| stay | — | — | 3 | 0.4541 | 0.052 | 1.6711 |

The known-dynamics nominal MPC uses the same search bank/cost, but it is not a global or stochastic optimum. Closest error and final error answer different questions when drift prevents holding a goal. True distance never enters the learned controller. No true-distance stopping rule is used.

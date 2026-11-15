# Fresh-source fixed-hyperparameter check

Five newly trained ICM policy/trajectory sources, seeds 140--144. Global-mining coefficient 100 fixed before generating these sources; no retuning. Same original scalar MLP. Small validation in the same deterministic renderer, not a new environment family.

| Objective | Any folds | Material folds | Mean backward fraction | Held-out MSE | Coverage-stress MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIGReg | 2/5 | 2/5 | 0.1599 | 0.135788 | 3.114902 |
| VICReg (scalar) | 1/5 | 0/5 | 0.0000 | 0.004945 | 0.038740 |
| InfoNCE (distance) | 0/5 | 0/5 | 0.0000 | 0.002560 | 0.119524 |
| Gaussian quantiles | 5/5 | 5/5 | 0.3570 | 0.195411 | 2.400380 |
| SIGReg + global mining 100 | 0/5 | 0/5 | 0.0000 | 0.000334 | 0.013081 |

Global-mining paired results vs SIGReg: 5/5 held-out prediction improvements; 2 material folds prevented, 0 created. Geometric-mean prediction MSE ratio 0.021.

All final geometries and frozen-encoder decoder errors were independently recomputed from checkpoints. Counts are descriptive; five sources are too few to establish a general success rate. There is no hard no-fold guarantee.

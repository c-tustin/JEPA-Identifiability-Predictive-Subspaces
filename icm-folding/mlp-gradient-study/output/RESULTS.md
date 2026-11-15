# Original MLP: gradients and regularizers

Baseline and transition-gradient fits share initialization and training batches. Ten historical ICM sources; no ordering or dynamics labels enter encoder training. Counts are descriptive.

| Objective | Gradient | Any folds | Material folds | Mean backward fraction | Held-out MSE | Coverage-stress MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIGReg | baseline | 5/10 | 5/10 | 0.1746 | 0.210549 | 1.316871 |
| SIGReg | transition | 5/10 | 5/10 | 0.1738 | 0.207936 | 1.287567 |
| VICReg (scalar) | baseline | 0/10 | 0/10 | 0.0000 | 0.004397 | 0.046030 |
| VICReg (scalar) | transition | 0/10 | 0/10 | 0.0000 | 0.004389 | 0.045807 |
| InfoNCE (distance) | baseline | 0/10 | 0/10 | 0.0000 | 0.002071 | 0.054770 |
| InfoNCE (distance) | transition | 0/10 | 0/10 | 0.0000 | 0.002063 | 0.058153 |
| Gaussian quantiles | baseline | 9/10 | 9/10 | 0.3153 | 0.374455 | 8.263022 |
| Gaussian quantiles | transition | 9/10 | 9/10 | 0.3149 | 0.372705 | 8.222022 |

The MSE is next-observation prediction from a frozen scalar code and action, with targets whitened using training data. The common uniform-distance set is a coverage/extrapolation stress test, not another held-out draw from the native distribution. Scalar VICReg has no covariance term; InfoNCE uses a scalar RBF score. Material folds exclude reversals with backward variation <= 0.001.

## Paired prediction comparisons

Bootstrap intervals resample ten paired source seeds. They are exploratory intervals, without multiple-comparison correction.

- SIGReg: 9/10 held-out prediction improvements; geometric-mean MSE ratio 0.980, bootstrap interval [0.9595797126906346, 0.998083801288776]; material folds repaired 0, created 0.
- VICReg (scalar): 7/10 held-out prediction improvements; geometric-mean MSE ratio 0.999, bootstrap interval [0.99611897731198, 1.0005898837887819]; material folds repaired 0, created 0.
- InfoNCE (distance): 9/10 held-out prediction improvements; geometric-mean MSE ratio 0.997, bootstrap interval [0.9940606726454366, 0.9993117489220109]; material folds repaired 0, created 0.
- Gaussian quantiles: 9/10 held-out prediction improvements; geometric-mean MSE ratio 0.992, bootstrap interval [0.982853538527556, 0.9973837368312531]; material folds repaired 0, created 0.

## Prediction references

The raw-input decoder is an optimization/conditioning reference, not a rigorous lower bound: a scalar feature can make this finite predictor easier to fit.

- native_mse: mean raw-input decoder 0.005355; action-only mean predictor 0.737383.
- common_mse: mean raw-input decoder 0.020692; action-only mean predictor 17.537606.

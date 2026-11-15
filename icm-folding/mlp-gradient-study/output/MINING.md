# Adaptive global transition-collision mining

Same scalar MLP, base SIGReg objective, random initialization and base batches. Mining and coefficients were chosen adaptively after inspecting development results. Extra computation/observation exposure is not matched. This is not a confirmatory ten-seed generalization test.

| Global mining coefficient | Any folds | Material folds | Mean backward fraction | Held-out MSE | Coverage-stress MSE | Held-out improvements vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5/10 | 5/10 | 0.1746 | 0.210549 | 1.316871 | 0/10 |
| 1 | 5/10 | 5/10 | 0.1724 | 0.184759 | 0.712630 | 7/10 |
| 10 | 2/10 | 2/10 | 0.0568 | 0.038962 | 0.417623 | 10/10 |
| 100 | 1/10 | 1/10 | 0.0325 | 0.019100 | 0.103342 | 10/10 |

No geometry labels or dynamics equations enter mining or the encoder gradient. Development did inspect geometry diagnostics. Zero observed folds is not an all-input guarantee; dense checks cover this simulator curve. Same-action successor differences may include nuisance or stochastic variation in other environments.

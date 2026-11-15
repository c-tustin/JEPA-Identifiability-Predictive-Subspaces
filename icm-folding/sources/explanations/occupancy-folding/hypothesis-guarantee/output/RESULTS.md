# Hypothesis tests and guarantees

| Concentration | Method | Models | Final folded | Mean fixed-batch objective | Mean population objective | Mean affine log-distance R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| native | free | 10 | 5 | 4.1298 | 3.9352 | 0.6119 |
| native | positive | 10 | 0 | 10.5014 | 10.5198 | 0.7657 |
| native | projected | 10 | 0 | 1.2570 | 1.1615 | 0.5832 |
| 87.5% | free | 20 | 20 | 4.5010 | 4.1697 | 0.3500 |
| 87.5% | positive | 20 | 0 | 7.9727 | 7.7963 | 0.7699 |
| 87.5% | projected | 20 | 0 | 1.2167 | 0.9825 | 0.5811 |
| 93.75% | free | 20 | 5 | 5.5248 | 5.3067 | 0.7732 |
| 93.75% | positive | 20 | 0 | 12.9712 | 12.9370 | 0.7659 |
| 93.75% | projected | 20 | 0 | 1.3679 | 1.2501 | 0.5518 |

## Actor geometry controls

| Sensor | Mean distance | Fraction below 2 | Fraction above 15 | Mean final training reward |
| --- | ---: | ---: | ---: | ---: |
| native | 1.7012 | 0.8835 | 0.0002 | 0.018587 |
| mirror | 15.0518 | 0.0002 | 0.8488 | 0.019937 |
| equalized | 6.7011 | 0.2127 | 0.0382 | 0.006463 |

## Concentration switches: source 131, one far bump

| Old → new concentration | Switch update | Reset Adam | Final folded | Turning distances | Fixed-batch objective | Population objective |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 87.5% → 93.75% | 100 | False | False | — | 6.1913 | 5.8594 |
| 87.5% → 93.75% | 500 | True | True | 2.6468 | 4.8425 | 4.5786 |
| 87.5% → 93.75% | 500 | False | True | 2.6538 | 4.8454 | 4.5781 |
| 93.75% → 87.5% | 100 | False | True | 2.5461 | 4.2827 | 4.0645 |
| 93.75% → 87.5% | 500 | True | False | — | 7.9026 | 7.4403 |
| 93.75% → 87.5% | 500 | False | False | — | 7.9247 | 7.4601 |

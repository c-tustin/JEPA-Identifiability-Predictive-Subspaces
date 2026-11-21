# New-source controlled validation

All 80 fits passed independent dense float64 reload and source/scope/initializer/plan checks.

| Near mass | Objective | Fits | Any turn | Material turn |
| --- | --- | ---: | ---: | ---: |
| 50.000% | sigreg | 10 | 3 | 3 |
| 50.000% | raw_infonce | 10 | 0 | 0 |
| 75.000% | sigreg | 10 | 10 | 10 |
| 75.000% | raw_infonce | 10 | 0 | 0 |
| 93.750% | sigreg | 10 | 0 | 0 |
| 93.750% | raw_infonce | 10 | 0 | 0 |

The frozen twelve-encoder risk bank predicts material status correctly in 8/10 native fits; the coarse near-mass window in 9/10. Predictions were recorded and hash-locked before encoder fitting; the bank was frozen before generating the ten policy sources. No sources were replaced.

| Source | Native near mass | Observed material fold | Risk-bank prediction | Coarse mass prediction |
| --- | ---: | --- | --- | --- |
| 9100 | 96.237% | False | False | False |
| 9101 | 89.581% | True | True | True |
| 9102 | 93.777% | False | False | False |
| 9103 | 87.625% | True | True | True |
| 9104 | 74.686% | True | True | True |
| 9105 | 89.807% | True | False | True |
| 9106 | 73.911% | True | True | True |
| 9107 | 83.060% | True | True | True |
| 9108 | 85.544% | True | True | True |
| 9109 | 56.720% | True | False | False |

These are newly trained actors and newly collected trajectories in the same renderer. The independent endpoint interventions hold source-131 alignment fixed; temporal InfoNCE and original joint-data VICReg are checked separately. A finite reference-bank predictor is not a universal distribution theorem.

[All measurements](summary.json) · [Independent audit](verification.json)

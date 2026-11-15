# Concentration-matched results

Ten previously inspected ICM sources. All primary targets retain the same observed-point support and 2% uniform-over-occupied-log-bins background. One/two-bump cases match full first two log-distance moments. All counts include extrapolation turns on [1,16].

| Estimator | Near mass | Tail | Final folded | Within stored range | Ever observed folded | Mean R² | Mean backward travel (%) | Mean weight ESS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native_paired | native | native | 5/10 | 5/10 | 7/10 | 0.6119 | 17.3820 | 1024.0 |
| native_floor | native | native | 5/10 | 5/10 | 6/10 | 0.6286 | 17.5972 | 1020.1 |
| native_replay | native | native | 5/10 | 5/10 | 7/10 | 0.6398 | 17.9408 | 1024.0 |
| weighted | 25% | native | 0/10 | 0/10 | 0/10 | 0.9860 | 0.0000 | 205.0 |
| weighted | 25% | diffuse | 0/10 | 0/10 | 0/10 | 0.9883 | 0.0000 | 146.5 |
| weighted | 25% | one | 0/10 | 0/10 | 1/10 | 0.9572 | 0.0000 | 150.8 |
| weighted | 25% | two | 0/10 | 0/10 | 0/10 | 0.9768 | 0.0000 | 96.1 |
| weighted | 50% | native | 4/10 | 0/10 | 4/10 | 0.8880 | 0.0655 | 394.9 |
| weighted | 50% | diffuse | 10/10 | 10/10 | 10/10 | 0.7569 | 6.2216 | 295.4 |
| weighted | 50% | one | 0/10 | 0/10 | 0/10 | 0.9647 | 0.0000 | 300.3 |
| weighted | 50% | two | 0/10 | 0/10 | 0/10 | 0.9656 | 0.0000 | 203.8 |
| weighted | 87.5% | native | 10/10 | 10/10 | 10/10 | 0.5608 | 33.5343 | 966.7 |
| weighted | 87.5% | diffuse | 10/10 | 10/10 | 10/10 | 0.3511 | 36.5407 | 894.1 |
| weighted | 87.5% | one | 10/10 | 10/10 | 10/10 | 0.4950 | 34.4175 | 884.9 |
| weighted | 87.5% | two | 10/10 | 10/10 | 10/10 | 0.2050 | 43.0484 | 814.9 |
| weighted | 93.75% | native | 0/10 | 0/10 | 1/10 | 0.8569 | 0.0000 | 979.7 |
| weighted | 93.75% | diffuse | 0/10 | 0/10 | 1/10 | 0.8579 | 0.0000 | 955.5 |
| weighted | 93.75% | one | 3/10 | 3/10 | 5/10 | 0.7642 | 3.9986 | 953.0 |
| weighted | 93.75% | two | 2/10 | 2/10 | 4/10 | 0.7822 | 3.4520 | 933.6 |
| replay | 87.5% | native | 10/10 | 10/10 | 10/10 | 0.5179 | 34.4271 | 1024.0 |
| replay | 87.5% | diffuse | 10/10 | 10/10 | 10/10 | 0.3772 | 35.4096 | 1024.0 |
| replay | 87.5% | one | 10/10 | 10/10 | 10/10 | 0.5927 | 35.5278 | 1024.0 |
| replay | 87.5% | two | 10/10 | 10/10 | 10/10 | 0.1514 | 43.7017 | 1024.0 |

## Paired one-bump versus two-bump comparisons

| Estimator | Near mass | Both folded | Only one-bump folded | Only two-bump folded | Neither folded | Single-turn cases | Median turn ratio: two/one |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted | 25% | 0 | 0 | 0 | 10 | 0 | — |
| weighted | 50% | 0 | 0 | 0 | 10 | 0 | — |
| weighted | 87.5% | 10 | 0 | 0 | 0 | 10 | 1.1591 |
| weighted | 93.75% | 2 | 1 | 0 | 7 | 2 | 1.1267 |
| replay | 87.5% | 10 | 0 | 0 | 0 | 10 | 1.2582 |

[Graph gallery](index.html) · [Explanation and limitations](../README.md) · [All numerical metrics](summary.json) · [Verification](verification.json)

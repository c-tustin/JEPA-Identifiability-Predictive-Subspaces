# Gradient intervention results

| Stage | Input gain | Method | Final folded runs | Mean log-distance R² | Mean backward travel (%) | Minimum output std |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| scratch | 4 | Original SIGReg | 5/10 | 0.6115 | 17.38203 | 0.6831 |
| scratch | 4 | Balance each distance bin | 0/10 | 0.9820 | 0.00000 | 0.8502 |
| scratch | 4 | Gentler correction (square root) | 9/10 | 0.1421 | 35.65613 | 0.6265 |
| scratch | 4 | Full correction, cap extreme weights | 0/10 | 0.9848 | 0.00000 | 0.8502 |
| scratch | 4 | Balance bins + discourage doubling back | 0/10 | 0.9815 | 0.00000 | 0.8383 |
| scratch | 4 | Original SIGReg + discourage doubling back | 7/10 | 0.3529 | 10.65839 | 0.5650 |
| scratch | 8 | Original SIGReg | 9/10 | 0.4263 | 30.95509 | 0.7106 |
| scratch | 8 | Balance each distance bin | 7/10 | 0.9186 | 6.26296 | 0.7670 |
| scratch | 8 | Balance bins + discourage doubling back | 7/10 | 0.9249 | 6.10815 | 0.7845 |
| scratch | 8 | Balance bins + strong order prior | 4/10 | 0.9598 | 0.00424 | 0.8261 |
| repair | 4 | Original SIGReg | 5/5 | 0.4232 | 36.43227 | 0.6721 |
| repair | 4 | Balance each distance bin | 5/5 | 0.0327 | 39.87984 | 0.9061 |
| repair | 4 | Balance bins + discourage doubling back | 5/5 | 0.6490 | 23.39492 | 0.8400 |
| repair | 4 | Balance bins + reset Adam | 5/5 | 0.0280 | 40.07111 | 0.9071 |
| repair | 4 | Balance bins + strong order prior | 0/5 | 0.9901 | 0.00000 | 0.8749 |
| repair | 4 | Strong order prior on stored observations | 0/5 | 0.9900 | 0.00000 | 0.8754 |

## Held-out predictor error: same evolving policy, new trajectories

| Seed | Update 100 | Update 600 | Update 1,800 | Frozen-target fitting: start → end |
| ---: | ---: | ---: | ---: | ---: |
| 130 | 0.002803 | 0.088398 | 0.061595 | 0.079599 → 0.027789 |
| 131 | 0.001951 | 0.084107 | 0.060102 | 0.087716 → 0.002423 |
| 136 | 0.000794 | 0.147100 | 0.044451 | 0.136379 → 0.003318 |

Error means half the summed squared error over 32 features, averaged over transitions. The frozen test has fixed targets and a separate held-out dataset; it is not a continuation of the moving-policy error curve.

Backward variation in summary metrics is sign-invariant: min(total upward variation, total downward variation) / total variation. The stored training frames also retain the original fixed-orientation backward measure; output sign is arbitrary.

[Explanation](../README.md) · [Graph gallery](index.html) · [Gradient metrics](gradient-summary.json) · [Predictor metrics](predictor-summary.json) · [Verification](verification.json)

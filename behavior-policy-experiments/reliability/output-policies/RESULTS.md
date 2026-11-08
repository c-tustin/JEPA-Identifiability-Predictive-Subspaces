# Reliability study results

One distance coordinate; settings fixed before inspecting these results. Seeds 3 and 7 diagnose previous failures; seeds 8–12 are new validation runs. The new runner separates initialization and batch randomness. Its ordinary ICM and OU runs replay the earlier baseline streams when data and initialization seeds match.

## Validation

| Condition | Runs | Shared-range $R^2$ | Broader OU $R^2$ | Reversed grid steps |
| --- | ---: | ---: | ---: | ---: |
| action_balanced | 5 | 0.995714 ± 0.001159 | 0.985001 ± 0.002576 | 0.000000 ± 0.000000 |
| count | 5 | 0.995908 ± 0.002275 | 0.983042 ± 0.006459 | 0.000000 ± 0.000000 |
| disagreement | 5 | 0.517587 ± 0.294229 | 0.486454 ± 0.304000 | 0.289736 ± 0.166023 |
| rnd | 5 | 0.467170 ± 0.294623 | 0.441996 ± 0.300243 | 0.304008 ± 0.170223 |

## Individual runs

| Data seed | Init seed | Condition | Shared $R^2$ | Broader $R^2$ |
| --- | --- | --- | ---: | ---: |
| 8 | 8 | rnd | 0.991148 | 0.972713 |
| 8 | 8 | disagreement | 0.991418 | 0.973255 |
| 8 | 8 | count | 0.995866 | 0.982251 |
| 8 | 8 | action_balanced | 0.994259 | 0.982158 |
| 9 | 9 | rnd | 0.337494 | 0.288306 |
| 9 | 9 | disagreement | 0.336269 | 0.285920 |
| 9 | 9 | count | 0.999178 | 0.992299 |
| 9 | 9 | action_balanced | 0.995965 | 0.985241 |
| 10 | 10 | rnd | 0.371792 | 0.378573 |
| 10 | 10 | disagreement | 0.617760 | 0.596516 |
| 10 | 10 | count | 0.996754 | 0.985920 |
| 10 | 10 | action_balanced | 0.996563 | 0.988801 |
| 11 | 11 | rnd | 0.349862 | 0.317648 |
| 11 | 11 | disagreement | 0.350857 | 0.318476 |
| 11 | 11 | count | 0.994577 | 0.979356 |
| 11 | 11 | action_balanced | 0.994794 | 0.983129 |
| 12 | 12 | rnd | 0.285555 | 0.252741 |
| 12 | 12 | disagreement | 0.291631 | 0.258100 |
| 12 | 12 | count | 0.993165 | 0.975383 |
| 12 | 12 | action_balanced | 0.996989 | 0.985675 |

Means ± sample standard deviations. No best-seed selection or distance-label tuning. Policy choices use the apparent-size geometry where stated; the monotone encoder adds an ordering prior. Long runs have twice the representation budget. This comparison is not a universal ranking of exploration algorithms.

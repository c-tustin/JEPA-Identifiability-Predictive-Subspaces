# Reliability study results

One distance coordinate; settings fixed before inspecting these results. Seeds 3 and 7 diagnose previous failures; seeds 8–12 are new validation runs. The new runner separates initialization and batch randomness. Its ordinary ICM and OU runs replay the earlier baseline streams when data and initialization seeds match.

## Diagnostic

| Condition | Runs | Shared-range $R^2$ | Broader OU $R^2$ | Reversed grid steps |
| --- | ---: | ---: | ---: | ---: |
| icm | 24 | 0.850531 ± 0.298665 | 0.841386 ± 0.295687 | 0.080849 ± 0.143910 |
| ou | 24 | 0.995042 ± 0.001251 | 0.981216 ± 0.002879 | 0.000000 ± 0.000000 |

## Individual runs

| Data seed | Init seed | Condition | Shared $R^2$ | Broader $R^2$ |
| --- | --- | --- | ---: | ---: |
| 0 | 100 | icm | 0.998886 | 0.991019 |
| 0 | 101 | icm | 0.998237 | 0.988350 |
| 0 | 102 | icm | 0.998885 | 0.989884 |
| 0 | 100 | ou | 0.996436 | 0.982246 |
| 0 | 101 | ou | 0.995832 | 0.980562 |
| 0 | 102 | ou | 0.993501 | 0.975787 |
| 1 | 100 | icm | 0.998524 | 0.995241 |
| 1 | 101 | icm | 0.999672 | 0.994481 |
| 1 | 102 | icm | 0.999266 | 0.992271 |
| 1 | 100 | ou | 0.996011 | 0.982950 |
| 1 | 101 | ou | 0.995464 | 0.981530 |
| 1 | 102 | ou | 0.993249 | 0.977213 |
| 2 | 100 | icm | 0.998944 | 0.996410 |
| 2 | 101 | icm | 0.999608 | 0.996207 |
| 2 | 102 | icm | 0.999409 | 0.996029 |
| 2 | 100 | ou | 0.995331 | 0.983871 |
| 2 | 101 | ou | 0.994856 | 0.982761 |
| 2 | 102 | ou | 0.992773 | 0.978904 |
| 3 | 100 | icm | 0.036003 | 0.066900 |
| 3 | 101 | icm | 0.719652 | 0.647134 |
| 3 | 102 | icm | 0.642865 | 0.584679 |
| 3 | 100 | ou | 0.996463 | 0.986412 |
| 3 | 101 | ou | 0.995844 | 0.984869 |
| 3 | 102 | ou | 0.993639 | 0.980438 |
| 4 | 100 | icm | 0.998563 | 0.995613 |
| 4 | 101 | icm | 0.999596 | 0.993716 |
| 4 | 102 | icm | 0.999200 | 0.992132 |
| 4 | 100 | ou | 0.996243 | 0.983591 |
| 4 | 101 | ou | 0.995542 | 0.981820 |
| 4 | 102 | ou | 0.993440 | 0.977710 |
| 5 | 100 | icm | 0.998270 | 0.995735 |
| 5 | 101 | icm | 0.999714 | 0.995067 |
| 5 | 102 | icm | 0.999281 | 0.993010 |
| 5 | 100 | ou | 0.996424 | 0.984788 |
| 5 | 101 | ou | 0.995732 | 0.983059 |
| 5 | 102 | ou | 0.993566 | 0.978810 |
| 6 | 100 | icm | 0.998344 | 0.995419 |
| 6 | 101 | icm | 0.999607 | 0.995677 |
| 6 | 102 | icm | 0.999300 | 0.995068 |
| 6 | 100 | ou | 0.996119 | 0.983404 |
| 6 | 101 | ou | 0.995656 | 0.982143 |
| 6 | 102 | ou | 0.993482 | 0.977828 |
| 7 | 100 | icm | 0.020338 | 0.057813 |
| 7 | 101 | icm | 0.544822 | 0.507865 |
| 7 | 102 | icm | 0.465749 | 0.437536 |
| 7 | 100 | ou | 0.996228 | 0.981954 |
| 7 | 101 | ou | 0.995647 | 0.980439 |
| 7 | 102 | ou | 0.993532 | 0.976090 |

Means ± sample standard deviations. No best-seed selection or distance-label tuning. Policy choices use the apparent-size geometry where stated; the monotone encoder adds an ordering prior. Long runs have twice the representation budget. This comparison is not a universal ranking of exploration algorithms.

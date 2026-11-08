# Learned OU parameters

Five independent seeds 13–17; no distance-labelled tuning. Simulator OU and the physical controller are separate experiments.

| Condition | Runs | Shared-range $R^2$ | Broader fixed-OU $R^2$ |
| --- | ---: | ---: | ---: |
| fixed_ou | 5 | 0.994897 ± 0.001245 | 0.981019 ± 0.003256 |
| fixed_gap_rho | 5 | 0.994901 ± 0.001069 | 0.981022 ± 0.002867 |
| learn_rho | 5 | 0.994906 ± 0.001069 | 0.981034 ± 0.002871 |
| learn_spread | 5 | 0.995849 ± 0.001417 | 0.983134 ± 0.003580 |
| learn_both | 5 | 0.995902 ± 0.001255 | 0.983249 ± 0.003203 |
| physical_only | 5 | 0.948075 ± 0.044865 | 0.921826 ± 0.054214 |
| physical_to_ou | 5 | 0.992230 ± 0.001382 | 0.975581 ± 0.003480 |

| Seed | Generator mode | Learned correlation | Learned spread |
| --- | --- | ---: | ---: |
| 13 | learn_rho | 0.50188 | 0.70000 |
| 13 | learn_spread | 0.90000 | 0.73428 |
| 13 | learn_both | 0.50188 | 0.73428 |
| 14 | learn_rho | 0.50015 | 0.70000 |
| 14 | learn_spread | 0.90000 | 0.72668 |
| 14 | learn_both | 0.50015 | 0.72668 |
| 15 | learn_rho | 0.50730 | 0.70000 |
| 15 | learn_spread | 0.90000 | 0.72930 |
| 15 | learn_both | 0.50730 | 0.72930 |
| 16 | learn_rho | 0.50293 | 0.70000 |
| 16 | learn_spread | 0.90000 | 0.72889 |
| 16 | learn_both | 0.50293 | 0.72889 |
| 17 | learn_rho | 0.50459 | 0.70000 |
| 17 | learn_spread | 0.90000 | 0.74871 |
| 17 | learn_both | 0.50459 | 0.74871 |

The generator maximizes an observed-correlation spectral-gap surrogate plus observed-size coverage, with an outside-range penalty. It has simulator access for reparameterized sampling; it is not a physical action policy. The physical controller learns target correlation, noise and center through count-reward actor–critic and retains the three original actions. Its realized transition need not be OU. The known-Gaussian gap is maximized at correlation 0.5, so a fixed 0.5 control tests whether parameter learning adds anything beyond that analytic choice.

# Every comparable discovery condition

Same seeds 8–27, 4,000 encoder updates. Rankings are descriptive; fresh tests are in the short results. Physical policies and abstract Gaussian pair sources are labelled separately. The 8,000-update exploratory runs are excluded from this equal-budget table.

## Pure

| Policy/source | Kind | Mean R² | SD | Folds |
| --- | --- | ---: | ---: | ---: |
| iid_ou_hp_w0.1 | Gaussian pair source | 0.995107 | 0.001293 | 0/20 |
| iid_ou_hp_w0.3 | Gaussian pair source | 0.995075 | 0.001264 | 0/20 |
| ou_hp_w0.1 | Gaussian pair source | 0.995056 | 0.001447 | 0/20 |
| ou_hp_w0.3 | Gaussian pair source | 0.995043 | 0.001382 | 0/20 |
| ou | Gaussian pair source | 0.995026 | 0.001324 | 0/20 |
| ou_traj | Gaussian pair source | 0.994991 | 0.001364 | 0/20 |
| iid_ou_lp_w1 | Gaussian pair source | 0.994942 | 0.001337 | 0/20 |
| ou_lp_w1 | Gaussian pair source | 0.994929 | 0.001379 | 0/20 |
| iid_ou_lp_w0.3 | Gaussian pair source | 0.994883 | 0.001335 | 0/20 |
| ou_lp_w0.3 | Gaussian pair source | 0.994880 | 0.001437 | 0/20 |
| iid_ou_lp_w0.1 | Gaussian pair source | 0.994867 | 0.001329 | 0/20 |
| ou_lp_w0.1 | Gaussian pair source | 0.994695 | 0.001536 | 0/20 |
| ou_hp_w1 | Gaussian pair source | 0.994490 | 0.001160 | 0/20 |
| iid_ou_hp_w1 | Gaussian pair source | 0.994476 | 0.001116 | 0/20 |
| ou_hp2_w1 | Gaussian pair source | 0.993691 | 0.000998 | 0/20 |
| iid_ou_hp2_w1 | Gaussian pair source | 0.993617 | 0.000946 | 0/20 |
| iid_ou_hp2_w3 | Gaussian pair source | 0.992953 | 0.000784 | 0/20 |
| ou_hp2_w3 | Gaussian pair source | 0.992945 | 0.000802 | 0/20 |
| gaussian_wide_diffusion | new physical | 0.991308 | 0.002392 | 0/20 |
| gaussian_target_slow | new physical | 0.990451 | 0.001553 | 0/20 |
| gaussian_target_fast | new physical | 0.990168 | 0.001233 | 0/20 |
| gaussian_deficit | new physical | 0.989928 | 0.001238 | 0/20 |
| gaussian_walk | new physical | 0.989444 | 0.000833 | 0/20 |
| gaussian_start_random | new physical | 0.989109 | 0.003876 | 0/20 |
| gaussian_log_diffusion | new physical | 0.988682 | 0.002532 | 0/20 |
| count | old physical | 0.984474 | 0.010168 | 0/20 |
| adaptive_bias | new physical | 0.975105 | 0.006246 | 0/20 |
| no_escort_log_diffusion | new physical | 0.971930 | 0.006558 | 0/20 |
| homing | old physical | 0.966676 | 0.002655 | 0/20 |
| ou_actions_hp_w0.3 | old physical | 0.965334 | 0.009876 | 0/20 |
| action_balanced | old physical | 0.961882 | 0.005865 | 0/20 |
| bias_sweep_mix | new physical | 0.961717 | 0.005772 | 0/20 |
| guarded_sweep | new physical | 0.961246 | 0.006457 | 0/20 |
| random | old physical | 0.960914 | 0.010754 | 0/20 |
| ou_actions_hp2_w3 | old physical | 0.958831 | 0.008620 | 0/20 |
| guarded_bias | new physical | 0.957666 | 0.011534 | 0/20 |
| biased | old physical | 0.957489 | 0.011531 | 0/20 |
| lazy | old physical | 0.956706 | 0.008413 | 0/20 |
| gentle_bias | new physical | 0.955620 | 0.010624 | 0/20 |
| ou_actions | old physical | 0.936460 | 0.042939 | 2/20 |
| sweep | old physical | 0.931423 | 0.007430 | 0/20 |
| levy | old physical | 0.907277 | 0.188140 | 1/20 |
| sticky | old physical | 0.901821 | 0.197722 | 1/20 |
| ou_actions_lp_w0.1 | old physical | 0.857957 | 0.236259 | 2/20 |
| task | old physical | 0.822925 | 0.347709 | 3/20 |
| icm | old physical | 0.603374 | 0.344675 | 10/20 |
| disagreement | old physical | 0.062416 | 0.200277 | 19/20 |
| rnd | old physical | 0.051793 | 0.160566 | 20/20 |

## Handover

| Policy/source | Kind | Mean R² | SD | Folds |
| --- | --- | ---: | ---: | ---: |
| guarded_bias | new physical | 0.998259 | 0.000617 | 0/20 |
| biased | old physical | 0.998258 | 0.000618 | 0/20 |
| ou_actions_lp_w0.1 | old physical | 0.998233 | 0.000898 | 0/20 |
| adaptive_bias | new physical | 0.997945 | 0.000548 | 0/20 |
| sweep | old physical | 0.997901 | 0.000696 | 0/20 |
| gentle_bias | new physical | 0.997683 | 0.000792 | 0/20 |
| no_escort_log_diffusion | new physical | 0.997621 | 0.000648 | 0/20 |
| ou_actions | old physical | 0.997549 | 0.001070 | 0/20 |
| levy | old physical | 0.997384 | 0.000936 | 0/20 |
| random | old physical | 0.996553 | 0.001020 | 0/20 |
| lazy | old physical | 0.996403 | 0.001013 | 0/20 |
| guarded_sweep | new physical | 0.996303 | 0.001021 | 0/20 |
| bias_sweep_mix | new physical | 0.996286 | 0.001078 | 0/20 |
| ou_actions_hp2_w3 | old physical | 0.996273 | 0.001038 | 0/20 |
| action_balanced | old physical | 0.996237 | 0.000908 | 0/20 |
| ou_actions_hp_w0.3 | old physical | 0.996208 | 0.000854 | 0/20 |
| count | old physical | 0.996107 | 0.002086 | 0/20 |
| gaussian_start_random | new physical | 0.995578 | 0.001725 | 0/20 |
| gaussian_wide_diffusion | new physical | 0.993732 | 0.001512 | 0/20 |
| gaussian_target_fast | new physical | 0.992888 | 0.001062 | 0/20 |
| gaussian_target_slow | new physical | 0.992667 | 0.001114 | 0/20 |
| gaussian_log_diffusion | new physical | 0.992127 | 0.001142 | 0/20 |
| gaussian_deficit | new physical | 0.992070 | 0.000975 | 0/20 |
| gaussian_walk | new physical | 0.992045 | 0.000723 | 0/20 |
| homing | old physical | 0.974923 | 0.004251 | 0/20 |
| sticky | old physical | 0.954858 | 0.191328 | 1/20 |
| task | old physical | 0.930175 | 0.199250 | 2/20 |
| icm | old physical | 0.817981 | 0.213180 | 9/20 |
| disagreement | old physical | 0.458429 | 0.213398 | 18/20 |
| rnd | old physical | 0.378235 | 0.175896 | 19/20 |

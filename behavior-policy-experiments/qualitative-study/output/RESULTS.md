# Ten-trial results: representation folds, planning and coverage

Smaller physical goal error is better. Every row is a matched seed; the same held-out goals and predictor-training data compare the three main encoders. ICM + OU means ICM→OU.

| Seed | ICM error | OU error | ICM→OU error | Curriculum − OU | Passed separate screen? |
| --- | ---: | ---: | ---: | ---: | --- |
| 130 | 1.4157 | 0.1328 | 0.0781 | -0.0547 | yes |
| 131 | 0.2227 | 0.0430 | 0.2188 | +0.1758 | no |
| 132 | 0.2266 | 0.0586 | 0.2539 | +0.1953 | no |
| 133 | 0.2321 | 0.2070 | 0.2281 | +0.0211 | no |
| 134 | 0.5977 | 0.0469 | 0.0000 | -0.0469 | no |
| 135 | 0.4844 | 0.0195 | 0.1484 | +0.1289 | no |
| 136 | 1.9588 | 0.2227 | 0.7695 | +0.5469 | no |
| 137 | 0.6250 | 0.0859 | 0.1367 | +0.0508 | yes |
| 138 | 1.2850 | 0.0781 | 0.1523 | +0.0742 | no |
| 139 | 1.6657 | 0.0938 | 1.6696 | +1.5759 | no |

## Statistical comparison

The seed is the independent unit. The exact sign test asks whether the curriculum wins more often than it loses; it does not test the mean error. Both sign-test p-values below are Holm-adjusted. Mean changes and paired bootstrap intervals are also reported.

| Comparison | Seeds | Wins / losses / ties | Mean error change | 95% paired-seed bootstrap interval | Adjusted sign-test p |
| --- | ---: | --- | ---: | --- | ---: |
| all_trials | 10 | 2 / 8 / 0 | 0.2667 | [+0.0465, +0.5911] | 0.2188 |
| independent_screen | 2 | 1 / 1 / 0 | -0.0020 | [-0.0547, +0.0508] | 1.0000 |

The subgroup is selected using separate validation goals and a fold check, never the held-out errors in the table. Its conclusion is conditional on passing this particular screen. A small subgroup has little power; testing only the held-out winners would be invalid. Supplementary paired t-tests and their assumptions are in [statistics.json](statistics.json).

The percentile bootstrap interval describes the mean, whereas the sign test concerns win probability. With only ten seeds and skewed differences, the bootstrap interval is also fragile; it is descriptive, not a substitute for the specified tests.

## What to inspect qualitatively

- **ICM:** 5/10 runs have a substantial grid reversal; average shared-data goal error 0.8714.
- **OU:** 0/10 runs have a substantial grid reversal; average shared-data goal error 0.0988.
- **ICM → OU:** 5/10 runs have a substantial grid reversal; average shared-data goal error 0.3656.

Each representation panel shows the learned coordinate against physical distance, with only affine normalization. Red curve segments reverse direction. A fold maps different distances to similar outputs; a monotone curve can still compress useful distances or make action prediction difficult. These plots diagnose geometry; they do not prove an optimization mechanism.

Fold location also matters: a reversal mostly outside the routes used by our local goals can look less damaging than a reversal across the center of the task range. The following uses oracle routes only as an evaluation diagnostic, not as training data or a success screen.

| ICM seed | Reversed distance span | Oracle positions in reversed intervals | Shared-data goal error |
| --- | --- | ---: | ---: |
| 130 | none | 0.0% | 1.4157 |
| 131 | 1.00–2.37 | 1.2% | 0.2227 |
| 132 | 1.00–2.18 | 0.6% | 0.2266 |
| 133 | 1.00–2.65 | 4.1% | 0.2321 |
| 134 | none | 0.0% | 0.5977 |
| 135 | none | 0.0% | 0.4844 |
| 136 | 3.15–16.00 | 96.6% | 1.9588 |
| 137 | none | 0.0% | 0.6250 |
| 138 | none | 0.0% | 1.2850 |
| 139 | 1.00–1.91 | 1.0% | 1.6657 |

## Binning and planning

| SIGReg sampling in ICM→OU | Shared-data mean goal error | Native-data mean goal error | Mean SIGReg exposure entropy |
| --- | ---: | ---: | ---: |
| ICM → OU | 0.3656 | 1.9064 | 3.086 |
| Balanced log-size | 0.1004 | 1.2422 | 3.399 |
| Balanced raw size | 0.1016 | 1.2348 | 3.360 |

Replay reweights existing observations; it does not add unseen states or missing action examples. Log-size and raw-size bins define different target sampling measures for SIGReg. Shared-data planning holds the predictor data fixed; native-data planning also exposes the physical action-coverage problem. Passive OU has no physical action labels, so an OU-native predictor is not fabricated.

## Which distance regions matter?

The controlled intervention freezes each OU encoder and removes one region from predictor training. Positive error change means that removal harmed planning. This is different from correlating a coverage histogram with error.

| Removed distance region | Mean error increase | Tasks whose oracle route crosses it | Adjusted sign-test p |
| --- | ---: | ---: | ---: |
| 11.31–16.00 | +0.0277 | 28.0% | 1.0000 |
| 2.83–4.00 | +0.0219 | 13.4% | 1.0000 |
| 5.66–8.00 | +0.0191 | 32.8% | 1.0000 |
| 4.00–5.66 | +0.0152 | 24.7% | 1.0000 |
| 8.00–11.31 | +0.0141 | 38.0% | 1.0000 |
| 2.00–2.83 | +0.0121 | 3.8% | 1.0000 |
| 1.41–2.00 | -0.0047 | 1.1% | 1.0000 |
| 1.00–1.41 | -0.0520 | 0.5% | 1.0000 |

These are eight four-bin regions, not separately identified individual-bin effects. The result is specific to these local goals, frozen OU encoders, predictor family and budget. Removing data can also change finite-fit behavior globally; negative changes are retained. Per-bin correlations are exploratory and do not establish causal importance.

[Representation curves](figures/representations_overlay.png) · [Planning wins/losses](figures/curriculum_wins_and_losses.png) · [Binning comparison](figures/binning_exposure_and_planning.png) · [Controlled region importance](figures/controlled_region_importance.png)

# Learning-rate follow-up results

Three paired historical sources, seeds 130, 131, 136; 4,000 updates from initialization. 15 new fits, nine reused original controls. All 15 new checkpoint curves/slopes reproduced exactly after reload.

| Method | Rate multiplier | Final folds | Mean log-distance R² |
| --- | --- | --- | --- |
| ordinary | 1.0 | 2/3 | 0.4826 |
| inverse_gradient | 0.25 | 2/3 | 0.5336 |
| inverse_gradient | 1.0 | 1/3 | 0.6377 |
| inverse_gradient | 4.0 | 2/3 | 0.4790 |
| inverse_gradient | Per-step norm matched | 3/3 | 0.4766 |
| weighted_moments | 0.25 | 0/3 | 0.9878 |
| weighted_moments | 1.0 | 0/3 | 0.9841 |
| weighted_moments | 4.0 | 0/3 | 0.9804 |

Norm matching compensates the full raw gradient norm before Adam, not the final Adam step. Folds are sampled sign reversals. These are prevention experiments with privileged occupancy estimation; no fresh actors or late-repair runs.

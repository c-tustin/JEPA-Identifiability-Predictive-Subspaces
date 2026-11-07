# Planning evaluation

Frozen encoders; observation/action-trained predictors; identical reachable goals and candidate sequences. Each run uses 32 cases, 12 replanning steps, horizon 8 and 128 candidate sequences. The planner receives only current and goal embeddings, never true distance.

| Encoder | Predictor data | Seeds | Terminal distance error ↓ | Within 0.25 ↑ | One-step latent MSE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| icm_carry | common | 8 | 0.272461 ± 0.323453 | 0.867188 ± 0.198171 | 0.003051 ± 0.005034 |
| icm_carry | native | 8 | 1.613664 ± 0.900641 | 0.281250 ± 0.282487 | 0.075229 ± 0.073853 |
| ou_carry | common | 8 | 0.081055 ± 0.061507 | 0.988281 ± 0.016173 | 0.000180 ± 0.000083 |
| random_carry | common | 8 | 0.057617 ± 0.054757 | 0.984375 ± 0.033408 | 0.000214 ± 0.000098 |
| random_carry | native | 8 | 0.086914 ± 0.067650 | 0.996094 ± 0.011049 | 0.000198 ± 0.000084 |

Common data holds predictor-training coverage fixed to compare representations. Native data also changes coverage and tests the full pipeline. Native OU is omitted because passive OU pairs have no physical action labels. A lower latent prediction loss need not mean better physical planning. Goals are local and reachable; this is an initial short-horizon planning test, not arbitrary-distance or uncertainty-aware planning. Cases within a seed share a trained model; averages across seeds, not case-level significance tests, are reported.

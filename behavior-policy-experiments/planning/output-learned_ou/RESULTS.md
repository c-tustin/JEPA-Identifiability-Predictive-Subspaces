# Planning evaluation

Frozen encoders; observation/action-trained predictors; identical reachable goals and candidate sequences. Each run uses 32 cases, 12 replanning steps, horizon 8 and 128 candidate sequences. The planner receives only current and goal embeddings, never true distance.

| Encoder | Predictor data | Seeds | Terminal distance error ↓ | Within 0.25 ↑ | One-step latent MSE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed_gap_rho | common | 5 | 0.093750 ± 0.062500 | 0.993750 ± 0.013975 | 0.000236 ± 0.000123 |
| fixed_ou | common | 5 | 0.082812 ± 0.059652 | 0.993750 ± 0.013975 | 0.000238 ± 0.000126 |
| learn_both | common | 5 | 0.101562 ± 0.053844 | 0.975000 ± 0.055902 | 0.000255 ± 0.000131 |
| learn_rho | common | 5 | 0.093750 ± 0.062500 | 0.993750 ± 0.013975 | 0.000236 ± 0.000123 |
| learn_spread | common | 5 | 0.101562 ± 0.055518 | 0.981250 ± 0.041926 | 0.000256 ± 0.000134 |
| physical_only | common | 5 | 0.275000 ± 0.264900 | 0.831250 ± 0.189572 | 0.000536 ± 0.000281 |
| physical_only | native | 5 | 1.270901 ± 0.956591 | 0.462500 ± 0.330276 | 0.005044 ± 0.008306 |
| physical_to_ou | common | 5 | 0.096875 ± 0.073413 | 0.950000 ± 0.081490 | 0.000215 ± 0.000109 |
| physical_to_ou | native | 5 | 1.017187 ± 0.817917 | 0.525000 ± 0.319759 | 0.009968 ± 0.013911 |

Common data holds predictor-training coverage fixed to compare representations. Native data also changes coverage and tests the full pipeline. Native OU is omitted because passive OU pairs have no physical action labels. A lower latent prediction loss need not mean better physical planning. Goals are local and reachable; this is an initial short-horizon planning test, not arbitrary-distance or uncertainty-aware planning. Cases within a seed share a trained model; averages across seeds, not case-level significance tests, are reported.

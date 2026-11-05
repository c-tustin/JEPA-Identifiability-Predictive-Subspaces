# Planning evaluation

Frozen encoders; observation/action-trained predictors; identical reachable goals and candidate sequences. Each run uses 32 cases, 12 replanning steps, horizon 8 and 128 candidate sequences. The planner receives only current and goal embeddings, never true distance.

| Encoder | Predictor data | Seeds | Terminal distance error ↓ | Within 0.25 ↑ | One-step latent MSE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| icm | common | 7 | 0.380580 ± 0.291261 | 0.781250 ± 0.167316 | 0.009634 ± 0.004745 |
| icm | native | 7 | 1.846543 ± 0.699039 | 0.169643 ± 0.098586 | 0.090910 ± 0.055152 |
| icm_coverage | common | 7 | 0.262277 ± 0.117608 | 0.848214 ± 0.073129 | 0.008533 ± 0.005385 |
| icm_coverage | native | 7 | 1.691534 ± 0.587222 | 0.151786 ± 0.079526 | 0.080132 ± 0.052497 |
| icm_epsilon | common | 7 | 0.251116 ± 0.161149 | 0.852679 ± 0.135874 | 0.007067 ± 0.006013 |
| icm_epsilon | native | 7 | 1.547732 ± 0.737196 | 0.281250 ± 0.318689 | 0.041616 ± 0.028720 |
| monotone | common | 7 | 0.869158 ± 0.514989 | 0.513393 ± 0.160217 | 0.009231 ± 0.006778 |
| monotone | native | 7 | 1.734641 ± 0.665340 | 0.183036 ± 0.169662 | 0.016410 ± 0.015382 |
| ou | common | 7 | 0.072545 ± 0.050573 | 0.982143 ± 0.035434 | 0.000202 ± 0.000112 |
| random | common | 7 | 0.085676 ± 0.069976 | 0.991071 ± 0.015248 | 0.000223 ± 0.000113 |
| random | native | 7 | 0.088170 ± 0.057003 | 1.000000 ± 0.000000 | 0.000223 ± 0.000111 |

Common data holds predictor-training coverage fixed to compare representations. Native data also changes coverage and tests the full pipeline. Native OU is omitted because passive OU pairs have no physical action labels. A lower latent prediction loss need not mean better physical planning. Goals are local and reachable; this is an initial short-horizon planning test, not arbitrary-distance or uncertainty-aware planning. Cases within a seed share a trained model; averages across seeds, not case-level significance tests, are reported.

# Planning evaluation

Frozen encoders; observation/action-trained predictors; identical reachable goals and candidate sequences. Each run uses 32 cases, 12 replanning steps, horizon 8 and 128 candidate sequences. The planner receives only current and goal embeddings, never true distance.

| Encoder | Predictor data | Seeds | Terminal distance error ↓ | Within 0.25 ↑ | One-step latent MSE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| action_balanced | common | 5 | 0.109009 ± 0.069082 | 0.975000 ± 0.040745 | 0.000265 ± 0.000088 |
| action_balanced | native | 5 | 0.115259 ± 0.030680 | 0.937500 ± 0.091109 | 0.000278 ± 0.000107 |
| count | common | 5 | 0.078125 ± 0.045888 | 1.000000 ± 0.000000 | 0.000274 ± 0.000123 |
| count | native | 5 | 0.117554 ± 0.030119 | 0.956250 ± 0.052291 | 0.000265 ± 0.000124 |
| disagreement | common | 5 | 0.405521 ± 0.266186 | 0.806250 ± 0.164471 | 0.008548 ± 0.005357 |
| disagreement | native | 5 | 1.259375 ± 0.712276 | 0.362500 ± 0.303463 | 0.034819 ± 0.020991 |
| rnd | common | 5 | 0.377396 ± 0.277933 | 0.831250 ± 0.185668 | 0.008508 ± 0.005423 |
| rnd | native | 5 | 1.431250 ± 0.720624 | 0.325000 ± 0.328942 | 0.030935 ± 0.018954 |

Common data holds predictor-training coverage fixed to compare representations. Native data also changes coverage and tests the full pipeline. Native OU is omitted because passive OU pairs have no physical action labels. A lower latent prediction loss need not mean better physical planning. Goals are local and reachable; this is an initial short-horizon planning test, not arbitrary-distance or uncertainty-aware planning. Cases within a seed share a trained model; averages across seeds, not case-level significance tests, are reported.

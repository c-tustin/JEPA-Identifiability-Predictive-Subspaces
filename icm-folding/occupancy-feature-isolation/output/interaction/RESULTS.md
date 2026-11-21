# raw InfoNCE encoder-gradient intervention

All 36 fits passed independent dense reload and source/plan/initializer checks. All six detached controls reproduce lambda-zero encoder parameters exactly.

| InfoNCE weight | Code detached | Any turn / 6 | Material / 6 |
| ---: | --- | ---: | ---: |
| 0 | False | 6 | 6 |
| 0.1 | False | 6 | 6 |
| 1 | False | 6 | 6 |
| 10 | False | 2 | 2 |
| 100 | False | 0 | 0 |
| 100 | True | 6 | 6 |

At weight 100, the changes in the original SIGReg/alignment objective are +6.2218, +6.3649, +6.2834, +6.6360, +6.2423, +6.6816. These are exact expected finite-pair-batch risks under the controlled joint law, not the plug-in full-data penalty with its multiplier changed. The formula passes exact enumeration on a two-transition toy case.

The attached/detached contrast isolates the effect of the discrimination gradient on the encoder. It does not establish an all-input ordering guarantee, prove every contrastive gradient is anti-folding, or identify a universal occupancy threshold. All tested weights and failures are retained.

[Measurements](summary.json) · [Independent audit](verification.json)

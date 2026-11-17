# ICM-occupancy-matched OU results

November 16, 2026. Twelve completed 4,000-update fits. Both losses use the same minibatch and encoder outputs in all arms.

| Seed | Data | Final folds (root distances) | Roots inside source support | 64-bin TV | KS distance | Mass below 2 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 130 | native | none | 0 | 0.0000 | 0.0000 | 0.9298 |
| 130 | ou-small | none | 0 | 0.0337 | 0.0263 | 0.9255 |
| 130 | ou-large | 6.465 | 1 | 0.0299 | 0.0278 | 0.9299 |
| 131 | native | 2.407 | 1 | 0.0000 | 0.0000 | 0.8464 |
| 131 | ou-small | 2.542 | 1 | 0.0569 | 0.0457 | 0.8476 |
| 131 | ou-large | 2.264 | 1 | 0.0184 | 0.0080 | 0.8463 |
| 134 | native | none | 0 | 0.0000 | 0.0000 | 0.9527 |
| 134 | ou-small | none | 0 | 0.0646 | 0.0544 | 0.9613 |
| 134 | ou-large | none | 0 | 0.0161 | 0.0069 | 0.9583 |
| 136 | native | 3.124 | 1 | 0.0000 | 0.0000 | 0.6858 |
| 136 | ou-small | 3.377 | 1 | 0.0586 | 0.0468 | 0.6567 |
| 136 | ou-large | 3.213 | 1 | 0.0241 | 0.0112 | 0.6749 |

| Arm | Final folded / 4 | Any sampled training fold / 4 | Mean 64-bin TV | Mean per-seed median absolute step |
| --- | ---: | ---: | ---: | --- |
| native | 2/4 | 4/4 | 0.0000 | 0.1886 |
| ou-small | 2/4 | 4/4 | 0.0534 | 0.0254 |
| ou-large | 3/4 | 4/4 | 0.0221 | 0.0750 |

All saved final float32 curves/slopes reproduce exactly after checkpoint reload. Final roots were independently recomputed on 20,001 float64 log-distance points. Roots inside source support are distinguished from extrapolation; the gray bands in the gallery mark unobserved distances. History checks use a coarser 1,025-point grid every 100 updates.

The OU rank map targets the entire empirical endpoint distribution without smoothing, including exact empty bins and source multiplicities. Actual finite correlated rollouts only approximate that distribution; TV and KS above measure this approximation, not independent-sample significance. No fresh seeds, learned OU actor, bounded-action equivalence or general injectivity guarantee.

Matching occupancy does not match temporal pairs: OU targets the pooled stationary marginal, whereas ICM retains its recorded transitions. Both new losses share inputs in the native control too, so its fold outcomes need not equal the historical separate-regularizer-input control.

See [histograms/encoders/slopes](icm_matched_histograms_encoders.png), [fold histories](fold_histories.png), [multimodal gallery](bimodal_trimodal_shared_inputs.png), [mapping diagram](ou_quantile_transform.png), [full metrics](summary.json), and [verification](verification.json).

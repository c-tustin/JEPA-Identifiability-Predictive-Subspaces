# Finite-time action effects and independent-noise sampling

Four interior source states. Each action has independently sampled noise. This is a simulator diagnostic, not an encoder-training result. [Derivation and limits](../DRIFT_ESTIMATION.md).

256 independent groups, 128 endpoints per action/group, 64 Euler microsteps per interval.

| Interval | Empirical / leading variance | Largest deterministic action-effect bias | First-order remainder |
| ---: | ---: | ---: | ---: |
| 0.02 | 0.941 | 0.000682608 | 2.64472e-06 |
| 0.05 | 0.985 | 0.00170581 | 1.65923e-05 |
| 0.10 | 0.955 | 0.00342009 | 6.67777e-05 |
| 0.20 | 1.025 | 0.00687248 | 0.000270212 |

The leading variance is $\sigma(d)^2/(2N\delta)$. Deterministic bias is measured against the instantaneous coefficient 0.25. Its first-order approximation is $0.25+0.25\delta(0.3\cos d)/2$. The deterministic calculation does not include stochastic conditional-mean corrections. Full per-state values and settings are retained in metrics.json.

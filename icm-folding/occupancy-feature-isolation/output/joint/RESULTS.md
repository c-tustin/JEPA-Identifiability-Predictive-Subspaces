# Joint-data temporal InfoNCE and scalar VICReg bridge

All 160 fits passed independent dense reload, source/initializer/plan, target-region and shared-input checks. Native corners recover the empirical joint distribution exactly. Every method receives the same sampled transition rows and one encoder forward per update.

| Endpoint near mass | Objective | Any turn / 10 | Material / 10 | Turns within stored range / 10 |
| --- | --- | ---: | ---: | ---: |
| native | sigreg | 8 | 8 | 8 |
| native | infonce | 0 | 0 | 0 |
| native | raw_infonce | 0 | 0 | 0 |
| native | vicreg | 5 | 0 | 5 |
| 0.5 | sigreg | 1 | 1 | 1 |
| 0.5 | infonce | 0 | 0 | 0 |
| 0.5 | raw_infonce | 0 | 0 | 0 |
| 0.5 | vicreg | 1 | 0 | 1 |
| 0.75 | sigreg | 10 | 10 | 10 |
| 0.75 | infonce | 0 | 0 | 0 |
| 0.75 | raw_infonce | 0 | 0 | 0 |
| 0.75 | vicreg | 5 | 1 | 5 |
| 0.9375 | sigreg | 0 | 0 | 0 |
| 0.9375 | infonce | 0 | 0 | 0 |
| 0.9375 | raw_infonce | 0 | 0 | 0 |
| 0.9375 | vicreg | 0 | 0 | 0 |

This is a joint-data bridge, not a pure-marginal intervention. Transition-category weights change with near mass, and the conditional endpoint profiles change by a bounded amount because cross-transition probability is held fixed. Maximum near/far conditional-mixture TV bounds are 0.00578 / 0.11363. These changes are retained and prevent interpreting this alone as an occupancy-only causal test. The separate endpoint-only study supplies that control.

Temporal and raw/own-code InfoNCE results are reported separately. Scalar VICReg can have small turns under these interventions; its tested behavior is not an injectivity guarantee. Differences in candidate counts, losses and critics mean this is not a matched-compute method ranking.

[All measurements](summary.json) · [Independent audit](verification.json) · [Protocol](../../JOINT_PROTOCOL.md)

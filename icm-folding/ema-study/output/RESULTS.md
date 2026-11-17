# EMA and folding: measured results

Ten original gain-4 ICM sources, seeds 130–139; unchanged scalar MLP, native occupancy, 4,000 updates. Parameter averaging for evaluation and EMA targets in training are distinct interventions. Historical diagnostic cohort; no new actors.

| Encoder | Final folds | Material folds | Any sampled training fold | Mean log-distance R² | Min output std |
| --- | --- | --- | --- | --- | --- |
| Live original | 5/10 | 5/10 | 9/10 | 0.6119 | 0.6815 |
| Weight EMA 0.9 | 5/10 | 5/10 | 9/10 | 0.6119 | 0.6815 |
| Weight EMA 0.99 | 5/10 | 5/10 | 10/10 | 0.6119 | 0.6818 |
| Weight EMA 0.999 | 5/10 | 5/10 | 10/10 | 0.5979 | 0.6847 |
| Target EMA 0.99: student | 7/10 | 6/10 | 9/10 | 0.6017 | 0.6852 |
| Target EMA 0.99: target | 7/10 | 6/10 | 10/10 | 0.6016 | 0.6855 |
| Target EMA 0.999: student | 9/10 | 8/10 | 10/10 | 0.2653 | 0.6518 |
| Target EMA 0.999: target | 9/10 | 9/10 | 10/10 | 0.2578 | 0.6750 |
| Target EMA 0: student | 1/1 | 1/1 | 1/1 | 0.5876 | 2.3282 |
| Target EMA 0: target | 1/1 | 1/1 | 1/1 | 0.5876 | 2.3282 |

All final geometries were independently recomputed after reload on 20,001 float64 grid points. Original live reruns reproduce all ten prior checkpoints bit-for-bit. The full zero-lag target run (seed 131) also matches its original checkpoint exactly. Formula checks verify target detachment, EMA recurrence and the gradient-matched alignment control.

Training fold histories are sampled every 25 updates on 1,025 points. "Any sampled training fold" includes initialization. A final ordered EMA may have folded earlier; averaging need not repair a live fold or preserve monotonicity. Output variation and R² help identify stale/weak representations but do not establish downstream prediction utility.

This EMA-target experiment uses the original scalar alignment model without a predictor, keeps SIGReg on student endpoints, and uses the symmetric sum alignment to match the original gradient at zero lag. It is not a full BYOL/image-JEPA implementation. No learning-rate retuning, gain-8 stress test, fresh-seed validation, or direct continuation from existing folded models is included. No analytic no-fold guarantee is claimed.

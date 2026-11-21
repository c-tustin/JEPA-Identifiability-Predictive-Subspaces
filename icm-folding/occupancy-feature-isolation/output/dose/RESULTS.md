# Completed mass × conditional shape × objective sweep

All 120 fits passed independent dense float64 reload, component/source/initializer/plan checks, fixed raw normalization and derivative finite differences. Counts are descriptive across four reused conditional-profile templates and one encoder initializer.

| Near mass | SIGReg material / 4 | Raw/own-code InfoNCE material / 4 | Variance control material / 4 |
| --- | ---: | ---: | ---: |
| 25.000% | 0 | 0 | 2 |
| 50.000% | 0 | 0 | 0 |
| 62.500% | 4 | 0 | 0 |
| 68.750% | 4 | 0 | 0 |
| 75.000% | 4 | 0 | 0 |
| 81.250% | 4 | 0 | 0 |
| 87.500% | 3 | 0 | 0 |
| 90.625% | 1 | 0 | 0 |
| 93.750% | 0 | 0 | 0 |
| 96.875% | 0 | 0 | 0 |

Totals (any turn / material turn): sigreg: 20/40 / 20/40, raw_infonce: 0/40 / 0/40, variance_control: 28/40 / 2/40.

The mass intervention creates a substantial-fold window for SIGReg: all templates fold materially at 62.5%, 68.75%, 75% and 81.25%, none at 25%, 50%, 93.75% or 96.875%. At 87.5% and 90.625%, conditional shape affects the branch (3/4 and 1/4 material folds). Thus a universal threshold based only on near mass is false at the boundary. The historical substantial-fold masses (68.58–87.33%) and nonfolded masses (92.98–95.39%) lie on different sides of this experimentally manipulated window, consistent with the stage-1 mass swaps.

Raw/own-code InfoNCE has no detected final turn in any of the 40 cells. The variance control has numerous small turns, including two just above the material threshold at 25% near mass; it must not be described as universally fold-free. It uses the explicitly specified separate-input control, not the earlier joint-data VICReg replay. Candidate counts, loss weights, critics and compute differ across methods; this is an occupancy dose test within each method, not a method ranking.

This narrows the feature to a near-cluster/far-tail mass imbalance with a shape-dependent boundary, but does not establish a universal sufficient statistic. Objective-level risk comparisons, critic/collision checks, new initializations/plans, new sources, and temporal InfoNCE remain needed before completing the full goal.

[Figure](dose_response.png) · [Measurements](summary.json) · [Audit](verification.json) · [Protocol](../../DOSE_PROTOCOL.md)

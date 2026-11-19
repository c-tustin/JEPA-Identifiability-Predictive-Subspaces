# Completed dataset × initialization crossing

All 100 fits completed; all ten diagonal models reproduce historical parameters bit-for-bit. Independent final 20,001-point float64 reload checks pass for every cell. Detected final turns: 55/100; material turns: 51/100; turns inside source support: 50/100.

| Dataset | Final folds / 10 | Material folds / 10 | Endpoint mass below 2 |
| --- | ---: | ---: | ---: |
| 130 | 0 | 0 | 92.981% |
| 131 | 10 | 10 | 84.637% |
| 132 | 10 | 10 | 86.874% |
| 133 | 10 | 10 | 86.710% |
| 134 | 2 | 1 | 95.273% |
| 135 | 2 | 0 | 95.386% |
| 136 | 10 | 10 | 68.582% |
| 137 | 1 | 0 | 93.500% |
| 138 | 0 | 0 | 93.488% |
| 139 | 10 | 10 | 87.326% |

Datasets 131/132/133/136/139 fold for every tested initialization; 130/138 for none. Initialization changes outcomes for 134/135/137, including small turns. The dataset/sampling condition strongly controls final branch choice in this historical panel. It does not uniquely identify occupancy: rows also vary temporal pairs and minibatch plans. Initial seed 136 has a sampled turn, so final folds must not all be called newly created folds.

Reused sources, ten initializations, one plan per dataset, finite training budget and finite derivative grids limit this inference. There is no analytic injectivity claim. See [audit](verification.json), [audited measurements](audited-summary.json), and [matrix](crossing.png).

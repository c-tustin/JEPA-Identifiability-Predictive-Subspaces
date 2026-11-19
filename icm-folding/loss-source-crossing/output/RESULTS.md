# Completed loss-source intervention

All 100 source-crossing, 30 independent-endpoint and ten alignment-removal fits completed and passed independent checkpoint reload and identity checks. Derivative finite differences also passed for all 130 crossing/IID fits. Full crossing: 50/100 final turns, 50/100 material turns. The (131,131) cell reproduces historical encoder 131 exactly.

| SIGReg dataset | Final folds / 10 alignment datasets | IID endpoint folds / 3 plans | No alignment |
| --- | ---: | ---: | --- |
| 130 | 0 | 0 | No detected turn |
| 131 | 10 | 3 | Fold |
| 132 | 10 | 3 | Fold |
| 133 | 10 | 3 | Fold |
| 134 | 0 | 0 | No detected turn |
| 135 | 0 | 0 | No detected turn |
| 136 | 10 | 3 | Fold |
| 137 | 0 | 0 | No detected turn |
| 138 | 0 | 0 | No detected turn |
| 139 | 10 | 3 | Fold |

Changing the alignment source away from 131 changes final binary fold status in 0/100 cells. IID endpoint outcomes differ from the paired row-131 outcome in 0/30 fits. These are descriptive counts in a fixed historical panel, not independent-source statistical estimates.

The crossed design intervenes separately on the data supplied to each loss. It identifies the regularizer-data pathway if outcomes follow columns while alignment changes leave their status unchanged. The iid follow-up preserves each empirical endpoint distribution in expectation while removing paired-endpoint dependence; persistent folds there do not require paired endpoints. Alignment can still affect turn position, scale, timing and the training path. These experiments do not remove alignment from the objective.

The separate alignment-removal ablation does remove that term: 5/10 final folds, with 0/10 status differences from the paired row-131 baseline. Any fold in this ablation demonstrates that temporal alignment is not necessary for folding in that tested configuration. This is 0.1 SIGReg-only training with the same data, initialization and regularizer plan.

Endpoint distribution means the full sensor distribution with its multiplicities, not only mass below a threshold, coarse histogram, or number of modes. Initialization, input transform, objective and optimizer also participate in fold development. One common initialization in this crossing, three iid plans, reused sources, finite budget and derivative grids limit generalization; no all-input guarantee or unique actor-learning cause is proved.

[Full matrix](source_crossing.png) · [Audited measurements](audited-summary.json) · [Verification](verification.json) · [Main protocol](../PROTOCOL.md) · [IID protocol](../IID_PROTOCOL.md) · [Alignment-removal protocol](../ABLATION_PROTOCOL.md)

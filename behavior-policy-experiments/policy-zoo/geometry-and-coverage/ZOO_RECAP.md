# What we have done with the behavior-policy zoo

The world always has one changing scene coordinate: distance. The triangle shrinks as $1/d$. We have kept the encoder and SIGReg comparable while changing which observations and transitions train it.

| Study | What it established | What it did not establish |
| --- | --- | --- |
| [Original zoo](../README.md): 17 physical policies and Gaussian sources, 20 paired seeds | Count, random, persistent actions and curiosity policies produce very different recovery/fold outcomes. Pure ICM averages R² 0.603 with 10/20 folds; random 0.961 with none; ideal OU 0.995 with none. | A universal ranking of exploration algorithms, or a causal explanation separating occupancy from transition rules |
| [13 added controllers](../new-policy-results/README.md), bringing the physical catalog to 30 | Some policy→OU curricula beat fixed OU on fresh seeds; none establishes superiority over the best old curriculum. Gaussian-looking occupancy alone is insufficient. | That ICM→OU is reliably best, or that a small R² gain implies better planning |
| [Original stationary matched-coverage test](../matched-coverage/README.md) | Four stationary finite-state policies with identical marginals/action counts produce nearly identical recovery and shared-data reaching. | An explanation of original ICM: its correction changed its movement rule; only one frozen actor shaped that surrogate |
| [Original-policy follow-up](../original-policy-followup/README.md) | Re-evaluated 700 original encoders, including pure and handover fits. Removing alignment preserves fold status in 100 fits. Collecting earlier from the same frozen ICM actors repairs observed folds. | That the marginal distribution is the sole cause: paired endpoints still co-occur in finite SIGReg batches |
| [Fresh collection-age replication](../collection-age-replication/README.md) | Across 20 fresh seeds and five ages, late ICM windows often harm recovery/reaching. Switching the same early-trained encoder to late data creates folds in 15/20 runs, versus 0/20 with equal extra early-data training. | Early ICM superiority over random/count; monotonic damage at every age; a population identifiability theorem |

## What this campaign adds

1. **Separate the loss sources.** Cross early/late alignment pairs with early/late observations for SIGReg. Independent pooled-endpoint sampling isolates the regularizer's marginal exposure in this intervention. Compare with the old coupled controls to check whether pairing noise matters.
2. **Show how the curve moves.** Save slopes, raw curves and Adam components. Nearby training examples can change distant slopes through shared network parameters; we measure the actual function changes rather than infer geometry from gradient norms alone.
3. **Match exposure using original actors.** ICM, random, count, RND, disagreement and low/high-pass OU actions receive identical reset-state observations. SIGReg's observation multiset is also fixed exactly across conditions. Actors/action processes remain original. Successor distributions and stationarity are not claimed to match.
4. **Keep task consequences separate.** Compare affine recovery, order/aliases and actual physical goal-reaching. Shared predictor data better isolates representation effects; native action data additionally tests policy-specific exposure.

The remaining question is why the late sample changes geometry. Concentration, transition statistics, finite-batch correlations and finite-network optimization are distinct candidates. The new source swaps and exposure controls are designed to distinguish them, not to force ICM to win.

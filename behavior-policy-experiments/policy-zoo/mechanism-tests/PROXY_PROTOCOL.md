# Post hoc loss-landscape diagnostic

Added after inspecting the crossed matrices. This is a proposed explanation to test, not an independently confirmed prediction.

For each of the 20 already fixed datasets (four sources × five data seeds), compare a monotone empirical Gaussian-quantile code with codes obtained by Gaussianizing `(log distance − center)²`. Scan 81 centers evenly spaced in log distance from 1 to 16. Keep the output scale fixed at 1 for all candidates. Include the monotone candidate separately. The simulator labels are used only to construct these oracle diagnostic comparators; they do not train or modify any encoder.

Compute the **exact expected implemented loss when 512 pairs are sampled with replacement from that finite stored dataset**, including the covariance of the two endpoints in SIGReg. Verify that expression against 512 sampled minibatches on one fixed candidate before interpreting it. Endpoint marginals need not agree: use their pooled characteristic function.

Question: Does pairing alone prefer a folded ordering when the approximate output distribution is held Gaussian? Does its best center resemble the trained encoder's turn? Report failures of that proxy as well as successes. A minimum within this restricted rank-code family is not a network optimum or a prediction of Adam's final root. Rank ties and bounded finite support prevent exact continuous Gaussianization.

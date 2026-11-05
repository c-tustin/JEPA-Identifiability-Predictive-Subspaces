# Is 2.4 a stopping point, or a training cutoff?

Exploratory continuation control, specified before running these continuations.

The original learning rate is 0.001 through update 2,000, then cosine-decays to exactly zero at update 3,999 (the 4,000th update). Thus an almost stationary late curve does not establish a stationary loss or a preferred turning distance.

Primary example: seed 131, ordinary pure ICM, gain 4. Secondary diagnostic examples: the already selected seeds 130 (originally monotone) and 136 (initial fold). This is not a fresh population-level or significance study.

Start from the saved update-4,000 weights **and Adam state**, checked against the published final model. Retain the existing 16,384 ICM pairs, architecture, losses and independent 512-pair alignment/SIGReg batches. No new environment observations, policy training, world coordinates or planning claims.

Continue for 8,000 additional updates in three matched arms:

1. Ordinary sampling, constant learning rate 0.00001.
2. Ordinary sampling, constant learning rate 0.0001.
3. Constant learning rate 0.0001, change only SIGReg sampling to the earlier fixed log-size-balanced replay of the same stored pairs.

All arms share deterministic future alignment and regularizer index streams (seeds 2,100,000 + trial seed and 2,200,000 + trial seed). Arm 3 uses the same regularizer indices into the balanced replay pool. Adam state is carried, not reset. Save states every 500 updates, resumable with generator states. Retain the start state as a zero-learning-rate reference; explicitly verify one optimizer step at zero learning rate leaves all weights exactly unchanged.

At each checkpoint evaluate on the same 1–16 log-distance grid: raw curve and derivative, turning locations, and the pre-existing fold/R² diagnostics. Evaluate exact mean alignment over the stored pair pool, and the implemented SIGReg loss averaged over 16 fixed independent 512-pair batches, using the arm's regularizer sampling pool. The latter is a Monte Carlo diagnostic of its own training objective, not a shared cross-arm benchmark.

Questions: Does continuation move the fold substantially away from 2.4? Does its position depend on learning rate or regularizer sampling? Is there continued systematic movement over the last part of the continuation? Preserve missing/multiple roots and folds newly created in controls. No assumption that longer training repairs folding, or that any finite continuation proves convergence.

Publish checkpoint curves, learning-rate and root trajectories, diagnostics, source hashes and a concise explanation. Distinguish a finite training cutoff, a locally stable branch, and a globally preferred representation. Do not claim an exact universal final root without evidence.

# Tests of policy differences and initialization

Exploratory follow-up: the zoo's aggregate results have already been inspected. These are not new preregistered confirmations. This specification is fixed before computing the additional tests and crossed fits. Preserve the original zoo tables and p-values.

## Reanalyse the existing 20 matched seeds, 8–27

The independent replication unit is the seed, not a grid point, pair, training update or goal. Pair by seed across policies. Report mean paired differences with paired-seed percentile bootstrap 95% intervals (20,000 resamples; RNG 20261104), win/loss counts and two-sided exact binomial sign tests. Exclude differences within 1e-10 as numerical ties. A sign test concerns win probability among non-ties, not the mean. Holm-adjust within each family below.

1. Four ranking checks: count vs random; ICM vs random; RND vs ICM; disagreement vs ICM.
2. Persistence family: low-pass OU actions vs unfiltered OU actions; high-pass OU actions vs unfiltered; high-pass vs low-pass (pure); and the interaction `(high-pass − low-pass)_handover − (high-pass − low-pass)_pure`. This tests the observed reversal across training protocols, rather than comparing two separate significance decisions.
3. Baseline change: compute the truly untrained encoder's shared-grid R² for each seed, then compare each of 17 pure physical policies to that seed's baseline. This replaces no original result: the old first-update score remains a distinct measurement.
4. Equivalence family: eight filtered latent sources vs their matched independent Gaussian-pair controls, with paired t-based TOST at margins ±0.001 and ±0.005 R². For each margin, Holm-adjust the maximum of the two one-sided p-values across eight sources. State the paired-difference normality assumption and report 90% t intervals. Margins are post hoc sensitivity thresholds, not planning-derived tolerances; non-significance of a difference test is never called equivalence.
5. Associations: within each seed, rank the 17 physical policies by R², Gaussian-CDF discrepancy and size-bin entropy. Report each seed's Spearman associations, their seed-bootstrap mean intervals, and sign tests for negative Gaussian-discrepancy association and for its advantage over entropy (compare `−rho_CDF − rho_entropy`). Holm across these two tests. These are cross-policy associations, not causal mediation estimates or 340 independent replicates.

The original sign-flip test is exact under conditional sign symmetry/exchangeability, not under every possible zero-mean distribution. New sign tests are robustness checks with a different null. Neither the original nor the new tests establish planning performance.

## Cross data and initialization, rather than changing both at once

Four pure sources: ICM, random, count coverage, fixed OU. Cross the first five existing data seeds 8–12 with the first five existing encoder seeds 8–12: 100 fits. No selection of seeds by failure. Each dataset and its minibatch stream stay fixed while initialization changes. Keep the original 4,000-update schedule, architecture, coupled alignment/SIGReg batch, environment and loss. No new coordinates or policy training.

Before any crossed fit, regenerate/cache each source and require its observation-pair SHA256 to match the zoo record. Refit the 20 diagonal combinations and require final weights and shared-grid R² to match the original zoo checkpoints; abort on mismatch. Save the genuinely untrained and final curves for every combination.

For the crossed decomposition, use one common evaluation/calibration seed (20261104) for every combination, so calibration sampling does not become part of the data factor. Also record the original data-seed-specific evaluation for the diagonal reproduction gate. The existing-20-seed reanalysis retains the original metric and its matching baseline.

Report all 5×5 R²/fold matrices. Decompose centered squared variation into dataset main effect, initialization main effect and interaction using two-way means. Interaction includes any unmodelled batch-stream interaction; the dataset factor includes collection and minibatch randomness, not only occupancy. This decomposition is descriptive on five selected-by-index levels per factor, not 25 independent population trials. Report any negative random-effects variance estimates rather than silently converting them into positive evidence; the main report uses nonnegative finite-table sum-of-squares shares instead.

Questions: Are the same bad datasets bad under different initializations? Does one initialization fail across data? How large is their interaction? Do ICM folds require an initially folded encoder? No claim of a universal final root, convergence basin, or algorithm ranking follows from these finite fits.

Publish concise interpretation, exact numerical outputs, plots, source hashes and verification. Distinguish a population loss preference from which solution finite optimization reaches.

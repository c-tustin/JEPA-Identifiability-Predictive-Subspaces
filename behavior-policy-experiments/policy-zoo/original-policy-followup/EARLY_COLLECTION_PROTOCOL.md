# Test the concentration hypothesis with unchanged behavior policies

Specified after the alignment-removal results (100/100 fold statuses unchanged) and before these fits.

Hypothesis: the ICM actor steers trajectories into a narrow region during discarded burn-in. Fitting the encoder to the later concentrated sample can preserve or develop folds elsewhere under our finite architecture/optimizer. Collecting earlier with the same actor should reduce ICM folding and improve recovery. This is an empirical optimization/sampling hypothesis, not a population theorem or a claim that ICM curiosity itself prefers folded encodings.

Policies: original frozen ICM, random, count; original seeds 8–27. Original phase collects steps 128–191 after 256 uniform physical-distance resets. Early phase collects steps 0–63 after the same resets. Same action rule, actor weights, 16,384 physical transitions, observation renderer and encoder training protocol. Reset transitions are never positive pairs. Regenerate original phase and require its stored SHA-256 before each early-source fit; never retrain an actor. Changes to the collection phase intentionally change occupancy and the empirical joint law. The earlier phase is transient; this test does not hold stationarity or marginals fixed.

Fit 60 encoders with the unchanged full objective for 4,000 updates, same historical initialization/minibatch indices and learning-rate schedule. Compare against the saved original pure encoders using common-grid affine recovery, rank correlation, folds and refined alias witnesses. Track pooled endpoint near-distance mass, occupied log bins and normalized log-bin entropy for early and original data.

Primary inference: three source-wise recovery changes and three fold-status changes, one Holm family of six paired tests. Exact two-sided sign tests; discordant-pair binomial test for folds. All 20 seeds retained; bootstrap mean-difference intervals are unadjusted.

For seeds 8–17, evaluate shared and early-native predictor data for all three early encoders, using the same 16,384-pair shared source, 2,000 predictor updates, normalization and 64 reachable goals as the original-policy study. Compare shared-data goal errors against the original same-policy encoders: separate Holm family of three paired sign tests. Native comparisons jointly change encoder and action data and are secondary descriptive results. Every goal has zero oracle error; score physical distance.

Prediction: ICM should improve more than the broad random baseline. Count may change differently because its original stationary-like occupancy was already helpful. If ICM's sampling spreads but its folds persist, concentration alone is insufficient. If recovery improves without planning improvement, the task-level hypothesis remains unconfirmed. A fold repaired by earlier collection does not isolate occupancy from the other empirical pair-law changes; the alignment-removal result is complementary evidence.

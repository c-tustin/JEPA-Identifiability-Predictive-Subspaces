# Fresh replication: when does ICM help or hurt?

Frozen before fresh actor training or encoder fits. This is a prospective replication of an exploratory observation, not a theorem proof or an attempt to select a winning subset.

## Meaning of early and late

Collection age is the number of environment actions discarded **after a reset**, with a fully trained and then frozen behavior actor. It is not the actor's training epoch. Nor do separate-window encoder fits demonstrate that one encoder folds as it trains. The continuation test below addresses that latter question directly.

## Fresh cohort and unchanged world

Twenty new joint seeds, 200–219, unused in the prior zoo/follow-ups. Train original ICM and count actors for the existing 600 A2C updates on each seed; random has no trained actor. Same one-coordinate triangle world, inverse-distance renderer, physical distances 1–16, moves −0.25/0/+0.25 and reflection. No added coordinates, no policy redesign, no altered SIGReg. Save frozen actor checkpoints atomically; an interrupted actor training restarts that actor. Check actor-training replay gates on old seed 13 before interpreting new fits.

For each policy/seed, generate one 192-action rollout of 256 parallel trajectories from common uniform physical-distance resets. Extract 64-step windows starting at ages **0, 16, 32, 64, 128**. Windows share underlying trajectories, may overlap, and are paired repeated observations, not independent samples. Each dataset has 16,384 physical transitions. Reset transitions are never positive pairs. Keep every actor frozen across its windows. Save source hashes and actor hashes.

Also train an ideal Gaussian OU pair reference per fresh seed, under the original pair law. It is not a physical action policy and has no native action labels.

## Encoder and recovery

Train all 3 × 5 × 20 = 300 physical encoders plus 20 OU references. Same historical seed-dependent initialization/minibatch plan, 4,000 Adam updates, batch 512, unchanged 0.9 alignment + 0.1 SIGReg and learning-rate schedule. Compare with the untouched initial encoder on the same grid. No selection by initial fold, final fold, recovery or policy-training outcome.

Use common uniform-log-distance evaluation: affine R² (primary recovery metric), absolute rank correlation, dense-grid reversal diagnostic, and refined equal-code alias witnesses. Low R² is not necessarily information loss; no detected grid fold is not an injectivity proof. Record encoder state/optimizer every 1,000 updates and final optimizer for continuations.

## Does later data create a fold in the same encoder?

For every early ICM encoder (all 20, not a successful subset), start two 4,000-update continuations from exactly its saved encoder **and Adam state**:

- early→early: keep using its age-0 data;
- early→late: use its age-128 data.

Restart the learning-rate clock equally for both arms; carry Adam in both. Pair continuation minibatch indices, seed 3,100,000 + seed. No other optimizer or architecture changes. Record fold/recovery histories every 500 updates. These 40 additional fits test the causal effect of the supplied continuation dataset on a common initial representation and optimizer. Even here the data intervention changes both occupancy and pair statistics.

Total: **360 fitted encoder conditions**. The extra-training control has the same budget as the late continuation. Separate-window comparisons alone are not called within-encoder fold creation.

## Planning for every condition

Freeze every encoder, including both continuation arms. Fit the existing action-conditioned predictor for 2,000 updates, batch 256, same initialization/minibatch streams within seed. Shared data: identical 16,384 feasible physical transitions and calibration observations per seed. Native data: the encoder's physical training window; for continuation arms use the corresponding continuation window. OU gets shared-data evaluation only. Total: 360 shared and 340 native evaluations, **700 predictor fits**.

Use the existing 64 full-range reachable goals per seed, 12 replanning actions, horizon 8, and 128 candidates including every reachable net displacement. Score actual terminal physical-distance error and success within 0.25, plus errors by starting region. Verify physical oracle error zero. The planner receives only current/goal codes and action predictions, not distance labels.

Shared-data comparisons isolate differences through the representation for this predictor/planner. Native data additionally changes state/action exposure. Goals are reused within seed and are not independent replicates.

## One correction family for all 19 planned tests

Paired, exact two-sided sign tests; fold-status comparisons use discordant pairs (exact binomial/McNemar). All 20 seeds included. Positive gain has the direction specified below. One **Holm family of 19**, encompassing recovery, folds, planning and trends, avoids changing correction scope after seeing outcomes. Report raw p, corrected p, all seed outcomes and unadjusted 95% paired-mean bootstrap intervals (20,000 resamples, seed 20261105).

1–3. ICM age 0 versus age 128: higher R², fewer folds, lower shared-data goal error.
4–5. ICM age 0 versus random age 0: higher R², lower shared goal error.
6–7. ICM age 0 versus count age 0: same two endpoints.
8–9. ICM age 0 versus ideal OU: same two endpoints.
10. ICM age 0 versus untouched initialization: higher R².
11–12. ICM ordered age trend: per-seed negative Spearman correlation of collection age with R², and positive Spearman correlation with shared goal error. Sign-test these oriented correlations; ties/constant outcomes are zero. This tests direction consistency, not a linear dose model.
13–14. ICM age-0-versus-128 benefit minus the corresponding random benefit: recovery and shared planning.
15–16. The same interaction versus count.
17–19. Early→early versus early→late continuations: higher R², fewer folds, lower shared goal error. Improvement in this orientation means late data is harmful relative to equal additional early-data training.

Sign tests concern consistent seed-wise direction, not a zero-mean null. A mean gain, lower p-value or fold-free subset alone is not called superiority. Native planning and exposure statistics are descriptive in this protocol; no extra inferential claims are added after inspection.

## Interpretations allowed

- Early ICM beating late ICM supports a collection-phase rescue, not superiority to random/count/OU.
- Superiority to an alternative requires the corresponding prespecified comparison; a nonsignificant difference is not equivalence.
- Worse late-window fits plus an age trend support a collection-age association. A fold appearing during early→late continuation, relative to early→early, supports an effect of late data on that initially shared encoder.
- Occupancy concentration, reflection frequency and action coverage are measured but not separately randomized. They remain candidate mechanisms, not isolated causes.
- Fresh-seed replication generalizes beyond the old inspected cohort for this implementation. It does not certify a population identification theorem or all possible ICM implementations.

# Does a broad SIGReg observation pool help regular OU?

Fix this plan and executable hash before any cohort OU encoder fit. Seeds 6000–6059 reuse the preceding ICM study's initializations, broad pools, planner data and goals, but these OU arms have not previously been fitted. This is a new comparison on an existing paired cohort, not a claim that all seed-related quantities are unseen.

## Three conditions, identical OU alignment pairs

Use the original stationary independent Gaussian OU pairs: z~N(0,1), z'=0.9z+sqrt(1-0.9²)epsilon; d=4exp(0.7z); observations are the same V/d triangle. Do not clip Gaussian pairs to 1–16. No added coordinates or new action policy.

1. **Original OU:** SIGReg sees both endpoints of the same alignment minibatch, using the original update routine.
2. **OU observations:** SIGReg independently samples 1,024 observations from the pooled 32,768 OU endpoints.
3. **Broad observations:** SIGReg independently samples 1,024 observations from the identical 16,384-view broad pool used in the preceding ICM controls: log-uniform distance 1–16.

Conditions 2 and 3 use the preceding ICM controls' split-gradient implementation; condition 1 uses the historical combined-loss update. Both compute the same mathematical loss. The condition-2 control therefore checks the sampler/implementation change before attributing a condition-1-versus-3 difference solely to exposure. Between conditions 2 and 3, only the observation pool changes.

All conditions share 4,000 updates, encoder initialization, OU pairs, alignment indices, Adam/LR schedule, batch size 512, alignment coefficient 0.9 and unchanged finite-frequency SIGReg coefficient 0.1. Observation RNG seeds match across conditions 2 and 3. Gaussian OU endpoint observations can lie outside 1–16, whereas the broad pool is bounded; report this distribution change explicitly. The exact population theorem is not asserted for the hybrid loss with a different regularization marginal.

## Recovery and planning

Primary recovery diagnostics:

- Existing affine log-distance R² on the 1,025-point uniform log-distance grid covering 1–16. Also retain near/far scores and grid fold flags as descriptors.
- Affine R² on **16,384 held-out Gaussian latent states**, seed 9400000+seed, common across conditions.
- On those same Gaussian states, raw orthogonal recovery error min_s E[(h(x)-s z)²], s in {-1,+1}. No fitted scale or offset. This distinguishes high affine R² from recovery up to the theorem's actual one-dimensional rotation. Output mean/std and tail R² (|z|>2) are descriptive.

Fit the existing shared-data action-conditioned predictor for 2,000 updates and run the same 64 reachable physical goals/planner. This is a secondary application diagnostic in the bounded physical task. Gaussian OU pairs have no physical action labels, so no native OU action-conditioned predictor is fitted.

Total **180 encoders and 180 shared predictor/planning evaluations**, plus old-seed verification gates. No success filtering, changed sample size, outcome-dependent stopping, or parameter tuning.

## Fixed statistical comparisons

All 60 paired seeds retained. Use exact two-sided paired sign tests for directional consistency, with tolerance 1e-10; binary fold differences use discordant seeds. These test direction/win probability, not a population-mean null. Also show paired mean effects and descriptive unadjusted 95% percentile bootstrap intervals (20,000 seed resamples, RNG 20261107).

One Holm family contains **16 tests**:

- Broad versus independently sampled OU observations: bounded-grid R², Gaussian R², raw orthogonal MSE, shared-data goal error, folds (5).
- Broad versus original coupled OU: the same five outcomes (5).
- The preceding ICM-native-alignment/broad-SIGReg controls versus original OU, and versus broad-SIGReg OU: bounded-grid R², shared goal error, folds (6). These reuse the 60 previously observed ICM fits; they are new, declared comparisons, not a fresh replication of ICM. Both independent-SIGReg conditions use the same gradient implementation. The original-OU comparison additionally differs in sampler/implementation and in transition law/step scale.

Effects are oriented positively for better recovery or lower MSE/error/fold rate. Two-sided tests may reveal harm as well as benefit. Independent-OU versus original-OU is a descriptive sampler diagnostic, not an additional significance family. No equivalence claim is inferred from nonsignificance.

Do not substitute bounded-grid recovery for Gaussian-distribution recovery if they disagree. Do not call improved R² alone an improvement in orthogonal identifiability. Differences from physical ICM include the pair law and movement scale; they are recipe/source comparisons, not matching stationary joint laws.

## Verification and restart

Require an exact historical original-OU seed-13 encoder replay before freezing. Independently recreate all 60 sources, held-out Gaussian states, all 180 geometries and actual planning rollouts; independently verify statistical boundary/count calculations and one resumed independent-SIGReg fit. Save model/Adam every 500 updates and completed predictors atomically; interrupted predictor fits restart deterministically. Hash code and plan before training; keep regenerable data caches out of Git. Report an incomplete cohort as incomplete instead of replacing failures.

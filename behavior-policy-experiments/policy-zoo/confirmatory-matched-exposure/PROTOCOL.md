# Fresh-seed confirmation: policy differences and observation exposure

Written and committed before any cohort actor, encoder or predictor is trained. This is a prospective confirmation motivated by exploratory results, not an externally registered study. No claim of tolerances derived uniquely from physics.

## Fixed question and tolerances

Do seven original behavior policies give practically equivalent empirical recovery and shared-data reaching after matching starting observations and SIGReg exposure? The seven policies are ICM, random, count, RND, disagreement, low-pass OU actions (w=0.1), and high-pass OU actions (w=0.3).

- Recovery: absolute difference in population mean R² below **0.001**, separately for the whole uniform log-distance grid, d<4, and d>=9. Since 1-R² is the normalized best-affine reconstruction MSE, this allows a difference of 0.1% of target variance. Equal R² does not imply equal learned functions or exact identifiability.
- Shared-data planning: absolute difference in mean actual terminal physical-distance error below **0.005**, 2% of a 0.25 action step. This is a stringent task-resolution tolerance, retained from the exploratory analysis rather than chosen to fit new outcomes. No task-specific external utility function was supplied.
- All four endpoints AND all 21 policy pairs must pass; this is one conjunction/intersection-union claim. Thresholds and sample size never change in response to fresh results.
- Policy-specific predictor-data equivalence at 0.005 and paired fold-risk equivalence at 5 percentage points are secondary diagnostics, not confirmatory claims. Sixty seeds give only limited power for rare fold events. No success-conditioned subsets.

## Independent replication and unchanged training

Use **60 fixed seed blocks, 6000–6059**, all excluded from the earlier studies. Freshly train all four learned actors per seed (ICM/count/RND/disagreement) for the original 600 updates; never reuse the old actors. Random and colored OU sources get fresh collection randomness. Pair encoder initialization, alignment minibatch indices, independent SIGReg input indices, predictor initialization, and goals across policies within each block.

Keep one physical coordinate, renderer V/d, distance range 1–16, reflected moves {-0.25,0,+0.25}, architecture, finite-frequency SIGReg formula/weight 0.1, alignment weight 0.9, batch 512/1024, 4,000 Adam updates and original learning-rate schedule. Use exactly the preceding common-reset intervention: 16,384 stratified log-distance starting observations per seed, identical across policies; valid physical action pairs; SIGReg gets identical independent common-pool batches. Successors, action frequencies and stationarity are NOT matched. These are physical OU action controllers, not the theorem's Gaussian-latent OU source.

Frozen encoders get 2,000-update action-conditioned predictors on shared data and on their cell's policy-specific alignment data. Evaluate actual terminal error on the inherited 64 reachable goals, 12 actions, horizon 8, 128 candidates; verify the physical oracle. Shared predictor data retain the inherited endpoint-action omission; this limits the evaluated task distribution.

## Mechanism control that can go beyond a policy ranking

For ICM only, also train two native-alignment cells using its same new actor/data/initialization/optimizer streams:

1. **native exposure:** SIGReg independently samples the pooled native endpoints.
2. **common exposure:** SIGReg independently samples the common broad observation pool.

Alignment pairs are identical in these two cells. Both use independent observation sampling, so the exposure intervention does not also switch coupled versus independent batching. Common exposure is additional broadly distributed real observation data, not cost-free importance reweighting. This intervention changes the regularizer's data distribution; it does not change its formula or weight.

This yields **540 encoders and 1,080 predictor/planning evaluations**. Actor learning and source preparation are included; no extra training coordinate or dynamics is added. The mechanism controls test finite-training geometry and actual reaching outside the ideal population theorem.

## Inference fixed before the fresh outcomes

Primary global mean equivalence: paired t-based TOST at alpha .05 for every policy pair and each of the four endpoints; global p is the maximum component p. The seed is the sampling unit, never a distance point or goal. Ordinary 90% paired intervals are pointwise, not simultaneous. TOST assumes approximately normal seed differences; use paired sign-TOST as a prespecified robustness check of typical differences, which is a different estimand.

Three directional mechanism contrasts, **common minus native exposure with native alignment held fixed**, oriented positively for benefit: recovery gain, decrease in fold incidence, decrease in shared-data goal error. Continuous contrasts use exact one-sided paired sign tests (ties excluded); fold incidence uses exact one-sided McNemar/binomial on discordant seeds. These test directional win probability, not population mean improvement.

One **Holm family of four** covers the primary global equivalence claim plus these three mechanism contrasts. Report all four whether favorable or not. Also report paired mean effects and unadjusted percentile bootstrap 95% intervals (20,000 seed-block resamples, RNG 20261107); these are descriptive intervals, not substitutes for the sign tests.

Secondary diagnostics: policy-specific predictor-data paired TOST at the fixed 0.005 margin; fold-risk conservative exact paired equivalence via discordance probability at margin 0.05; individual policy fold counts with exact one-sided 95% binomial upper bounds. These secondary p-values are labeled unadjusted and support no additional confirmatory claims. At zero discordances, (1-0.05)^60=0.0461: rare-failure conclusions remain fragile. No pooling with old seeds to rescue a test.

No partial-cohort significance analysis, optional stopping, dropped failed seeds, policy selection, margin/budget tuning, or replacement seeds. Hardware/software failures resume the same seed; an irreparable failure is reported as missing, preventing a complete-cohort conclusion. Record signatures and checksums. Encoder/Adam resumes every 500 updates; completed actor/predictor states survive restarts. An interrupted actor or predictor restarts that bounded fit deterministically.

## Gates and interpretation

Before fresh training, replay old seed-13 ICM and RND actors and the original matched ICM cell, requiring exact final weights and planning outputs. Old seeds used only for verification are not added to the cohort. Check sources/actions, shared input hashes, deterministic goal/geometry replay, statistical boundary calculations, and representative full/resumed fits.

Significance answers a prespecified statistical question conditional on the stated assumptions. Usefulness requires an effect on recovery and physical reaching, not just a small p-value. Equivalence after a broad-data intervention does not show behavior policy never matters. A confirmed exposure-only rescue would identify a finite-network training mechanism the population/global-optimum theorem does not characterize. It would not prove identifiability for nonlinear dynamics, provide a new asymptotic guarantee, or establish new-to-the-literature priority.

Paper boundary: [Klindt et al., §§7–8](https://arxiv.org/html/2605.26379v1#S7) states finite-sample/optimization limitations and the need to learn action-conditioned transitions. Our claims concern this concrete one-distance experiment, not a contradiction of that theorem.

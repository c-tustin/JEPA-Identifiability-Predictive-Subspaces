# Why late data bends the representation: mechanism and coverage controls

Written before the new intervention fits. This is an exploratory mechanism campaign using inspected, fixed actors/datasets, not a new blind algorithm ranking. Same one-coordinate world, renderer, actions and SIGReg formula/weight. Original policy weights are never changed.

## A. Separate the two loss sources (80 continuations)

Use all 20 early ICM encoders and their Adam states from the collection-age replication, seeds 200–219. Cross alignment pairs from age 0 or 128 with SIGReg observations from age 0 or 128: four cells per seed. Start each from exactly the same early-trained weights/Adam state, restart the same 4,000-update learning-rate clock, batch 512 alignment pairs and 1,024 regularizer observations.

SIGReg observations are drawn **independently from the pooled endpoints** of its selected source, rather than as correlated pairs. Thus its input law in this intervention depends only on the empirical endpoint marginal. Both views remain eligible; no endpoint is dropped. The SIGReg formula and coefficient 0.1 are unchanged. Alignment coefficient remains 0.9. The four cells share alignment index draws and pooled-observation index draws within seed. This sampling design is intentionally different from the original coupled-batch loss.

Compare independent-source early/early and late/late cells against the saved original coupled continuations to detect a role for paired finite-batch noise. The factorial does not automatically explain the coupled original if its late/late result differs.

Record raw representation curves and derivatives every 100 updates. For the preselected first five seeds, 200–204, save loss gradients, model/Adam state and one-step component effects before updates 0, 100, 500, 1,000 and 2,000. Decompose the actual Adam update into carried momentum, current alignment gradient and current SIGReg gradient using the same combined second-moment denominator. Check numerical reconstruction and local curve/slope predictions. This is a local optimizer attribution, not separate Adam training or a global convergence proof.

## B. Match exposure while retaining original zoo policies (280 fits)

Seven original policies: ICM, random, count, RND, disagreement, low-pass OU actions and high-pass OU actions. Use every original seed 8–27. Reuse frozen actors and require original native pair pools to match their saved zoo SHA-256. For each seed generate a common 16,384-observation pool, stratified uniformly in log-distance over 1–16 and shuffled. This is a controlled reset distribution, not a learned policy or stationary law.

For each policy, collect valid one-step transitions from exactly the same ordered reset-state pool. Learned actors read those observations unchanged. Colored-action policies do not read the scene, so their exact original native action stream is reused, retaining its original filter warm-up and temporal correlations; only physical distance is reset between steps. Each positive pair is an actual physical action, never a reset jump. The policy's action probabilities are not corrected or symmetrized.

Train two cells per policy/seed from the same historical initialization: alignment on its original **native trajectory pairs**, or on these **common-reset pairs**. In all 14 cells within a seed, SIGReg receives exactly the same 1,024 independently sampled common-pool observations at every update. Thus both its population exposure and its finite-batch input multiset are held fixed across policy/source cells. Every fit uses 4,000 updates, original optimizer and 0.9/0.1 loss coefficients.

This matches **starting-state coverage and SIGReg exposure** exactly. It deliberately does not claim identical successor marginals, action frequencies, rollout occupancy, temporal state/history relationships, or stationarity. Supplying the broad shared observation pool adds real observations beyond any native coverage gaps; it is not merely replaying weights on existing late ICM data. Native-vs-reset alignment comparisons hold the regularizer stream fixed. Old raw zoo fits are contextual references, not substituted controls for the new sampling design.

## Evaluation

All 360 new encoders receive common-grid affine log-distance R², rank correlation, dense-grid fold diagnostics and refined equal-code witnesses. Do not select successful seeds. Good R² or no detected grid fold is not a population identification guarantee.

Fit the unchanged action-conditioned planning predictor for 2,000 updates on the same shared 16,384 physical transitions for every encoder within seed, with identical initialization, calibration and minibatch indices. Score 64 identical full-range reachable goals per seed, 12 replanning actions, horizon 8 and 128 candidate sequences. Verify zero oracle error; primary score is actual terminal physical-distance error.

Experiment A has only shared-data planning: a hybrid loss-source cell has no unique native behavior policy. Experiment B additionally trains a native predictor on that cell's actual alignment-source action data. Native planning is descriptive, because both predictor data and representation may differ. Total: 360 shared plus 280 native predictor fits, **640 planning evaluations**.

## All 54 tests share one Holm correction

Exact two-sided paired sign tests, zero tolerance 1e-10. Simple binary fold comparisons use discordant pairs. Fold interaction contrasts are general seed-level signed effects, not McNemar counts. Unit of inference is the seed, never a distance point, update, pair or goal. Bootstrap mean-effect intervals: unadjusted 95%, 20,000 resamples, fixed seed 20261106. All 20 seeds are retained for each specified contrast. No outcome-based exclusions or post hoc successful-run tests.

A (21 tests):
- Change regularizer early→late separately at each alignment source, for recovery, fold status and shared goal error: 6 tests.
- Change alignment early→late separately at each regularizer source, same three endpoints: 6 tests.
- Factorial interaction for each of the three endpoints: 3 tests.
- Independent early/early versus original coupled early continuation, and independent late/late versus original coupled late continuation, same three endpoints: 6 tests.

B (33 tests):
- Each of the six nonrandom policies versus random in the common-reset alignment condition, for recovery and shared goal error: 12 tests.
- Common-reset versus native alignment within each of seven policies, for recovery, fold status and shared goal error: 21 tests.

Positive simple-source effects mean a benefit from the earlier source or from common-reset alignment, respectively. Positive policy effects mean a benefit over random. Interaction orientation is specified algebraically in analysis code and reported without claiming that a positive interaction alone is beneficial. A sign-test result concerns seed-wise direction, not a zero-mean null. Native scores and gradient summaries add no unplanned significance tests.

## Required interpretation checks

- Distinguish source-marginal effects in the independently sampled factorial from pair-covariance effects in the original coupled objective.
- Distinguish a source-dependent pressure from parameter sharing: a gradient driven by nearby observations can alter slopes at far distances through the same weights. Show actual function/slope changes, not just gradient norms.
- Match reset observations and regularizer input multisets exactly; verify actual actions and unchanged actor parameters. Do not describe this as matching stationary joint distributions.
- Retain failures and coverage gaps in figures and tables. No universal causal conclusion from one illustrative seed.
- Save partial model/optimizer states and code/protocol hashes; reproduce saved geometry, physical rollouts, statistics and representative full/resumed training before publishing conclusions.

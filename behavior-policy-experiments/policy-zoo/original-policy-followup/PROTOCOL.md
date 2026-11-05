# Original-policy follow-up: hypothesis and fixed tests

This exploratory follow-up was specified after the matched-coverage comparison, before the intervention fits. It uses original saved zoo runs and hash-verified original datasets, rather than replacing ICM with a stationary surrogate.

## Hypothesis: observations can be enough to produce the fold

The large original ICM effect may be driven primarily by its highly uneven sampled observations interacting with our fixed encoder/training procedure. Its specific transition connections may be unnecessary for a fold to appear. This predicts that removing alignment, while keeping every sampled endpoint and SIGReg unchanged, retains many ICM folds. Random and count are negative controls; RND is another collapsed-coverage positive control; low-pass OU actions test a weaker, persistent-action failure.

This hypothesis is motivated by existing endpoint-sampling repairs and the matched-coverage null, not proven by them. The earlier reflection-removal test did not repair folds. The new intervention can establish whether alignment is necessary in these fits, not a complete causal explanation of the observation distribution or SIGReg geometry.

## Comparisons for identifiability

Re-evaluate all 17 original physical policies under pure and OU-handover training on the original 20 seeds (8–27), plus the pure ideal OU reference: 700 saved encoders. Do not retrain or select successful runs. Report historical shared-grid affine log-distance R², new absolute rank correlation, dense-grid fold rate, local recovery, and equal-encoding alias witnesses.

A nonlinear monotone map can have poor affine R² while remaining invertible. A continuous scalar map with branches taking the same value loses distance information. Search 257 output levels for two branch crossings, then refine both roots on the actual network in double precision. This is a numerical witness, not an exhaustive maximum-gap proof. Its two-goal lower bound is half their physical separation for a controller receiving exactly equal goal encodings; floating-point near-equality is not a certified deployed-controller impossibility.

## Alignment-removal intervention

Policies: ICM, random, count, low-pass OU actions (`ou_actions_lp_w0.1`), RND. Seeds 8–27. Original frozen behavior actors, original 16,384-pair pools, encoder initialization and historical batch-index plans. Every regenerated source must match its old SHA-256; never retrain a behavior actor.

Fit 100 encoders for 4,000 updates. Use exactly the old SIGReg, its 0.1 coefficient, 512 pairs, Adam, and learning-rate schedule. Set the alignment coefficient from 0.9 to zero. Every update retains exactly the same concatenated endpoints; no resampling, occupancy correction, reflection removal, or regularizer ablation. Original saved pure fits are paired controls. Save encoder/optimizer progress every 1,000 updates. Verify at least one full original-loss training replay against saved weights before interpreting interventions.

Primary inference: five policy-wise paired recovery changes plus five changes in fold status, Holm-corrected as one family of ten. Fold tests use only discordant pairs (exact two-sided binomial/McNemar test). R² tests use exact two-sided signs; intervals describe paired means and are unadjusted. Use every seed, never the subset where a policy succeeds. Conditional seed dependence across policies is handled by the conservative correction, not assumed away.

A fold persisting under removal shows alignment was unnecessary in that intervention; its old trained root need not remain at the same distance. A removed fold supports an interaction with alignment. A new fold after removal shows alignment was protective for that run. No result implies SIGReg alone is a universal cause or that stationary coverage correction preserves original policies.

## Original-policy planning

Evaluate six original pure encoders: the five policies above plus ideal OU; seeds 8–17, fixed before new planning fits. This is a ten-seed exploratory subset, not selected by folds or success. Keep the original encoders unchanged.

Train the existing action-conditioned predictor for 2,000 updates with identical initialization and batch-index streams per seed. Shared-data arm: the same 16,384 feasible-action transitions on the distance grid, log-uniform source occupancy. Native arm: original policy's 16,384 physical transitions. Predictor calibration uses the shared observations. The ideal OU pair source has no physical action labels, so it gets shared-data planning only.

Score the same 64 full-range reachable goals per seed, 12 replanning actions, horizon 8 and 128 candidate sequences; include every reachable net displacement and verify zero oracle error. Goal error is physical distance, not latent MSE. Native-vs-shared is a joint data-distribution/action-support effect; the native sets do not have matched occupancy. Original-policy ranking is not a matched-marginal causal test.

Primary planning comparison: five policies versus OU, shared-data error; Holm family of five paired sign tests. Secondary: native versus shared error within the five physical policies, corrected separately. Goal cases are not independent seeds. Report all results, folds and differences regardless of significance.

## What the matched-coverage result does and does not say

The prior four-policy comparison fixed one broad marginal and stationary local kernels. It found almost identical trained recovery and shared-data planning despite distinct slow features. This rejects neither all transition effects nor a role for original ICM. For one-dimensional stationary nearest-neighbor movement, fixing the marginal forces the forward/backward edge ratio; the prior ICM correction therefore removed important aspects of its original behavior. The present original-policy comparison intentionally retains the real occupancy and pair law instead.

# Merged progress: ICM policy, folding, and repairs

Updated November 15, 2026 from the stored studies linked in [the index](README.md). These are previously inspected diagnostic sources, primarily seeds 130–139, rather than a new independent validation cohort. Fold counts from different protocols should be read with their stated grid and support definitions.

## November 15: Original-MLP gradients and regularizers (completed)

Added [study and reproduction](mlp-gradient-study/README.md), [protocol](mlp-gradient-study/PROTOCOL.md), and [training code](mlp-gradient-study/run.py). Completed 145 encoder fits: 80 main paired fits, 30 adaptive global-mining fits, ten matched-strength local controls, and 25 fits on five freshly trained ICM policy sources. Every encoder is the original scalar MLP trained from random initialization. Per-update inputs contain observations/actions only; no ordered spline, physical ordering labels, or analytic dynamics enter the gradient.

Baseline final fold counts on ten development sources: SIGReg 5/10, scalar VICReg 0/10, distance-score InfoNCE 0/10, Gaussian quantile matching 9/10. On five fresh policy sources, respectively: 2/5, 1/5 tiny (0 substantial), 0/5, 5/5. Scalar VICReg has no off-diagonal covariance term; scalar InfoNCE uses an RBF distance score rather than degenerate scalar cosine. These are toy adaptations, not general method rankings.

The local successor-conflict gradient at coefficient 1 has little effect. Global mining at coefficient 100 reduces SIGReg folds to 1/10 on development sources and improves all ten held-out observation-prediction errors. The coefficient was fixed before generating fresh sources; there it reduces folds 2/5 -> 0/5 and improves all five prediction errors (mean MSE 0.135788 -> 0.000334). A matched-strength local coefficient-100 control leaves 4/10 sign reversals, including 3/10 substantial folds. Mining has additional compute and observation exposures, so this is not a matched-compute comparison. One global-mining development fit remains substantially folded, disproving any universal guarantee for this gradient rule.

All 145 final models were independently reloaded, checked on 20,001-point derivative grids, and their stored downstream decoder errors recomputed on whole held-out trajectories and a common coverage-stress set. Four objective finite-difference gradient checks, the collision-gradient direction, and global-sign invariance passed. Development coefficients were adaptive; the fresh cohort had fixed hyperparameters. Five fresh sources remain a small same-renderer validation. [Gallery](mlp-gradient-study/output/index.html), [main results](mlp-gradient-study/output/RESULTS.md), [mining](mlp-gradient-study/output/MINING.md), [fresh results](mlp-gradient-study/output/FRESH.md), [strength control](mlp-gradient-study/output/STRENGTH.md).

## November 15: Weighted-moment validation, direct-repair replication and GitHub snapshot

Added a [fixed-protocol validation](weighted-moments-validation/PROTOCOL.md) using five recent ICM sources (140–144), generated in the original-MLP study but unused for occupancy-weighted hyperparameter selection. Completed 20 paired 4,000-update fits: ordinary versus weighted moments at gains 4 and 8. At gain 4, ordinary training folds in 2/5 and weighted moments in 0/5; at gain 8, both ordinary and weighted training fold in 5/5. Weights use the same privileged distance-based 32-bin estimator; actors and encoders' model class are unchanged.

Continued both new gain-4 ordinary folds (141, 143) for 8,000 more updates at 0.0001, comparing ordinary continuation, weighted moments retaining Adam, and weighted moments resetting Adam. All six continuations remain folded. Weighted moments therefore show standard-gain prevention, not consistent removal of an existing fold. No reinitialization, order constraint, collision gradient or outcome-driven exclusions are used. [Results](weighted-moments-validation/output/RESULTS.md), [prevention/stress figure](weighted-moments-validation/output/prevention.png), [repair figure](weighted-moments-validation/output/repair.png). This is the same renderer, not evidence for arbitrary observations or a continuous-domain guarantee.

All 26 new checkpoints are reloaded for independent float64 20,001-point geometry checks, source/plan/initialization checks, and exact direct-repair start checks. A separate 30-checkpoint historical weighted-moment geometry re-audit is retained, distinguishing fresh recomputation from previously saved audits. Full evidence is in the [verification](weighted-moments-validation/output/verification.json), [historical re-audit](weighted-moments-validation/output/historical-reaudit.json), and [summary](weighted-moments-validation/output/summary.json).

Published a portable GitHub snapshot of all collected November 13–15 folding work on `codex/icm-folding-update`, in [draft PR #8](https://github.com/c-tustin/JEPA-Identifiability-Predictive-Subspaces/pull/8). Main remains unchanged pending explicit merge authorization. Original qualitative-study code and baseline data are reused from the existing main branch via relative links; no unrelated branch is merged. The [snapshot exporter](export_github.py) records original and published hashes and layout adaptations. All source code, protocols, reports, PNG/PDF figures and JSON audits are included. Recent policy source data and new learning-rate/validation checkpoints are bundled; older derived caches are explicitly omitted and inventoried, with regeneration instructions. Local canonical source-folder links and nested repositories are preserved.

## November 15: Requested occupancy galleries and learning-rate follow-up

Added [a paired learning-rate follow-up](inverse-occupancy-lr/README.md) directly under this folder, importing the existing linked occupancy study without duplicating it. Regenerated [native occupancy and encoder panels](inverse-occupancy-lr/output/native_occupancy_encoders.png) for non-folded seeds 130/134 and folded seeds 131/136, plus [bimodal and trimodal panels](inverse-occupancy-lr/output/bimodal_trimodal_encoders.png) with all three diagnostic seeds and three intervention methods. Occupancies use equal log-distance bins; controlled multimodal distributions are SIGReg sampling interventions, not new actors.

Completed 15 new 4,000-update fits on historical sources 130, 131, 136. Inverse-gradient scaling folded in 2/3 seeds at 0.25-times LR, 1/3 at the reused original rate, 2/3 at 4-times LR, and 3/3 with per-step full raw-gradient-norm compensation. Weighted moments folded in 0/3 at each of 0.25, 1, and 4 times LR (mean log-distance R² 0.9878, 0.9841, 0.9804). Ordinary original-rate controls fold in 2/3. Rate tuning does not establish a reliable gradient-only fix. [Results](inverse-occupancy-lr/output/RESULTS.md), [learning-rate curves](inverse-occupancy-lr/output/learning_rate_comparison.png), [machine-readable summary](inverse-occupancy-lr/output/summary.json).

All 15 new final curves, slopes, roots and R² values were verified after checkpoint reload; the seed-130 1-times inverse-gradient control was separately rerun and reproduced the original weights bit-for-bit. Analytic/autograd checks validate inverse-gradient scaling and weighted moments; [audit](inverse-occupancy-lr/output/audit.json). Source checkpoints and training code hashes are retained in each new checkpoint. Adam preconditioning prevents interpreting raw-norm compensation as exact final-update matching. These are finite-grid prevention-from-initialization results on historical sources, not fresh-seed evidence, late repair, actor retraining, or an injectivity guarantee.

## November 15: LaTeX proof and fresh training-code audit (completed)

Added a [mathematical note](no-fold-proof/no-fold.tex), [compiled PDF](no-fold-proof/output/pdf/no-fold.pdf), and [reproducible verifier](no-fold-proof/verify.py) directly under this workspace. The note proves the projected-knot and positive-slope guarantees over the entire interval, derives the occupancy-weighted characteristic-function gradient, and supplies explicit counterexamples for finite squared penalties and finite probe checks. A sufficient unchanged-MLP certificate is proved but is not implemented.

Freshly recomputed final geometry and losses for all 100 historical constrained fits and their 50 free MLP comparators. All constrained fits retain positive interval slopes and no folds. An additional 100,000 random ordered pairs satisfied the minimum-increment inequality within float64 tolerance. In a new 500-update adversarial Adam run, all 500 proposals had negative slopes and all 500 projected updates passed the positive-margin certificate (minimum computed slope 0.04999999999996414). PAVA passed 1,000 new random primal/dual KKT checks. [Machine-readable evidence and checkpoint hashes](no-fold-proof/output/verification.json), [stress-test figure](no-fold-proof/output/training-stress.png).

These results validate the implementation under the known apparent-height ordering prior. The real-arithmetic proof supplies the all-input guarantee; empirical checks alone do not. Historical per-update minima and histories were inspected, not replayed. No new actor or ICM training was performed. General observations without a known ordered coordinate and downstream performance remain open below.

## What is folding, and why does it matter?

The fold is in the scalar SIGReg encoder trained after the ICM actor is frozen. It can map distinct physical distances to the same code. The ICM curiosity predictor uses a separate 32-dimensional feature encoder. Repairing the later scalar representation does not change the earlier actor, and the scalar fold cannot cause the earlier predictor-error rise.

In seed 131, the first observed turn enters near distance 15.81 at update 205, moves to 6.06 at 250, 3.44 at 500, and 2.37 at 4,000. The input transform and sampling interact with optimization; distance 2–4 is not a fixed physical folding threshold. Continuation can move the turn further inward.

Sources: [occupancy diagnosis](sources/explanations/occupancy-folding/README.md), [location study](../behavior-policy-experiments/behavior-policy-experiments/qualitative-study/fold-location/README.md), and [mechanistic audit](sources/explanations/occupancy-folding/mech-interp/README.md).

## Prevention and repair

| Intervention | Recorded outcome | Interpretation |
| --- | --- | --- |
| Ordinary SIGReg, native gain 4 | 5/10 final folds | Baseline failure |
| Inverse-occupancy scaling of old output gradients | 8/10 final folds | Does not reliably fix the objective |
| Inverse-occupancy-weighted sine/cosine moments from initialization | 0/10 final folds; mean log-distance R² about 0.982 | Effective prevention in this configuration |
| Weighted moments applied to five existing folds | 5/5 remain folded, including with Adam reset | Prevention does not imply late repair |
| Strong order penalty plus weighted moments, stored-observation probes | 0/5 remain folded; mean R² 0.9900 | Repairs the five tested gain-4 folds with an ordering prior |
| Gain-8 stress test, full occupancy correction | 7/10 final folds | Weighting alone is insufficient |
| Gain-8 stress test, stronger order penalty | 4/10 tiny residual turns | A finite soft penalty does not guarantee ordering |

Correct the global characteristic-function averages before differentiating. Scaling the old gradients leaves those averages tied to the original occupancy. Neither marginal Gaussian matching nor finite probe penalties certify injectivity between probes.

Sources: [baseline study](sources/explanations/occupancy-folding/README.md) and [repair study](sources/explanations/occupancy-folding/gradient-followup/README.md).

## Matched concentration changes the earlier multimodal interpretation

Earlier loose bimodal/trimodal controls did not match native near-object concentration. The matched study holds support, near profile, background, initialization, and alignment pairs fixed; its one/two-far-bump contrast also matches the first two log-distance moments.

| Probability below distance 2 | One far bump: final folds | Two far bumps: final folds |
| --- | ---: | ---: |
| 25% | 0/10 | 0/10 |
| 50% | 0/10 | 0/10 |
| 87.5% | 10/10 | 10/10 |
| 93.75% | 3/10 | 2/10 |

These are the weighted-estimator results. Stratified replay independently yields 10/10 folds for each far-tail shape at 87.5%. More peaks do not require more folds, and more concentration does not monotonically increase folding at this budget. The complete measure and optimization path matter. Late concentration switches retain the earlier branch even when Adam is reset in the tested seed-131 one-bump case.

Sources: [matched comparison](sources/explanations/occupancy-folding/matched-concentration/README.md) and [switch results](sources/explanations/occupancy-folding/hypothesis-guarantee/output/RESULTS.md).

## Hard constraints provide a stronger kind of fix

Two piecewise-linear encoders use the known ordered apparent-height coordinate: one parameterizes positive segment slopes, and the other projects knot values onto a minimum-slope order constraint after each update. Both require segment slopes of at least 0.05 with respect to apparent height. That constraint certifies ordering throughout the modeled interval, rather than only on a diagnostic grid.

Both constructions have zero final folds across their 100 total fits: ten native sources plus twenty source/tail combinations at each of 87.5% and 93.75%, per construction. At 87.5%, projected models have mean population objective 0.9825 versus 4.1697 for the free models, establishing that the tested folded solutions are not required for a lower objective. However, zero folds does not imply linear log-distance recovery: projected mean R² there is 0.5811. These constructions change the model class and explicitly use ordering knowledge.

Sources: [hard-order protocol](sources/explanations/occupancy-folding/hypothesis-guarantee/PROTOCOL.md) and [stored results](sources/explanations/occupancy-folding/hypothesis-guarantee/output/RESULTS.md).

## Why the policy stays near, and why prediction error rises

The original actors prefer approach in the far interior and repeatedly return toward the reflecting near boundary. Later sensor interventions strengthen the geometry account: native actors average distance 1.7012, mirrored-sensor actors 15.0518, and geometry-equalized actors 6.7011. Mirroring moves preference to the opposite boundary; equalization reduces the original near preference. The interventions also change representation difficulty, so they do not isolate a unique cause or prove a general ICM law.

Prediction error depends on changing features, policy coverage, and optimization. Fixed-input probes show large target-scale growth, while Adam-component interventions reveal inverse/forward objective conflict, momentum directions that can be uphill, and finite-step overshoot. Small early raw error alone does not establish a useful learned world model, and later increases cannot all be dismissed as harmless rescaling.

Sources: [actor geometry results](sources/explanations/occupancy-folding/hypothesis-guarantee/output/RESULTS.md), [predictor follow-up](sources/explanations/occupancy-folding/gradient-followup/README.md), and [optimizer audit](sources/explanations/occupancy-folding/mech-interp/README.md).

## Remaining work

- Validate the proposed repairs on fresh seeds and observation models beyond this known one-dimensional geometry.
- Measure downstream prediction or planning performance for repaired and hard-constrained representations; fold removal alone does not establish task improvement.
- Develop a correction or order constraint for observations without a known physical ordering coordinate.
- Separate actor-learning causes further with matched geometry and optimizer interventions.

This report merges existing findings and artifacts into one entry point. No training was rerun as part of the consolidation.

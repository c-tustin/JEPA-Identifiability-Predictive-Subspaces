# Merged progress: ICM policy, folding, and repairs

Updated November 17, 2026 from the studies linked in [the index](README.md). Most historical results use previously inspected diagnostic sources, primarily seeds 130–139, rather than a new independent validation cohort. Fold counts from different protocols should be read with their stated grid and support definitions.

## November 17: Mathematical conditions for fold development

Added [the mathematical framework](fold-development/README.md) directly under this folder. It derives the occupancy-weighted characteristic-function force, its contraction with the encoder response kernel, the alignment pair contribution, and Adam's current/history decomposition with the actual shared denominator. It distinguishes marginal occupancy from transition pairs and derives the expected iid finite-batch objective correction, with separate treatment of correlated endpoints.

The note proves sufficient next-update conditions using two opposite slope witnesses and a bounded finite-step remainder, and extends them to finite-horizon cumulative bounds with a surviving positive anchor. It also describes smooth interior versus boundary onset, existing-root motion, and a verification protocol for the original encoder. The existing seed-131 birth-event numbers are used as inspected empirical evidence; no new training or rigorous neural-network remainder enclosure is claimed. Computing validated remainder/error bounds and propagating training-state enclosures remain pending. The source identities and numerical examples are checked algebraically and with a small independent numerical validation; document links are checked before publication.

## November 16: Portable OU publication update

Extended [the exporter](export_github.py) to include both shared-input OU studies in the portable research snapshot, alongside the already indexed EMA and occupancy-integral studies. The update is published directly to `main` in `icm-folding/`, per the user's publication preference; the earlier PR #8 is already merged. Future publication must not create Codex-named folders or feature branches. Source code, protocols, PNG/PDF figures, numerical results, audits and source/checkpoint hashes are included. The 30 derived OU checkpoint files (about 515 MB) remain local, following the snapshot's derived-cache policy; `PUBLISHING.md` records the exact generation order required before running checkpoint-dependent reports. Local source-folder links and nested repositories are preserved.

## November 16: ICM-occupancy-matched OU (completed exploratory comparison)

Added [study](icm-matched-ou/README.md), [protocol](icm-matched-ou/PROTOCOL.md), and [generator/trainer](icm-matched-ou/run.py) directly under this folder. The OU state is mapped through the Gaussian CDF to a percentile, then through each seed's inverse empirical endpoint CDF to an actual original sensor vector. This preserves native source multiplicities and empty bins in the stationary target, without smoothing or a uniform background. Both alignment and ordinary SIGReg receive the exact same output tensor. The earlier multimodal code/checkpoints remain unchanged.

All twelve 4,000-update fits completed for gallery seeds 130/131/134/136: native shared-input pairs plus OU dt=0.01/0.1. Native controls fold in 2/4; OU dt=0.01 in 2/4; OU dt=0.1 in 3/4. Seeds 131/136 fold in all three arms. Respective native/small-dt/large-dt turns are 2.407/2.542/2.264 for seed 131 and 3.124/3.377/3.213 for seed 136. The additional large-dt seed-130 turn is at 6.465. Every final turn lies within actual training support. All four seeds exhibit at least one sampled transient root in every arm; these histories are sampled every 100 updates on a coarser grid.

Initializers and pair-index streams are paired across all arms; normal innovations are paired between OU arms. Generator recurrence, exact source-observation selection, adjacent-pair continuity, rank-cell correctness, empty-bin preservation, and imported shared-loss audits passed. All twelve final checkpoint curves/slopes reproduce exactly in float32 after reload, followed by independent 20,001-point float64 derivative checks. Mean measured 64-log-bin TV from the ICM target is 0.0534 for dt=0.01 and 0.0221 for dt=0.1. Source near-object mass is preserved approximately in finite correlated rollouts; original native data have TV=0 by definition. See [full results](icm-matched-ou/output/RESULTS.md), [histogram/encoder/slope gallery](icm-matched-ou/output/icm_matched_histograms_encoders.png), [training histories](icm-matched-ou/output/fold_histories.png), [compact shared-input multimodal gallery](icm-matched-ou/output/bimodal_trimodal_shared_inputs.png), [OU quantile transformation diagram](icm-matched-ou/output/ou_quantile_transform.png), [verification](icm-matched-ou/output/verification.json), and [provenance](icm-matched-ou/output/provenance.json).

This targets the empirical marginal in the stationary population, not ICM transition dynamics, action bounds or actor-training compute. In particular OU median distance increments differ from the native actor and some steps exceed 0.25. The comparison recreates folding on concentrated ICM-like occupancy with shared-input OU data; it does not show that occupancy alone uniquely determines fold occurrence or position. Bounded-action equivalence, fresh-source replication and transition-kernel matching remain open. Conclusions and fold checks remain exploratory and empirical on four diagnostic sources.

## November 16: Multimodal OU transitions shared by alignment and SIGReg (completed exploratory comparison)

Added [generator and trainer](shared-occupancy-ou/run.py), [protocol](shared-occupancy-ou/PROTOCOL.md), and [study](shared-occupancy-ou/README.md) directly under this folder. This addresses the user's requested correction: both losses consume exactly the same temporal pairs and the same encoder-output tensor, without separate regularizer inputs or occupancy weights. Original SIGReg-only multimodal results retain their earlier interpretation.

The passive process transforms a stationary latent OU chain through the prescribed log-distance inverse CDF. It gives the earlier bimodal/trimodal 64-bin target occupancy in the population; dt controls temporal correlation. Completed all 18 fits: uniform-log/bimodal/trimodal targets at dt=0.01/0.1 on diagnostic seeds 130/131/136, 4,000 updates each. No final folds were detected in any condition (0/12 multimodal fits and 0/6 uniform controls). Final encoder standard deviations on a uniform-log grid span 0.8114–0.8908. Measured endpoint TV from the multimodal target averages 0.0531/0.0536 at dt=0.01 and 0.0260/0.0271 at dt=0.1 for bimodal/trimodal, respectively. Correlated samples are not treated as independent observations.

Generator inverse-CDF/occupancy, exact OU replay, adjacent-pair continuity, sensor reconstruction, one-forward-call, same-tensor and both-endpoint-gradient audits passed. All 18 checkpoints were reloaded, their saved float32 curves/slopes reproduced exactly, and final slope sign reversals independently checked on 20,001-point float64 grids. Generated trajectories, observations, pair indices, normal innovations and initial weights were independently regenerated and paired across conditions. See [results](shared-occupancy-ou/output/RESULTS.md), [measured occupancy/trajectory/encoder figure](shared-occupancy-ou/output/shared_occupancy_encoders.png), [transition matrices](shared-occupancy-ou/output/transition_matrices.png), [audit](shared-occupancy-ou/output/audit.json), [verification](shared-occupancy-ou/output/verification.json), and [source/runtime provenance](shared-occupancy-ou/output/provenance.json).

This is a privileged simulator-coordinate generator with variable continuous distance increments, not the original bounded three-action actor. For bimodal/trimodal targets, about 31%/46% of dt=0.01 moves and 60%/73% of dt=0.1 moves exceed 0.25. Bounded-action implementation and fresh-seed replication remain open. Finite-grid fold checks and finite-sample occupancy diagnostics remain empirical; stationary marginal preservation is a property of the specified process. Changing target occupancy also changes transition geometry, so this is not an occupancy-only causal contrast.

## November 15: EMA fold-development experiment (completed)

Added [EMA training and evaluation study](ema-study/README.md) directly under this folder with a [fixed protocol](ema-study/PROTOCOL.md). Ten historical source seeds 130–139; original scalar MLP, gain 4, native sampling and unchanged original training schedule. Track parameter averages (0.9/0.99/0.999) without changing live training; separately train with detached EMA targets (0.99/0.999) and a symmetric alignment sum chosen to match the original gradient at zero lag. A full seed-131 zero-lag control checks this equivalence. There is no added predictor, and this is not a complete BYOL/image-JEPA implementation.

Completed all 31 fits. Final original-live folds are 5/10; all three evaluation weight averages also fold in 5/10. EMA-target training yields 7/10 folded students at decay 0.99 (six material) and 9/10 at 0.999 (eight material), with targets folded in 7/10 and 9/10 respectively. Original-live models have sampled transient/final folds in 9/10 runs, versus 10/10 for evaluation averages at 0.99/0.999. EMA is not a reliable fold-prevention method in this tested configuration. Seed 131's live/0.9/0.99/0.999 first sampled folds occur at updates 225/225/300/450; all remain folded at 4,000.

All 82 saved final encoders across the 31 fits passed independent float64 20,001-point curve/derivative reload checks. All ten live originals and the complete seed-131 zero-lag run reproduce original parameters bit-for-bit. EMA recurrence, target detachment and zero-lag alignment-gradient checks passed. Minimum output standard deviation remains above 0.65. Training histories sample geometry every 25 updates on 1,025 points; missed temporal/inter-grid folds remain possible. [Results](ema-study/output/RESULTS.md), [fold-development figure](ema-study/output/fold_development.png), [per-seed summary](ema-study/output/summary.json), [verification](ema-study/output/verification.json).

All encoders start from initialization; no late repair, downstream benefit, continuous-domain guarantee or conclusion about all BYOL/JEPA recipes is implied. Decays were fixed in advance; LR retuning, gain-8 stress and fresh-source validation were not performed. The local snapshot exporter includes EMA artifacts and the already-indexed occupancy-integral diagnostic for future publication; the integral dependency path is adapted only in exported copies.

## November 15: Occupancy integral evaluated at fold birth (completed diagnostic)

Added [a reproducible integral diagnostic](occupancy-integral/README.md), [code](occupancy-integral/run.py), [results](occupancy-integral/output/results.json), and [figure](occupancy-integral/output/integral.png). At seed 131/gain 4 immediately before update 205, the empirical occupancy/slope integral at d=16 is +38.779525 under native weights and −5057.090716 under equal occupied log-bin weights, holding the encoder fixed. The corresponding regularizer-only SGD linearizations at coefficient 0.1 and rate 0.001 are −0.00387795 and +0.50570907. All 32,768 stored endpoints are used, with the regularizer multiplier fixed at the training batch size 1024. The native dataset spans d≈1.00062–13.73643; the query lies outside that observed range. One of 32 log-distance bins is empty and is not synthesized.

The exact historical minibatches and Adam before-state reproduce the saved after-state with zero maximum parameter error. The measured slope changes +0.004150231 → −0.000832708. The actual-update mixed-derivative contraction predicts −0.004968168 versus the measured −0.004982939. With the actual combined Adam denominator, current SIGReg contributes **+0.001426908**, historical SIGReg momentum **−0.006383163**, and the alignment component including history **−0.000011922**. Thus the instantaneous current gradient opposes this fold-creation step; accumulated regularizer momentum drives it. The actual minibatch's raw integral is −139.440067, differing in sign from the full-data plug-in value.

Independent forward/reverse derivative contractions agree to about 1e−12 or better; a central finite difference checks the actual-update contraction. The diagnostic retains bin contributions, 65-distance profiles, fixed-moment/outer-measure contrasts, finite SGD-step slopes, and provenance hashes. No new training sweep or no-fold guarantee is claimed. This is one inspected source/update; full empirical moments differ from the expected finite-batch objective. Historical per-bin occupancy attribution, replication across seeds, and analysis of later fold motion inside observed support remain next steps.

## November 15: Original-MLP gradients and regularizers (completed)

Added [study and reproduction](mlp-gradient-study/README.md), [protocol](mlp-gradient-study/PROTOCOL.md), and [training code](mlp-gradient-study/run.py). Completed 145 encoder fits: 80 main paired fits, 30 adaptive global-mining fits, ten matched-strength local controls, and 25 fits on five freshly trained ICM policy sources. Every encoder is the original scalar MLP trained from random initialization. Per-update inputs contain observations/actions only; no ordered spline, physical ordering labels, or analytic dynamics enter the gradient.

Baseline final fold counts on ten development sources: SIGReg 5/10, scalar VICReg 0/10, distance-score InfoNCE 0/10, Gaussian quantile matching 9/10. On five fresh policy sources, respectively: 2/5, 1/5 tiny (0 substantial), 0/5, 5/5. Scalar VICReg has no off-diagonal covariance term; scalar InfoNCE uses an RBF distance score rather than degenerate scalar cosine. These are toy adaptations, not general method rankings.

The local successor-conflict gradient at coefficient 1 has little effect. Global mining at coefficient 100 reduces SIGReg folds to 1/10 on development sources and improves all ten held-out observation-prediction errors. The coefficient was fixed before generating fresh sources; there it reduces folds 2/5 -> 0/5 and improves all five prediction errors (mean MSE 0.135788 -> 0.000334). A matched-strength local coefficient-100 control leaves 4/10 sign reversals, including 3/10 substantial folds. Mining has additional compute and observation exposures, so this is not a matched-compute comparison. One global-mining development fit remains substantially folded, disproving any universal guarantee for this gradient rule.

All 145 final models were independently reloaded, checked on 20,001-point derivative grids, and their stored downstream decoder errors recomputed on whole held-out trajectories and a common coverage-stress set. Four objective finite-difference gradient checks, the collision-gradient direction, and global-sign invariance passed. Development coefficients were adaptive; the fresh cohort had fixed hyperparameters. Five fresh sources remain a small same-renderer validation. [Gallery](mlp-gradient-study/output/index.html), [main results](mlp-gradient-study/output/RESULTS.md), [mining](mlp-gradient-study/output/MINING.md), [fresh results](mlp-gradient-study/output/FRESH.md), [strength control](mlp-gradient-study/output/STRENGTH.md).

## November 15: Weighted-moment validation, direct-repair replication and GitHub snapshot

Added a [fixed-protocol validation](weighted-moments-validation/PROTOCOL.md) using five recent ICM sources (140–144), generated in the original-MLP study but unused for occupancy-weighted hyperparameter selection. Completed 20 paired 4,000-update fits: ordinary versus weighted moments at gains 4 and 8. At gain 4, ordinary training folds in 2/5 and weighted moments in 0/5; at gain 8, both ordinary and weighted training fold in 5/5. Weights use the same privileged distance-based 32-bin estimator; actors and encoders' model class are unchanged.

Continued both new gain-4 ordinary folds (141, 143) for 8,000 more updates at 0.0001, comparing ordinary continuation, weighted moments retaining Adam, and weighted moments resetting Adam. All six continuations remain folded. Weighted moments therefore show standard-gain prevention, not consistent removal of an existing fold. No reinitialization, order constraint, collision gradient or outcome-driven exclusions are used. [Results](weighted-moments-validation/output/RESULTS.md), [prevention/stress figure](weighted-moments-validation/output/prevention.png), [repair figure](weighted-moments-validation/output/repair.png). This is the same renderer, not evidence for arbitrary observations or a continuous-domain guarantee.

All 26 new checkpoints are reloaded for independent float64 20,001-point geometry checks, source/plan/initialization checks, and exact direct-repair start checks. A separate 30-checkpoint historical weighted-moment geometry re-audit is retained, distinguishing fresh recomputation from previously saved audits. Full evidence is in the [verification](weighted-moments-validation/output/verification.json), [historical re-audit](weighted-moments-validation/output/historical-reaudit.json), and [summary](weighted-moments-validation/output/summary.json).

Published a portable GitHub snapshot of all collected November 13–15 folding work on `codex/icm-folding-update`, in [draft PR #8](https://github.com/c-tustin/JEPA-Identifiability-Predictive-Subspaces/pull/8). The user explicitly authorized merging PR #8 into `main`; the pull request records merge status. Original qualitative-study code and baseline data are reused from the existing main branch via relative links; no unrelated branch is merged. The [snapshot exporter](export_github.py) records original and published hashes and layout adaptations. All source code, protocols, reports, PNG/PDF figures and JSON audits are included. Recent policy source data and new learning-rate/validation checkpoints are bundled; older derived caches are explicitly omitted and inventoried, with regeneration instructions. Local canonical source-folder links and nested repositories are preserved.

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

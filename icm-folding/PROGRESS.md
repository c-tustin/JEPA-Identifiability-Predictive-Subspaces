# Merged progress: ICM policy, folding, and repairs

Updated November 19, 2026 from the studies linked in [the index](README.md). Most historical results use previously inspected diagnostic sources, primarily seeds 130–139, rather than a new independent validation cohort. Fold counts from different protocols should be read with their stated grid and support definitions.

## November 22: One framework for the fold; encoder geometry decides the response

- **Framework.** [FRAMEWORK.md](occupancy-feature-isolation/FRAMEWORK.md). Three ingredients:
  - force: a shape target on the code histogram;
  - target: a skewed initial histogram with the tail in SIGReg's pull-back band;
  - response: the encoder's tangent kernel lets the tail bend faster than the heavy cluster can stretch.
- **Geometry test** (pre-registered, 48 fits). Same occupancies, initial weights and training; only the input map
  changed. `tanh(4x)` folds at m = 0.625–0.875 (6/6). `x` and `tanh(x)` never fold (0/32, m = 0.125–0.9375) and reach
  skewness ≈ 0 monotonically.
  - The pre-registered skewness-in-f0 rule failed: 19 of 38 correct.
  - H1 failed (m = 0.9375 does not fold under `x`); H2 failed (no mirror fold at m = 0.125); H3 held (m = 0.5 does not
    fold).
- **Compact-cluster test** (pre-registered, 32 fits). Falsified "a cluster with tiny input span suffices" (4 of 11
  correct). A [1, 1.02) cluster under `x` did not fold; a [1, 1.25) cluster folded at initialization 133 under `x` and
  `tanh(x)`.
- **Observed, not pre-registered:**
  - Race replays (bit-exact): under `tanh(4x)` the tail's far end overtakes its median by step ~400. Under `x` the tail
    moves as a block and the cluster spreads.
  - Tangent-kernel decoupling at initialization (near cluster / tail): `tanh(4x)` 0.003–0.005 / 0.14–0.16; `x` 0.06 /
    0.02–0.03.
- **Remaining:**
  - a prospective, kernel-based prediction on new maps or initializations;
  - a law-weighted kernel statistic;
  - an analytic account of the force zero-crossings.

## November 21: Occupancy feature and InfoNCE mechanism (identified, with limits)

- **SIGReg as a kernel distance.** The encoder is a scalar function of distance. SIGReg1D is exactly 1024 × MMD² to
  N(0,1) with a unit Gaussian kernel.
- **Force structure.** On positively skewed code laws, the exact SIGReg force pulls back codes at standardized
  1.1 < z < 3.45 and pushes codes beyond about 3.45 outward.
- **Training replays.** Bit-exact replays of the dose fits show three regimes: monotone at 50% near mass, a fold at 75%
  (the far branch is retracted over the near branch), and ejection at 93.75% (the far tail is thrown to code about 14,
  beyond the kernel).
- **Why near mass looked causal.** The tanh-compressed initialization geometry makes the near cluster nearly a point,
  so near mass almost fixes code skewness.
- **Pre-registration.** The [skewness protocol](occupancy-feature-isolation/SKEW_PROTOCOL.md) and its hashed
  predictions were pushed to GitHub before training.
- **Part A, 38 fits** reweighting actual source observations. 20 of 26 scored predictions were correct.
  - Removing asymmetry at fixed near mass 0.75, 0.875 and 0.9375 stopped the fold.
  - Raising skewness by piling far mass at distance 2–3 did not cause folds at 0.5, 0.9375 or 0.96875. These failures
    falsify skewness as a sufficient feature.
- **Part B, 29 fits.**
  - Even-only SIGReg (sine term removed) still folds, falsifying that pre-registered prediction.
  - A pure skewness penalty makes raw InfoNCE fold at 75% and 93.75% near mass (backward 0.29–0.32) and makes the
    variance control fold at several laws.
- **[Cluster-position protocol](occupancy-feature-isolation/CLUSTER_PROTOCOL.md), 24 fits.**
  - The wall hypothesis is falsified: at 75% every window folds.
  - Compact wall clusters fold even at 93.75%.
  - Three of 31 laws flip fold status with the initialization.

Conclusion:
- The causal feature is the occupancy's asymmetry in the encoder's input geometry: a heavy cluster the encoder cannot
  spread, plus a lighter tail spread over distance.
- It is necessary (136 of 138 junction folds in 338 cells have skewness > 0.45) but not sufficient.
- Whether it folds depends on tail spread, cluster compactness, and, near the boundaries, the initialization. No exact
  threshold on the distribution alone exists for this encoder.
- InfoNCE does not fold because it sets no target shape for the code marginal; adding one makes it fold under the same
  occupancy.
- Junction folds are now separated from tiny far-end turns with a backward-fraction threshold of 0.1.

[Results](occupancy-feature-isolation/output/skew/RESULTS.md). All are empirical checks, not analytic guarantees.

Remaining work:
- a quantitative sufficient criterion combining tail spread and cluster compactness, tested on new sources;
- other encoder input geometries;
- an analytic account of the ejection boundary.

## November 19: Occupancy-feature isolation (started; no final feature claim)

The user clarified that identifying the SIGReg data pathway does not explain which distribution feature causes the historical five/five folding split. Added a [separate feature-isolation protocol](occupancy-feature-isolation/PROTOCOL.md) and [trainer](occupancy-feature-isolation/run.py). First-stage design: five preselected folded/nonfolded source pairs, independently swapping mass below distance 2, conditional shape below 2, and conditional shape at/above 2 (40 fits). Keep initialization 131, original architecture/optimizer/budget and source-131 alignment stream common. A shared float64 uniform plan samples exact empirical hybrid measures without smoothing or new states. All ten native empirical corner-recovery checks passed; source profiles and provenance are recorded in [preflight](occupancy-feature-isolation/output/preflight.json). Training has started; completed outcomes and independent audits are pending.

All 40 hybrid fits have now completed and passed independent dense geometry, component normalization, source/initialization/plan and fixed-component audits. All 20 mass-only paired changes introduce material folds; all 20 near-shape-only and all 20 far-shape-only changes leave material status unchanged. Any-turn status has one exception: high near mass (93.500%) and source-137 near shape combined with source-136 far shape produces a tiny in-range turn at 12.4203 (backward variation 0.0007723). The exception is retained. [Results](occupancy-feature-isolation/output/RESULTS.md), [matrix](occupancy-feature-isolation/output/hybrid_components.png), [contrasts](occupancy-feature-isolation/output/summary.json), [audit](occupancy-feature-isolation/output/verification.json).

Started a [120-fit controlled mass-dose/objective study](occupancy-feature-isolation/DOSE_PROTOCOL.md): ten masses × four conditional-profile templates × SIGReg, raw/own-code InfoNCE and an explicitly labeled separate-input variance control. Fixed normalization and common plans prevent raw normalization from confounding the occupancy intervention. [A mechanism protocol](occupancy-feature-isolation/MECHANISM_PROTOCOL.md) specifies exact finite-batch quadratic risks for fixed trained candidates, an exact two-input InfoNCE collision check, and frozen-encoder critic comparisons. Results and audits for these stages are pending; no complete feature/mechanism claim is made.

Completed and independently audited the 120-fit mass-dose/objective sweep. SIGReg has 20/40 material folds, concentrated at intermediate near mass; raw/own-code InfoNCE has 0/40 turns; the explicitly separate-input variance control has 28/40 turns, mostly tiny, including 2/40 material at low mass. [Dose results](occupancy-feature-isolation/output/dose/RESULTS.md) and [figure](occupancy-feature-isolation/output/dose/dose_response.png). Exact expected finite-iid-batch SIGReg risk quadratics cross between ordered/folded candidates near the dose boundaries, but are only finite-candidate comparisons. All 54 frozen-encoder critic probes passed independent reload/reconstruction; the folded candidate has worse raw/own-code InfoNCE in all 36 paired comparisons (excess 0.3938–1.1299). [Mechanism results](occupancy-feature-isolation/output/critic-probe/RESULTS.md).

Completed and audited 120 initializer/sampling replications: at masses 0.5/0.75/0.9375, SIGReg material folds are 0/20,20/20,0/20, and raw/own-code InfoNCE is 0/20 at every mass. [Replication](occupancy-feature-isolation/output/robustness/RESULTS.md). Ten newly trained policy sources (9100–9109) and all predictions were frozen before new encoder fits. Their 80 controlled fits are audited: SIGReg folds 3/10 at 0.5,10/10 at 0.75,0/10 at 0.9375; raw/own-code InfoNCE has no final turn. The predeclared risk bank predicts 8/10 native outcomes, the coarse mass window 9/10. Native sources 9105/9109 falsify the risk-bank prediction; 9109 at 56.72% near mass also falsifies the coarse lower boundary. [New-source results](occupancy-feature-isolation/output/fresh/encoders/RESULTS.md). Do not promote the mass window or bank to an exact feature criterion.

Running explicitly motivated follow-ups: [fixed-50% conditional near/far shape swaps](occupancy-feature-isolation/FRESH_SHAPE_PROTOCOL.md) on the fresh exceptions; [all-four-loss joint-data comparisons](occupancy-feature-isolation/JOINT_PROTOCOL.md) with exact native joint-law recovery, controlled endpoint mass and quantified conditional-shape coupling; [alternative split/common-support checks](occupancy-feature-isolation/SPLIT_SUPPORT_PROTOCOL.md). All joint preflight native-corner, endpoint-mass and one-forward/both-endpoint-gradient checks pass. These outcomes and their independent audits remain pending. The full active goal is not complete.

This is an exploratory decomposition of reused sources. Distance 2 is a prior diagnostic split, not a discovered universal threshold. The goal remains open until a specific feature survives direct interventions and robustness checks, and the difference from InfoNCE/other objectives is tested under that same feature intervention with a verified mechanism. Neither the prior source-swap experiment nor old zero-fold contrastive results alone meets that requirement.

## November 19: Saved data isolated from minibatch sampling (completed)

Completed the [fixed-protocol sampling crossing](sampling-crossing/PROTOCOL.md): ten unchanged datasets × all ten original initializations × five independent paired alignment/regularizer streams, 500 fits at 4,000 updates each. Every plan's index tensors are shared across all datasets and initializations, removing the earlier dataset/plan association. [Results](sampling-crossing/RESULTS.md), [matrices](sampling-crossing/output/sampling_crossing.png), [cell measurements](sampling-crossing/output/summary.json) and [independent audit](sampling-crossing/output/verification.json) are saved. The batch began with four one-thread workers and resumed from validated checkpoints with eight; the scientific cells and update sequence were unchanged.

There are 270/500 detected final turns and 256/500 material folds. A secondary range diagnostic finds 250/500 turns inside each dataset's observed distance range, all from datasets 131/132/133/136/139; the 20 extra turns from 134/135 lie outside their respective stored ranges. Across five fresh streams, no fixed dataset/init pair changes binary fold status (0/100 pairs, 0/1000 stream-pair comparisons); one pair changes material status (dataset 134, initialization 136; 4/1000 stream-pair comparisons). Datasets 131/132/133/136/139 fold materially in 50/50 cells each. Datasets 130/137/138 have no final turn in 50/50 each. Datasets 134/135 have 10/50 turns each, with 6/50 and 0/50 material folds respectively. The descriptive balanced-factor decomposition assigns 87.118% of binary-outcome centered sum of squares to dataset main effects, 2.576% to initialization, 10.306% to their interaction, and zero to terms involving the five fresh sampling plans. For material folds, the dataset main-effect fraction is 95.774%. These are finite-matrix descriptive shares, not population causal fractions.

Two full original replays (130/131), short exact model/Adam resume validation, all original source/initialization hash checks, and all 500 independent dense final reload/geometry checks passed. The old dataset-associated historical stream had one extra tiny turn (dataset 137/init 130) absent under all five fresh streams; sampling can therefore affect marginal cases outside the new five-plan set. Saved data dominates the tested binary pattern, but data still bundles actor learning and trajectory collection. No fresh actor, recollection, downstream improvement or analytic no-fold guarantee is claimed. Checkpoints remain local with hashes and reproduction commands published; separating actor learning from collection remains next.

## November 19: Causal optimizer-history intervention and loss-source crossing

Added [historical sample attribution and Adam numerator surgery](occupancy-integral/momentum_history.py). All 205 seed-131 updates reproduce the saved birth-event weights exactly. Historical per-sample regularizer contributions sum to −0.0063831652 through the actual shared denominator, matching the saved history contraction within 3.7e−10. At identical before-weights/current gradients/second moments, removing SIGReg first-moment history changes the actual post-step far slope from −0.000832708 to +0.005567166 and removes the detected turn; removing alignment history leaves a turn at 15.8136. [Measurements](occupancy-integral/output/momentum_history.json), [figure](occupancy-integral/output/momentum_history.png), and [scope](occupancy-integral/README.md#november-19-historical-attribution-and-actual-optimizer-state-intervention). This is a causal one-step intervention at one historical event, not universal or permanent fold prevention.

Extended the local optimizer intervention to every detected ordered-to-folded event through update 500 in the five original failure runs. All five full 4,000-update paths match their originals exactly; all 32 fork checkpoints pass independent dense reload checks. Removing SIGReg history prevents six of eight events, including at least one in each source; removing alignment history prevents none. Source 139's first-update transient and source 132's update-368 event still fold without first-moment history, ruling out universal momentum necessity. [Complete event table](occupancy-integral/output/BIRTH_RESULTS.md), [figure](occupancy-integral/output/birth_replication.png), [fixed protocol](occupancy-integral/BIRTH_PROTOCOL.md), [audit](occupancy-integral/output/birth_verification.json).

Independently reloaded and remeasured all 100 dataset × initialization cells; every stored geometry reproduces exactly, all ten historical diagonals remain bit-exact, and all initialization/plan hashes regenerate. Of 55 final grid folds, 50 have turns within source support; 51 satisfy the material threshold over [1,16]. [Audited results](initialization-crossing/output/RESULTS.md), [matrix](initialization-crossing/output/crossing.png), [audit](initialization-crossing/output/verification.json).

Completed the [fixed-protocol 100-cell alignment-source × regularizer-source crossing](loss-source-crossing/PROTOCOL.md), holding initialization 131 and both index plans common across all cells. Final status follows SIGReg source in every cell: sources 131/132/133/136/139 fold under all ten alignment sources, the other five under none (50/100 material folds). Changing alignment source flips zero statuses. All 50 turns lie within both sources' observed support. The (131,131) control is bit-exact.

Completed [three iid endpoint plans per regularizer source](loss-source-crossing/IID_PROTOCOL.md) and [ten alignment-removal fits](loss-source-crossing/ABLATION_PROTOCOL.md). All 30 iid fits and all ten alignment-free fits preserve that same five/five source split. The former preserve the full empirical endpoint marginal in expectation while removing paired-endpoint dependence; the latter use only 0.1 SIGReg. Thus neither temporal alignment nor paired endpoint dependence is necessary for these tested folds. All 140 final geometries reproduce exactly after independent dense reload; identity and regenerated-plan checks pass, and 130 crossing/IID models pass derivative finite differences. The renderer/configuration is common across sources. [Causal interpretation and limits](loss-source-crossing/README.md), [results](loss-source-crossing/output/RESULTS.md), [matrix](loss-source-crossing/output/source_crossing.png), [audit](loss-source-crossing/output/verification.json).

This identifies the empirical endpoint-occupancy pathway through SIGReg under the tested architecture, initialization, optimizer and budget. Marginal distribution matching does not enforce physical order, and the optimizer interventions identify local reversal triggers. It does not establish a unique universal occupancy threshold, necessity of momentum for every event, permanent prevention, new-policy generalization, or an analytic no-fold guarantee. The separate independent-sampling study remains separately labeled with its own status.

## November 19: Scalar projected-area observations for InfoNCE

Added [a scalar-area study](scalar-area-infonce/README.md), [fixed protocol](scalar-area-infonce/PROTOCOL.md), [trainer](scalar-area-infonce/run.py) and [independent report](scalar-area-infonce/report.py). The observation adapter reduces the three projected vertices to one unsigned area using the determinant formula. For this fixed frontoparallel triangle, area is A0/d². Both the encoder and contrastive scorer are now 1→48→48→1 tanh MLPs (encoder input tanh(4A), linear output). No coordinates or distance labels enter either learned network. Own-area/code InfoNCE uses 256 candidates, training-only raw standardization and the existing 4,000-update Adam schedule. Shape variation is not tested; independently varying shape can make area ambiguous about distance.

All 15 fits on reused sources 130–144 completed. Detected final folds: **0/15**, minimum output std **3.468461**. The separate frozen-encoder decoder predicts next scalar area from code and action: mean held-out whitened MSE **0.162994**, mean uniform-distance coverage MSE **0.012023**. These targets differ from earlier six-coordinate targets, so MSE values are not directly comparable. Curves strongly compress some distances despite no sampled sign reversals. [Full results](scalar-area-infonce/output/RESULTS.md), [observation/encoder/distance figure](scalar-area-infonce/output/scalar_observation_and_representation.png), [all representations](scalar-area-infonce/output/distance_representations.png), [training and prediction](scalar-area-infonce/output/training_and_prediction.png).

Area shoelace, translation/rotation invariance, quadratic scaling, inverse-square distance, scalar-shape, explicit-score, finite-difference gradient and raw-detachment checks passed. Independent reloads verify source and batch hashes, initialization/statistics, dense 20,001-point float64 geometry through an independent analytic area renderer, physical-distance derivatives and saved InfoNCE/decoder errors. [Audit](scalar-area-infonce/output/verification.json). The exporter includes the new study and checkpoints.

The frozen actors and ICM were trained on six coordinates; this intervention changes only observations for the later representation and downstream decoder. Scalar-observation actor retraining, varying shape/noise, a tuned decoder and an all-input injectivity certificate remain unperformed. The scalar observation's analytic ordering is distinct from the learned encoder's empirical fold checks.

## November 19: Latest-only InfoNCE representations and guarantee scope

Added [a standalone latest-method plotter](raw-embedding-infonce/plot_latest.py), [all final distance representations](raw-embedding-infonce/output/latest_distance_representations.png), [seed-131 representation and physical-distance slope](raw-embedding-infonce/output/latest_seed131_distance.png), and [learning snapshots](raw-embedding-infonce/output/latest_distance_learning.png). These figures exclude alternative methods, retain original scalar code units and signs, and select learning examples by existing lowest/median/highest prediction error. Recomputed final geometries on 20,001 float64 points from hash-checked checkpoints; physical-distance derivatives use the chain rule and are compared with finite differences. [Audit](raw-embedding-infonce/output/latest_plots_verification.json).

Explained why finite-sample InfoNCE alone does not provide an all-input no-fold certificate. A validated fixed-sign derivative bound or strictly monotone architecture would supply a separate sufficient guarantee; neither has been implemented for these InfoNCE encoders. Existing zero-of-15 final fold results remain empirical. No training, data, hyperparameter or checkpoint changes.

## November 19: Clarified distance-representation figure labels

Renamed the raw/code representation figure titles and axes to distinguish physical distance from its learned scalar representation. Comparison legends identify teal as the latest raw/own-code InfoNCE. Documented unconstrained code units, sign and offset, and the display-only normalization in comparison panels. Regenerated figures from existing audited checkpoints and replay histories; no training or numerical-result changes. Empirical fold checks remain distinct from guarantees.

## November 19: Dataset × initialization crossing (completed)

Added [a fixed-protocol crossing](initialization-crossing/PROTOCOL.md), [trainer](initialization-crossing/run.py) and [study notes](initialization-crossing/README.md). The design crosses all ten historical frozen-policy datasets with all ten original scalar-encoder initializations (100 fits, 4,000 updates each). Both original alignment and regularizer minibatch streams are held fixed within each dataset row; initial weights are held fixed within each initialization column. Diagonal fits must reproduce the historical final encoders bit-for-bit before crossed fits proceed. Checkpoints retain provenance, initial/final parameters, Adam state and 100-update geometry histories; final geometry is measured after reload on a 20,001-point float64 derivative grid.

All 100 fits completed, with 55 detected final folds and 51 material folds. All ten diagonal replays exactly match the historical final parameters; pairing checks passed for all cells. The five originally folded datasets (131,132,133,136,139) fold under all ten initializations. Datasets 130/138 fold under none; 134/135 under initializations 130 and 136; 137 under initialization 130 only. [Completed results](initialization-crossing/RESULTS.md) and [cell measurements](initialization-crossing/output/summary-n4000.json) support a strong dataset/sampling-condition association, with initialization sensitivity in three rows. Dataset rows bundle actor learning, collection randomness and the fixed sampling plan, so this phase isolates initialization sensitivity conditional on those rows rather than separately identifying all sources of variation. Recollection from fixed actors and independent minibatch replications remain future phases. These are reused historical sources and finite-grid empirical checks, not fresh-source validation or injectivity guarantees.

## November 19: Dense raw/code InfoNCE convergence and representation plots

Recovered dense histories for all 15 completed raw/code fits by [deterministic replay](raw-embedding-infonce/convergence.py), with [methods](raw-embedding-infonce/CONVERGENCE.md), [plotter](raw-embedding-infonce/plot_convergence.py) and [measurements](raw-embedding-infonce/output/convergence/RESULTS.md). Original checkpoints and protocol are unchanged. Every replay exactly matches saved final encoder, scorer and downstream decoder tensors, all six original sampled training losses, and final full-dataset decoder errors. Encoder/critic fixed train/held-out sets are evaluated every 25 updates; whole downstream datasets every 50 separate decoder updates. Histories, hashes and [audits](raw-embedding-infonce/output/convergence/verification.json) are saved as portable derived artifacts.

[Training curves](raw-embedding-infonce/output/training_convergence.png) show median held-out InfoNCE 7.736 → 4.287. [Prediction curves](raw-embedding-infonce/output/prediction_convergence.png) show median held-out next-observation MSE 0.7178 → 0.001148 during the 1,500-update frozen-encoder decoder fit; final mean is 0.001224. These are different training stages and different target units. Bands are descriptive 10–90% ranges across reused sources, not confidence intervals. Intermediate-encoder downstream probe errors are not measured.

The additional raw-MSE diagnostic reveals scorer drift: raw reconstruction MSE initially falls then rises, despite improving contrastive loss. At least 99.927% of final held-out scorer MSE is perpendicular to the observed one-dimensional raw-data line in every run. Projecting away that component changes candidate log probabilities by at most 1.02e-6. Such a component adds a row-constant distance term and cancels from the softmax; the scorer must not be interpreted as a calibrated reconstruction model. This does not change the saved final representations or existing downstream measurements.

[Representation examples](raw-embedding-infonce/output/representative_representations.png) use sources 130, 144 and 141, selected by lowest/median/highest final held-out MSE, with initial/intermediate/final curves and final cross-objective comparisons. [Seed 131](raw-embedding-infonce/output/seed131_comparison.png) is shown separately as the earlier SIGReg fold case. Display-only centering, scaling and sign orientation are explicit. No new sources, tuning, repair or all-input guarantee is claimed.

## November 18: Raw observation versus own embedding InfoNCE

Added and completed [the raw/code study](raw-embedding-infonce/README.md) with a [fixed protocol](raw-embedding-infonce/PROTOCOL.md). Positives are `(x, f(x))`, negatives pair the same code with other raw observations. Raw coordinates are standardized using training-endpoint statistics, with no learned raw branch. A learned 1→48→48→6 network supplies a quadratic distance critic. This replaces temporal pairing rather than renaming the earlier temporal InfoNCE experiment; no temporal alignment, SIGReg, EMA or reconstruction-MSE term is added.

All 15 encoders start from the matched initializations and train for 4,000 updates on existing sources. Final detected folds: 0/10 historical and 0/5 reused follow-up, with minimum output standard deviation 1.525. Mean held-out next-observation MSE is 0.001102 and 0.001468, versus temporal InfoNCE's 0.002071 and 0.002560. Coverage-stress MSE is 0.058873 and 0.090494, so improvement is not uniform across objectives and evaluation distributions. These are descriptive comparisons with different critic, score units, candidate counts and compute, not matched-compute causal estimates.

All 15 new checkpoints were independently reloaded and checked on 20,001-point float64 derivative grids. Source/batch hashes, encoder initializations, training raw statistics and downstream decoder errors were verified. Explicit-score, finite-difference code-gradient, permutation, mismatch-sensitivity, raw-detachment and encoder/critic-gradient checks passed. [Results](raw-embedding-infonce/output/RESULTS.md), [curves](raw-embedding-infonce/output/curves.png), [audit](raw-embedding-infonce/output/verification.json). The exporter includes the new code, artifacts and checkpoints. Reused sources, one restricted critic and one hyperparameter setting do not establish fresh validation, critic robustness, late repair or an analytic injectivity guarantee. Independent sources, broader critics and nuisance/stochastic observations remain open.

## November 18: Plain-language development sequence (no new training)

Added [a plainly labeled training sequence](occupancy-causal-dwell/output/development_plain.png), [extraction code](occupancy-causal-dwell/explain_development.py) and [recorded timing bracket](occupancy-causal-dwell/output/development_plain.json) from existing run-131 dwell histories. With 128 source pairs plus 384 stays per batch, native-like stays show no detected turn at update 200 and a first saved turn at 300; the final turn is near distance 2.519. Spread-out stays show no turn at any saved 100-update checkpoint. The initial run-131 model is checked to have no sampled turn. These snapshots bracket onset in (200,300] without excluding unsaved transient turns or establishing continuous-domain order. Source code and checkpoint identities are preserved; no new fit is reported.

Clarified that the old F/F/F labels describe three separate trained models, ordered means no detected turn, and lowercase f denotes source-pair fraction. Explained stay pairs as identical observations and distinguished encoder fitting from actor training. Development should track onset, turn position and persistence, rather than treating final fold counts as creation rates. Other inspected histories contain temporary reversals; the initial seed-136 encoder already has a sampled turn, which disappears before a later native-path turn develops.

## November 17: Occupancy causality with invariant alignment (completed exploratory comparison)

Added [a fixed-protocol dwell intervention](occupancy-causal-dwell/PROTOCOL.md), [trainer](occupancy-causal-dwell/run.py) and [explanation](occupancy-causal-dwell/README.md) directly under this folder. The earlier OU map changes physical transition geometry along with the marginal. This new experiment retains identical source pairs and adds equal numbers of stay pairs (s,s) in different locations. Both losses consume the same endpoint-output tensor. Added stays have exactly zero alignment value/gradient; their encoded activation is reused at both positions to enforce the analytic cancellation numerically. Only their influence on ordinary SIGReg changes between allocations at a fixed source fraction.

Completed all forty 4,000-update fits: ten historical seeds 130–139, source-pair fractions 0.5/0.25, native-like versus occupied-log-bin-balanced stay allocations. With 50% source pairs, native allocation folds in 5/10 and balanced in 9/10 (five introduced folds, one removed). With 25% source pairs, native folds in 5/10 and balanced in 0/10. All detected final turns lie within the stored source range. Mean measured mass below 2 is 88.48% for native allocation, 57.94% for balanced at source fraction 0.5, and 42.67% for balanced at 0.25. Realized histogram TV to target averages 0.00083–0.00122. Compare allocations within a fixed fraction; cross-fraction changes also change alignment strength.

Source index streams, starting weights, stay count, batch size, optimizer and schedule are paired within each fraction. Independent initial-state audits over all twenty source/fraction cells passed shared output identity, single forward call, exact zero stay output-gradients and identical alignment values/parameter gradients (maximum error zero). All forty final curves/slopes reproduce exactly after reload in float32 and were independently checked on a 20,001-point float64 derivative grid. Sixty same-state next-step forks at updates 100/200/400 have zero alignment-gradient difference and nonzero actual Adam-update contrasts. The combined-gradient contrast is the SIGReg contrast up to float32 roundoff (maximum absolute residual 4.20e-5).

See [causal summary](occupancy-causal-dwell/output/causal_summary.png), [histograms/encoder curves/slopes](occupancy-causal-dwell/output/occupancy_encoders_slopes.png), [same-state forks](occupancy-causal-dwell/output/same_state_forks.png), [full results](occupancy-causal-dwell/output/RESULTS.md), [verification](occupancy-causal-dwell/output/verification.json), and [source provenance](occupancy-causal-dwell/output/provenance.json). Added an [analytic same-histogram/different-order example](occupancy-causal-dwell/output/same_histogram_different_order.png): -cos(pi*U) is ordered and cos(2*pi*U) folded for uniform U, but both share the arcsine output distribution. This is an exact marginal-identifiability counterexample, not a trained neural checkpoint or a Gaussian-target solution.

The controlled stay interventions identify a causal sampled-regularizer-measure effect on folding, rather than an alignment-function change. They do not establish occupancy as the only cause, a monotone concentration rule, newly collected behavior-policy stationarity, or a no-fold guarantee. Population objective versus finite-batch effects, independent policy/initialization cohorts, smaller stay doses closer to the original protocol, and long-budget/batch-size replication remain open. The complementary mathematical fold-development note still needs validated neural-network remainder bounds for certified forecasts.

## November 17: Full LaTeX fold-development paper

Converted the complete derivation into [standalone LaTeX](fold-development/fold-development.tex) with a [compiled PDF](fold-development/output/pdf/fold-development.pdf). The paper retains all 25 numbered mathematical statements, adds native equation cross-references and a formal proposition/proof environment, and typesets the existing birth-event table. The LaTeX source is the editable paper; the Markdown version remains a browsing companion with build instructions. Compilation, reference resolution, PDF text checks, and visual review of every page are performed before publication. Mathematical claims, existing numerical evidence, and pending trained-MLP remainder bounds are unchanged.

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

The initial consolidation merged existing findings without rerunning training. Subsequent dated entries describe new fits and audits.

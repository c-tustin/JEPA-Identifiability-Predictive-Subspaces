# Learning distance from a shrinking triangle

An agent sees a fixed triangle whose apparent size changes inversely with distance:

$$
x(d)=\frac{V}{d}.
$$

It receives six corner coordinates, not distance labels. We test whether its learned number has a straight-line relationship with log-distance,

$$
z=\frac{\log(d/4)}{0.7}.
$$

The scene still has **one changing coordinate: distance**.

**Fresh confirmation is running:** [Fixed tolerances, new actors/seeds and ICM exposure controls](behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/README.md). The protocol and executable hash were published before training. No fresh-cohort conclusion is available until all 60 seed blocks finish.

**Latest mechanism and matched-exposure tests:** [Why late data changes geometry](behavior-policy-experiments/policy-zoo/geometry-and-coverage/README.md). Swapping the observations sent to SIGReg has much larger geometric/task effects than swapping alignment pairs in this test. With broad common exposure, paired TOST supports practical equivalence of recovery and shared-data planning among seven original policies at explicit tolerances; policy-specific-data planning does not pass the same criterion. The study adds 360 encoders, 640 planning evaluations and local checks of the first fold-producing updates.

**Previous fresh-seed test: [early ICM, late-data damage and planning](behavior-policy-experiments/policy-zoo/collection-age-replication/README.md).** On 20 new seeds, switching early-trained ICM encoders to late data creates folds in 15/20 runs; equal extra training on early data creates none. Shared-data goal error is 1.013 versus 0.101. Early ICM does not establish superiority over random/count, and OU retains higher affine recovery. All 19 planned tests use one correction; 360 encoder conditions and 700 planning evaluations are verified.

**Earlier cohort: [original policies, identifiability and the ICM-fold explanation](behavior-policy-experiments/policy-zoo/original-policy-followup/README.md).** Keeping each seed's ICM actor frozen gives 10/20 folded encoders with late collection and 0/20 with early collection. Recovery rises from 0.6035 to 0.9457; shared-data goal error falls from 0.8316 to 0.0516. The study includes all 700 original zoo encoders, 100 alignment-removal controls and 60 collection-age controls, with SIGReg unchanged.

**Earlier controlled test: [same coverage, different movement](behavior-policy-experiments/policy-zoo/matched-coverage/README.md).** Forty encoder fits and 80 planning evaluations keep stationarity, both endpoint distance distributions and overall physical movement fixed. Recovery is nearly identical across the four tested policies; no corrected planning comparison is significant. SIGReg is unchanged.

**Earlier: [ten-trial curves and planning comparison](behavior-policy-experiments/qualitative-study/README.md).** ICM and ICM→OU each fold in 5/10 new trials; OU folds in 0/10. ICM→OU beats OU in only 2/10 held-out planning comparisons, with no significant advantage. Balancing only the SIGReg sampling removes the observed folds in both tested binnings and brings average shared-data goal error close to OU's. The gallery includes coverage versus planning plots and controlled region removals.

[Why the ICM curves bend near distance 2–4](behavior-policy-experiments/qualitative-study/fold-location/README.md): matched input-gain controls move the learned extrema while keeping the data fixed. The encoder's input scale matters; this is not a special physical distance.

[The optimizer mechanism](behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/README.md) traces a reversal that first appears far away and then moves inward. Pure-ICM sampling controls separate SIGReg reweighting from the earlier OU handover.

## Current findings

Five new validation seeds, 8–12; mean shared-range $R^2$ after 4,000 representation updates:

| Training data / intervention | Recovery $R^2$ |
| --- | ---: |
| Fixed OU | 0.9944 |
| Random actions → OU | 0.9960 |
| ICM curiosity → OU | 0.7382 |
| Same ICM transitions, balanced replay → OU | 0.9940 |
| Count-based coverage → OU | 0.9959 |
| Action-balanced exploration → OU | 0.9957 |

**ICM→OU is not reliably best.** Its original three runs looked excellent, but later runs developed folded distance mappings. The two original failed datasets also fail across three independent encoder initializations. A separate loss-term ablation identifies the regularizer batch as the sensitive component: balancing SIGReg alone repairs the observed folds; balancing alignment alone does not. Balanced replay removes the observed folds in both historical failures and all five new validation seeds. It restores roughly OU-level recovery rather than establishing superiority.

A longer, 8,000-update OU control reaches 0.9992 on the same validation seeds. Extra ICM training, the tested entropy/action-noise changes, and an earlier handover do not reliably fix the problem. These are small-world, finite-training results, not universal rankings.

[Concise reliability findings](behavior-policy-experiments/reliability/INTERPRETATION.md) · [What the policy changes mathematically](behavior-policy-experiments/reliability/MECHANISM.md) · [Separate alignment/SIGReg test](behavior-policy-experiments/reliability/LOSS_ABLATION.md)

## Policy zoo

The [original 20-seed comparison](behavior-policy-experiments/policy-zoo/README.md) covers 17 physical behavior policies and Gaussian OU source variants. The [expanded catalog adds 13 controllers](behavior-policy-experiments/policy-zoo/new-policy-results/README.md), with fresh-seed testing and comparisons against OU and the old best policies. Gaussian-looking visits alone are insufficient: pair dynamics, initialization and finite training also matter. These physical-policy extensions are [outside the paper's exact Gaussian-OU assumptions](behavior-policy-experiments/policy-zoo/ASSUMPTION_BOUNDARY.md); their reported gains concern finite-training representation recovery.

## Learned OU and planning

A separate study learns OU correlation and spread without distance-recovery labels. Correlation converges near 0.5; learning spread gives a small improvement. A second controller learns OU-inspired parameters while retaining the physical approach/stay/retreat actions; it is explicitly not an exact OU process. [Learned-OU study](behavior-policy-experiments/learned-ou/README.md).

Planning tests use actual goal-reaching error, with either shared predictor-training data or each policy's own data. This separates representation problems from coverage problems. The planner receives current and goal observations, never true distance. [Latest planning summary](behavior-policy-experiments/planning/SUMMARY.md).

## Nonlinear dynamics

The [action-anchor proof and prototype](behavior-policy-experiments/nonlinear-control/README.md) keep one distance coordinate while allowing nonlinear drift. Known constant physical actuation identifies affine coordinates among informative smooth encoders. The proof includes an approximate bound; it does not claim unconstrained JEPA training satisfies its assumptions.

A [36-model trajectory comparison](behavior-policy-experiments/nonlinear-control/SUMMARY.md) tests the original flexible encoder without paired interventions. Under nonlinear drift, persistent actions with raw SIGReg reach physical-distance $R^2=0.9975$ with no detected grid reversals. Balanced SIGReg performs worse in this world. Coverage gaps and coordinate warping mean this is an empirical result, not a nonlinear identification guarantee.

The [latest extension](behavior-policy-experiments/nonlinear-control/EXTENSION_SUMMARY.md) adds matched-coverage controls and actual nonlinear goal reaching. Raw persistent representations achieve 100% hit rate on the tested local goals with native predictor data; balanced versions perform worse. A new calculation and independent-noise check explain why instantaneous drift identification cannot simply be imposed on a finite-time predictor.

## Try the code

New research is continuing in the cloud: [Predictive Closure directions](behavior-policy-experiments/research-next/PREDICTIVE_CLOSURE.md) and [a same-world test of predictive information versus distance ambiguity](behavior-policy-experiments/research-next/SIGNED_OU_EXTENSION.md). These are derivations and proposed experiments; completed neural results are not claimed yet.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python behavior-policy-experiments/reliability/run_study.py --quick --output /tmp/reliability-smoke
```

The quick command checks execution, not trained performance. Full runs overwrite their own output directories unless a different `--output` is given. Each guide documents training, checkpoint-only reevaluation, and saved settings.

| Where | What it contains |
| --- | --- |
| [Original reproduction](behavior-policy-experiments/identifiability/README.md) | The setting where the Klindt-paper guarantee applies |
| [Curiosity comparison](behavior-policy-experiments/curiosity/README.md) | The source-derived ICM implementation and initial controls |
| [Schedules](behavior-policy-experiments/schedules/README.md) | Entropy decay, handovers and changing OU settings |
| [Curriculum controls](behavior-policy-experiments/curriculum-controls/README.md) | Eight-seed optimizer/exposure controls |
| [Reliability](behavior-policy-experiments/reliability/README.md) | Repairs, alternative policies and crossed seeds |
| [SIGReg guide](behavior-policy-experiments/identifiability/SIGREG.md) | Readable scalar code and the equations it implements |
| [Credits](THIRD_PARTY_NOTICES.md) | Paper/code sources and retained licenses |

Verification: all 58 numerical checks pass. Saved representation/planning evaluations reproduce, including the 12 latest coverage controls and 84 nonlinear action predictors tested in deterministic/noisy execution.

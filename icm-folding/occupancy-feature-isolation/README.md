# Which occupancy feature creates the fold?

Status November 22, 2026: one framework now covers all results. See [the fold framework](FRAMEWORK.md). Earlier: [feature and InfoNCE results](output/skew/RESULTS.md).

The [fixed first-stage protocol](PROTOCOL.md) decomposes the empirical endpoint distribution into mass below distance 2, its conditional near-distance shape, and its conditional far-distance shape. Forty completed hybrid fits independently exchange these components between five preselected folded/nonfolded source pairs. Alignment data, initialization and both sampling streams remain common. This is a sampling intervention using actual source observations, with no new sensor states synthesized.

[Trainer](run.py) preflight checks recover all native empirical distributions exactly, including repeated observations. [Preflight and source profiles](output/preflight.json) are saved. All 40 fits passed independent dense checkpoint reload, component and provenance checks. All 20 mass-only paired changes introduce a material fold, while conditional shape swaps leave material status unchanged. One far-tail swap produces a retained tiny-turn exception. See [complete first-stage results](output/RESULTS.md), [matrix](output/hybrid_components.png), [all contrasts](output/summary.json), and [audit](output/verification.json).

The active goal also requires identifying why the same feature affects InfoNCE and other tested objectives differently. This first decomposition does not answer that part. Follow-up controlled doses, robustness checks and a loss/gradient or collision-cost mechanism test remain required. A good classifier on the ten old sources alone would not establish causality or complete this study.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/occupancy-feature-isolation/run.py --validate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/occupancy-feature-isolation/run.py
```

Run from the workspace root; independent workers can use `--shard K --shards N`. Source-folder links retain their original targets and checkpoint paths. Original sources and other studies are unchanged.


## Mass doses and objective mechanism (running)

The [second-stage protocol](DOSE_PROTOCOL.md) crosses ten near masses, four
conditional-profile templates and three objectives (120 fits): original
separate-input SIGReg, raw/own-code InfoNCE with fixed raw normalization, and a
separate-input scalar variance control. This is a controlled dose study within
each objective, not a matched-compute method ranking. Training is underway;
complete outcomes and independent audits remain pending.

The [mechanism protocol](MECHANISM_PROTOCOL.md) requires an exact expected
finite-batch SIGReg risk comparison among fixed candidate encoders and a
verified InfoNCE collision-cost example, followed by frozen-encoder critic
probes. These checks can falsify a simple objective-preference explanation;
consistent training curves alone are insufficient. Temporal InfoNCE and new
initialization/source validation remain required parts of the active goal.


## Completed dose and mechanism checks; failed generalization retained

The [120-fit dose sweep](output/dose/RESULTS.md) and its [audit](output/dose/verification.json) are complete. SIGReg has a substantial-fold window across the four historical templates; raw/own-code InfoNCE has no final turn in 40 cells. The variance control has many small turns and two just above the material threshold, so it is not universally fold-free.

[Exact fixed-candidate risks and 54 critic probes](output/critic-probe/RESULTS.md) supply an objective-level mechanism: intermediate masses favor the folded Gaussian-matching candidate among the tested alternatives, while the folded encoder has worse raw/own-code discrimination in all 36 paired critic comparisons. This is not a global-optimum or exact branch-selection theorem.

The [120 new initialization/stream fits](output/robustness/RESULTS.md) replicate the core 50% / 75% / 93.75% contrast exactly: SIGReg material folds 0/20,20/20,0/20; raw/own-code InfoNCE 0/20 at each mass.

The [80 new-source controlled fits](output/fresh/encoders/RESULTS.md) are also audited. They falsify a universal mass-only window: at 50% near mass, three new sources still fold materially. All ten fold at 75% and none at 93.75%; no raw/own-code InfoNCE fit has a final turn. The risk bank frozen before source generation predicts only 8/10 native material outcomes, the coarse mass window 9/10. Failures 9105 and 9109 are retained; the bank is not an exact predictor. Source 9109 folds at its native 56.72% near mass, below the historical-template lower boundary.

Required follow-ups are running: [new-source fixed-mass shape swaps](FRESH_SHAPE_PROTOCOL.md), [joint-data temporal InfoNCE/original scalar VICReg comparisons](JOINT_PROTOCOL.md), and [alternative splits/identical support](SPLIT_SUPPORT_PROTOCOL.md). The [exploratory moment plots](output/exploratory_moments.png) are associations, not an isolated moment criterion. The active goal remains incomplete: regional mass is causal, but its shape-dependent boundaries and the relevant conditional-shape feature still need to be isolated correctly.


## Feature isolation and InfoNCE mechanism (November 21, 2026)

[Results](output/skew/RESULTS.md) · [skewness protocol](SKEW_PROTOCOL.md) · [cluster protocol](CLUSTER_PROTOCOL.md) · [locked predictions](output/skew/PREDICTIONS.json) · [figure](output/skew/skew_mechanism.png)

### Mechanism

- The encoder is a scalar function of distance.
- In one dimension, `SIGReg1D` equals 1024 × MMD² to N(0,1) with a unit Gaussian kernel. On a skewed code law, its
  force pulls back codes at standardized 1.1 < z < 3.45 and pushes codes beyond 3.45 outward.
- Bit-exact replays ([trace.py](trace.py)) show the far branch rising and then being pulled back over the near branch
  (the fold) at m = 0.75, and being ejected out of kernel reach at m = 0.9375.
- In the initialization's input geometry, the near cluster is nearly a point. That is why near mass and code asymmetry
  were confounded before.

### Interventions on actual source observations

- 67 pre-registered fits, plus 24 cluster-position fits.
- Removing asymmetry at fixed near mass stops folding.
- Raising asymmetry without a spread tail does not cause folding: 6 of 26 predictions failed and are retained.
- Compact wall clusters fold even at m = 0.9375.
- Three of 31 laws fold or not depending on the initialization.

### Objectives

- Even-only SIGReg still folds.
- A pure skewness target makes both raw InfoNCE and the variance control fold.
- Plain InfoNCE, temporal InfoNCE and the variance hinge set no target shape for the code marginal, and they do not fold.

### Limits

This is a necessary condition plus a mechanism, not an exact one-number criterion, and the work tests one encoder
class. Fit with `python3 skew.py --design` then `python3 skew.py --shard K --shards N`, and the same with `cluster.py`.
Report with `python3 skew_report.py`. [function_space.py](function_space.py) records why an unconstrained function-space
optimum is degenerate: it zigzags with the policy's 0.25 lattice.

## Encoder geometry and the fold framework (November 22, 2026)

[FRAMEWORK.md](FRAMEWORK.md) explains every result with three ingredients:
1. a shape target on the code histogram, which supplies the force;
2. a skewed initial histogram (heavy cluster plus a spread tail within SIGReg's pull-back band), which the force acts on;
3. an encoder whose tangent kernel lets the tail bend more easily than the cluster stretches, which decides the response.

New tests, all pre-registered:
- [Geometry test](GEOMETRY_PROTOCOL.md) ([code](geometry.py), 48 fits). Changing only the input map from `tanh(4x)` to `x`
  or `tanh(x)` removed every fold (0 of 32), despite initial skewness up to 2.7. The skewness rule scored 19 of 38;
  H1 and H2 failed, H3 held.
- [Compact-cluster test](COMPACT_PROTOCOL.md) ([code](compact.py), 32 fits). Falsified "tiny input span is enough":
  4 of 11 predictions correct.

Observed, not pre-registered:
- [race.py](race.py): bit-exact replays. Under `tanh(4x)` the tail's far end overtakes its median (the fold). Under `x`
  the tail moves as a block while the cluster spreads.
- [kernel.py](kernel.py): tangent-kernel decoupling. Under `tanh(4x)` the near cluster is 0.003–0.005 and the tail
  0.14–0.16; under `x` the near cluster is 0.06 and the tail 0.02–0.03.
- Figure: [framework_figure.py](framework_figure.py) → `output/framework.png`.

## Kernel test, force landscape, phase map and explainer (November 22, 2026, later)

- [Kernel-locking test](KERNEL_PROTOCOL.md) ([code](kernel_test.py), 52 fits, pre-registered): 40 of 52 correct.
  - Held: the ordinal prediction (fold fraction 0.50, 0.57, 0.96 across locking bins).
  - Failed: the threshold of 5, and the Spearman prediction (ρ = −0.04).
  - Within each input map, all four initializations agree. Squashing strength k decides; locking at initialization does not.
  - A first-order kernel-velocity statistic ([kernel_race.py](kernel_race.py)) failed beforehand and is recorded in the
    protocol.
- [force_landscape.py](force_landscape.py): SIGReg's per-sample force is the negative slope of (blurred code histogram −
  N(0, 2) density), checked to correlation ≥ 0.999. The pull-back band depends on the law.
- [phase.py](phase.py) (exploratory, 45 cells): end state (stretched, fold, inflated) over input map × near mass. The fold
  threshold in k is 6, 2, 1.5, 3 and 6 at m = 0.5, 0.625, 0.75, 0.875 and 0.9375. Figure: `output/phase.png`.
- [Explainer page](explainer/fold-explainer.html), built by [explainer/build.py](explainer/build.py) from
  [explainer/template.html](explainer/template.html).

## Causal stretch test (November 22, 2026, latest)

[STRETCH_PROTOCOL.md](STRETCH_PROTOCOL.md), [stretch.py](stretch.py) and [stretch_race.py](stretch_race.py); 36 fits,
pre-registered. A zero-initialized extra input channel keeps the initial code bit-identical and adds one way of moving
states.
- **near-only channel:** prevents the fold only at m = 0.875 (4/4). It still folds at m = 0.625 and 0.75 (8/8). The
  prediction failed.
- **far-only and duplicate channels:** fold 12/12 each. These predictions held.
- **Replays:** a faster-stretching spike did not delay the fold (the tail's far end crossed its median at step 300 in
  both runs).
- **Conclusions:** the stretch-race interpretation is falsified at m ≤ 0.75. What the encoder can move does change the
  outcome causally from an identical start (m = 0.875). The revised mechanism (the far end of the tail overtakes its
  middle) is not yet tested causally. See the confidence table in [FRAMEWORK.md](FRAMEWORK.md).

## Rigid-force tests (November 22, 2026, latest)

[RIGID_PROTOCOL.md](RIGID_PROTOCOL.md) (two addenda) and [rigid.py](rigid.py); 52 fits, each condition pre-registered.
The loss value and start are unchanged; only SIGReg's gradient is projected onto block shift and/or rescale of the spike
(d < 2) and the tail (d ≥ 2).

| Condition | Folds | Prediction |
| --- | --- | --- |
| tail rigid | 12/12 | no fold, **failed** |
| spike rigid | 12/12 | fold, held |
| both rigid | 12/12 | fold, held |
| shift only | 0/8, stays skewed | fold, **failed** |
| scale only | 0/8, scale blows up | no fold, held |

Block-level pressure (position and spread of the spike and tail) is enough, and differential pull along the tail is not
needed. How the encoder converts that pressure into a bend is open. See [FRAMEWORK.md](FRAMEWORK.md).

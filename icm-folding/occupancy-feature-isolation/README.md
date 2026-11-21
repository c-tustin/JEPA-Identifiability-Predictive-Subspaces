# Which occupancy feature creates the fold?

Status November 19, 2026: active investigation; no feature has yet been declared isolated.

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

# Why late data bends the code—and whether matched policies differ

**Prospective follow-up completed:** [60 fresh seeds, fixed tolerances and exposure controls](../confirmatory-matched-exposure/output/SUMMARY.md). It confirms the matched-policy equivalence finding and an exposure-only ICM improvement; all fresh results and baseline rollouts are verified.

**Completed:** 360 new encoders, 640 planning evaluations, 54 planned comparisons, and post hoc equivalence and difference tests.

[Zoo recap](ZOO_RECAP.md) · [Full results/graphs](output/RESULTS.md) · [Equivalence test](output/EQUIVALENCE_SUMMARY.md) · [Math](MATH.md)

## Which part of training creates the geometric effect?

Cross early/late transition pairs for **alignment** with early/late observations for **SIGReg**. All four cells start from the same early encoder and Adam state within each seed. SIGReg's formula and weight stay unchanged; its observations are independently sampled from pooled endpoints to isolate their marginal exposure.

| Alignment data | SIGReg observations | Recovery R² | Folds / 20 | Shared-data goal error |
| --- | --- | ---: | ---: | ---: |
| Early | Early | 0.8894 | 3 | 0.1289 |
| Early | Late | 0.4053 | 15 | 1.1332 |
| Late | Early | 0.8905 | 2 | 0.1232 |
| Late | Late | 0.3868 | 15 | 1.0486 |

**The larger change follows SIGReg's observation source.** Early instead of late SIGReg observations improves recovery, folds and reaching in both alignment conditions after the single correction across all 54 tests. Changing alignment alone has much smaller observed effects and no significant planned comparison here; this does not prove alignment is irrelevant.

Independent observation sampling also differs from the original coupled-batch loss. Early observations do not guarantee fold-free training: 2–3 cells fold here, versus none in the earlier coupled early continuation. The planned sampling-protocol comparisons retain that result.

## Why does exposure change the curve?

SIGReg asks sampled outputs to look Gaussian. If most inputs cluster near the object, matching that output distribution can require more code-space variation over a short physical interval. The same weights affect other distances, so nearby fitting pressure can compress or reverse slopes elsewhere.

We checked the actual update: in all **six first-fold events** among the preselected five actors/four cells, the regularizer-moment contribution alone creates the first detected fold; omitting it prevents that update's fold. All 20 detailed fits reproduce final weights/Adam state exactly. This uses the same combined Adam denominator, not separately trained optimizers.

Folding is **not mathematically required** by Gaussianization: a monotone quantile map exists. The reversal is a measured finite-network/optimizer outcome. [Derivation and qualifications](MATH.md).

## Original zoo policies with matched exposure

Seven unchanged original policies receive identical reset-state observations. SIGReg also gets exactly the same observation multiset at each update. Successors, action frequencies and stationarity are not forced to match. We additionally keep original native alignment pairs while fixing only SIGReg's broad exposure.

All 280 zoo-control encoders are unfolded, including those retaining native ICM/RND/disagreement alignment pairs. The broad shared observation pool is a real data intervention, not just reweighting existing native coverage. [Exact matching and precision](PRECISION.md).

## Are matched policies equivalent?

Use **paired TOST** (two one-sided tests), with seed as the independent unit. The null says at least one policy pair differs by the tolerance or more; the alternative says every pair is inside it. Every pair must pass. This tests practical equivalence, unlike merely failing to detect a difference.

For seven original policies and 20 paired seeds, using illustrative tolerances of **±0.001 R²** and **±0.005 physical goal error**:

| Outcome | Global mean-equivalence p | Equivalent within tolerance? |
| --- | ---: | --- |
| Recovery R² | 1.55 × 10⁻³³ | Yes |
| Shared-data goal error | 9.67 × 10⁻¹² | Yes |
| Policy-specific predictor-data goal error | 0.930 | Not established |

The joint claim for recovery **and** shared-data planning passes (p = 9.67 × 10⁻¹²). The earlier four-policy matched-coverage study also passes for those two outcomes (joint p = 4.44 × 10⁻⁷). The goal-error tolerance is 2% of one 0.25 action step. These are exploratory, post hoc tolerances; significance does not decide what difference is scientifically acceptable. Paired-t assumptions apply, and the sign-based check concerns typical differences rather than population means.

This supports practical similarity of linear recovery and shared-data planning in these matched conditions. It does **not** establish equivalence of policy-specific-data planning at the same tolerance. Policy-specific data here means each reset cell's own action transitions, not the original native trajectory pool. All pairs must pass, so the global equivalence claim uses an intersection-union test, not selected significant pairs.

[Concise explanation](output/EQUIVALENCE_SUMMARY.md) · [Full pairwise tests, intervals and tolerance sensitivity](output/MATCHED_EQUIVALENCE.md) · [Complementary difference tests](output/MATCHED_OMNIBUS.md)

## Identifiability follow-up

Whole-range, near-object and far-object linear recovery all pass paired TOST across the seven matched policies at **±0.001 R²**. Whole-range policy means span 0.997994–0.998000. These are empirical recovery diagnostics, not a proof of identifiability up to rotation.

All seven have zero grid-detected folds in 20 runs, but the exact paired test does **not** establish fold-risk equivalence within ±5 percentage points (p = 0.358). Twenty seeds are insufficient for that stricter failure-risk claim. [Focused tests, assumptions and results](output/IDENTIFIABILITY_EQUIVALENCE.md).

The full original native-policy zoo showed substantial observed differences: it has 17 original physical policies, 13 additions (30 total), and 18 Gaussian sources. [What each earlier study showed](ZOO_RECAP.md).

## Reproduce

From the repository root with the existing environment plus SciPy:

```sh
python3 -m unittest discover -s behavior-policy-experiments/policy-zoo/geometry-and-coverage -p 'test_*.py' -v
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/run_geometry.py --workers 6
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/compact_snapshots.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/replay_geometry_birth.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/gradient_regions.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/diagnose_geometry.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/analyze_geometry.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/matched_equivalence.py --require-complete
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/replay_gates.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/verify_geometry.py --require-complete
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/verify_equivalence.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/identifiability_equivalence.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/omnibus_test.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/verify_omnibus.py
```

Actors are frozen, sources are hash-checked, and model/Adam/attribution state resumes every 500 updates. Regenerable sources are ignored by Git. The compaction step removes unused backing storage from saved index views; every saved value is checked unchanged. [Main verification](output/VERIFICATION.json) · [Hypothesis-test verification](output/OMNIBUS_VERIFICATION.json) · [Training/restart gates](output/TRAINING_REPLAY_GATES.json).

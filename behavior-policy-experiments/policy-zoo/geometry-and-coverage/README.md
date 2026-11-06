# Why late data bends the code—and whether matched policies differ

**Completed:** 360 new encoders, 640 planning evaluations, 54 planned comparisons, and requested post hoc mean/rank hypothesis tests.

[Zoo recap](ZOO_RECAP.md) · [Full results/graphs](output/RESULTS.md) · [Requested hypothesis test](output/MATCHED_OMNIBUS.md) · [Math](MATH.md)

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

## Actual hypothesis test for matched-policy performance

Use a **seed-blocked permutation repeated-measures ANOVA** to test policy mean differences, with 100,000 within-seed permutations. Friedman permutation tests check policy ranks. Neither test needs an equivalence margin. Seeds are the independent blocks, not the individual goals.

For the seven original policies with common starting observations/SIGReg exposure (20 paired seeds):

| Outcome | F statistic | Mean-test permutation p | Friedman permutation p |
| --- | ---: | ---: | ---: |
| Recovery R² | 1.660 | 0.118 | 0.183 |
| Shared-data goal error | 0.925 | 0.716 | 0.715 |
| Policy-specific predictor-data goal error | 0.576 | 0.750 | 0.719 |

**No systematic policy-performance difference is detected.** All remain nonsignificant after one Holm correction across the 12 mean/rank tests for both matched studies. The earlier four-policy stationary study also shows no significant differences. This is failure to reject the null of no policy effect, not proof of exact equality. [Full hypotheses, assumptions and statistics](output/MATCHED_OMNIBUS.md).

Observed mean recovery ranges from 0.997994 to 0.998000 across the seven reset conditions; shared-data goal error ranges from 0.0576 to 0.0580. Policy-specific data here means each reset cell's own action transitions, not the original native trajectory pool.

The [optional equivalence sensitivity](output/MATCHED_EQUIVALENCE.md) uses illustrative margins and additional assumptions. Those margins were chosen after observing results and are not the basis of the conclusion above.

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
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/omnibus_test.py
python3 behavior-policy-experiments/policy-zoo/geometry-and-coverage/verify_omnibus.py
```

Actors are frozen, sources are hash-checked, and model/Adam/attribution state resumes every 500 updates. Regenerable sources are ignored by Git. The compaction step removes unused backing storage from saved index views; every saved value is checked unchanged. [Main verification](output/VERIFICATION.json) · [Hypothesis-test verification](output/OMNIBUS_VERIFICATION.json) · [Training/restart gates](output/TRAINING_REPLAY_GATES.json).

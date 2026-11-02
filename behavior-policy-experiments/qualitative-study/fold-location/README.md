# Why do the curves turn or flatten near distance 2–4?

**Latest mechanism check:** [replaying the creation and movement of a fold](training-mechanism/README.md). In seed 131 the first in-range turn appears near distance 15.81, then moves inward to 2.37. Pure-ICM sampling controls isolate the SIGReg measure without an OU handover.

**There is a distance scale built into our network.** Before its learned layers, the encoder transforms the six observed coordinates with `tanh(4*x)`. Because the image is $x=V/d$, the nonzero input magnitudes pass through the tanh transition around distances **2, 3.46 and 4**. These are smooth scale landmarks, not hard thresholds or physical events.

The renderer itself has no special change at those distances. Its observation is still exactly $V/d$. The reference distance 4 used to define the simulator's log-coordinate cancels on rendering; it is distinct from the encoder's input multiplier 4.

## Folding and tapering are different

- **Folding:** the learned slope changes sign. At the original gain, ICM turning points are roughly 1.91–3.14. The input tanh alone preserves order; the learned layers must supply the reversal.
- **Tapering:** the curve keeps its direction but its slope fades. A finite smooth network receiving $1/d$ inputs naturally approaches a constant as distance grows. Training must counteract this tendency over the distance range that matters. This does not require tanh saturation at far distances.

ICM's saved pairs are strongly concentrated near the object: **68–95% of source observations are at distance ≤2**, depending on seed. Far distances therefore constrain the learned curve much less often. With the same architecture and input scale, OU and balanced-SIGReg training produce different curves. Architecture supplies a scale and default shape; the data and objective determine whether training preserves or overcomes it.

## What we tested

We retrained ICM and OU at input gains **2 and 8**, on exactly the same saved pairs as the gain-4 trials. Network weight initialization, minibatch streams, optimizer and 4,000 updates stay matched. The actor is frozen; its original preprocessing stays at gain 4. This isolates the later encoder's gain conditional on the existing trajectories.

There are **40 new trained encoders**, 30 reused original encoders including balanced controls, and 30 untrained initialization profiles. A full gain-4 retraining of folded seed 131 reproduces every parameter exactly. No new planning experiment or independent validation cohort is claimed.

## What happened when we moved the scale

| Encoder input gain | ICM folded runs | Median ICM turning distance | Monotone ICM taper distance | OU folded runs |
| --- | ---: | ---: | ---: | ---: |
| 2 | 3/10 | 2.15 | 2.51 (7 runs) | 0/10 |
| 4, original | 5/10 | 2.37 | 3.26 (5 runs) | 0/10 |
| 8 | 9/10 | 4.78 | 5.51 (1 run) | 0/10 |

Turning medians include only runs with an extremum; taper medians include only nonfolded runs. Their membership changes, so the medians alone are not a paired effect estimate. The graphs show every seed.

The clearest matched example is **seed 131**: turning distances are **2.15 → 2.37 → 4.99** as gain increases. Seed 132 loses its detected fold at gain 2 and turns at **2.19 → 4.02** for gains 4 and 8. Thus 2–4 is movable, rather than a special physical interval. The change is not exactly proportional after retraining, because the learned weights can compensate for the input scale.

The result supports an **interaction of input scale, sampling and training**, not an inevitable tanh fold: OU has no detected folds at any gain. Changing the gain also changes conditioning; the experiment does not prove that a particular activation threshold is the sole cause. Lowering gain is not a complete repair: 3/10 ICM runs still fold, and the remaining curves can taper earlier. No planning advantage is inferred from this location study.

Initialization alone also does not explain the five original ICM folds. Only seeds **136 and 138** already have detected folds before training at gain 4. ICM retains a fold in 136, removes it in 138, and develops folds in four initially nonfolded networks. OU and balanced-SIGReg training remove both initial folds. Initial shape matters, but training can create or remove the problem.

## Inspect the graphs

- [All ten ICM trials, retrained at gains 2/4/8](output/figures/icm_retrained_gain_comparison.png): the most direct test of whether the location moves.
- [All ten OU controls](output/figures/ou_retrained_gain_comparison.png): same architecture and gain changes, different transition distribution.
- [Turning points versus monotone taper locations](output/figures/turning_and_taper_locations.png): a missing extremum is not counted as a shifted fold.
- [Input preprocessing and its sensitivity](output/figures/input_scale.png): the scale present before any training.
- [Original curves against ICM visit density](output/figures/location_and_sampling.png) and [signed derivatives, including before training](output/figures/signed_sensitivity.png).
- [Hidden-unit derivative contributions](output/figures/hidden_derivative_diagnostics.png): opposing contributions can cancel while units remain sensitive. A small cancellation ratio is different from every unit saturating.
- [Seed 131 close-up](output/figures/seed131_turn_zoom.png): all positive and negative contributions near its minimum.
- [Frozen-weight rescaling](output/figures/frozen_weight_rescaling.png): an algebraic check, clearly separate from retraining.

The gray band marks distances 2–4. Every comparison stays in the same one-distance world. Display normalization is affine and cannot move a turning point.

## What the diagnostics mean

Turning points use sign changes of the exact automatic derivative. The earlier “folded” flag separately requires more than 1% reversed grid steps above its numerical threshold; a small extremum need not meet that flag. For curves without a flagged fold, the taper location is the first distance after peak log-distance sensitivity where its magnitude falls to half the peak. This is an explicit diagnostic definition, not a physical boundary; it can lie outside the plotted range.

The derivative contributions sum back to the full derivative. Independent checkpoint audits recreate curves and compare automatic derivatives with finite differences. These checks establish what the saved functions do, not why optimization chose every weight.

For example, at the grid point nearest seed 131's turn, positive last-layer derivative contributions sum to **+1.056** and negative ones to **−1.058**. Their cancellation ratio is about **0.0006**. Neither hidden layer has an activation above the chosen saturation threshold there. The fold is a learned cancellation, rather than every hidden unit becoming saturated; this describes the function, not the history of how its weights were learned.

The same input preprocessing also appears in the original policy and ICM module. Changing their gain and recollecting data is a different experiment. Changing encoder gain also changes its initial function and optimization conditioning, so the present intervention does not isolate saturation alone. These same-seed follow-ups are exploratory, not fresh significance tests.

[Mathematical explanation](MATH.md) · [Fixed protocol](PROTOCOL.md) · [Per-model measurements](output/metrics.csv) · [Saved-model audit](output/verification.json)

From repository root, using the [study dependencies](../requirements.txt):

```bash
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/run_location.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/report_location.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/verify_location.py
```

The run resumes saved stages and checks its recorded settings. Original data and models remain in the parent study's output directory.

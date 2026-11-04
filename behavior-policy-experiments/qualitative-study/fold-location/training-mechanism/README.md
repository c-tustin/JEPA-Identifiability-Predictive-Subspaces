# What creates the fold, and why does its location move?

**The turn is learned and moves during training.** Distances 2, 3.46 and 4 are landmarks of `tanh(4*x)`, not three measured or inevitable fold locations. The input scale shapes the network's response; the sampling-weighted loss changes its weights and can reverse the slope elsewhere.

**Start with [Why does the fold move left?](WHY_LEFTWARD.md)** for the plain-language explanation, curve snapshots, and exact one-update comparisons of alignment versus SIGReg. At update 250, the turn moves from 6.0632 to 5.9207; applying only the SIGReg component gives 5.9210, while alignment alone gives 6.0629.

## We replayed the creation of a fold

For **seed 131, gain 4**, update **204→205** introduces the first detected turn in the evaluated interval, at **distance 15.81**. The raw far-end slope changes from **+0.00415 to −0.00083**. Applying only the SIGReg-moment component of that actual update also creates a turn; applying only the alignment component does not. The replay reaches the saved update-250 weights exactly.

These are one-step parameter interventions at the same state and shared Adam denominator, not separately trained optimizers. They identify the proximate forcing in this example. They do not imply that SIGReg always creates folds or that alignment never matters.

The turn subsequently moves inward:

| Updates completed | Turning distance |
| ---: | ---: |
| 205 | 15.81 |
| 250 | 6.06 |
| 500 | 3.44 |
| 1,000 | 3.07 |
| 2,000 | 2.66 |
| 4,000 | 2.37 |

So the final clustering near 2–4 is **not where the reversal first appears**. The later motion is consistent with the measured optimizer-induced slope changes. At update 250, for example, the local prediction of the change in log turning distance is −0.02436, predominantly from the SIGReg-moment component; the observed change is −0.02379. At update 500, the two components contribute roughly equally.

[Creation and inward migration](output/figures/birth_and_inward_migration.png) · [Training histories at all three gains](output/figures/slope_evolution_seed131.png) · [Separate loss contributions](output/figures/loss_components_seed131_g4_ordinary.png)

The other preselected diagnostic histories are also shown: [seed 130](output/figures/slope_evolution_seed130.png), whose original gain-4 curve is monotone, and [seed 136](output/figures/slope_evolution_seed136.png), which already has a turn at initialization.

## Changing the sampling changes the outcome

We now keep **pure ICM throughout** and balance only the examples sent to SIGReg. Alignment pairs, initialization and optimization stay matched. There is no final OU phase to explain the repair.

| Input gain | Ordinary SIGReg: folded runs | Balanced SIGReg: folded runs | Balanced mean log-distance $R^2$ |
| --- | ---: | ---: | ---: |
| 2 | 3/10 | 0/10 | 0.9954 |
| 4 | 5/10 | 0/10 | 0.9910 |
| 8 | 9/10 | 6/10 | 0.9474 |

This intervention establishes that the **regularizer's sampling measure participates in the failure**. The same Gaussian penalty works differently when applied more evenly across the same stored observations. It does not add new states or change their physical dynamics. Strong input scaling remains problematic: the six gain-8 turns after balancing lie farther out, at approximately **8.78–13.64**. That intervention repairs four gain-8 runs, leaves five folded, and **introduces a fold in previously nonfolded seed 135**. Balancing is therefore a conditional repair, not a universal guarantee.

Every trial is shown: [seeds 130–134](output/figures/pure_icm_sampling_130_134.png), [seeds 135–139](output/figures/pure_icm_sampling_135_139.png). $R^2$ is a log-distance readout score, not a new planning result.

## Why increasing the gain moves the turn

With weights frozen, $h_{g,\theta}(d)=h_{4,\theta}(4d/g)$. Doubling gain therefore doubles the physical location of any corresponding turn. Retraining changes the weights too, so this direct scale effect and weight adaptation add in log-distance.

For seed 131, gain 4→8 predicts **4.736** with weights frozen; retraining gives **4.989**. For gain 4→2, frozen weights predict **1.184**, but adaptation moves the result back to **2.150**. This explains why shifts after retraining need not be proportional. Missing roots are retained as missing, rather than forced into the comparison.

[Scale versus weight-adaptation graph](output/figures/scale_and_weight_adaptation.png) · [Exact decomposition and local root-motion derivation](MATH.md)

## What is verified, and what remains limited

There are 30 new balanced models and nine instrumented ordinary reproductions. All nine reproduce the earlier final weights exactly. Independent checks replay **144 sampled-loss/Adam snapshots** and validate **108 mixed derivatives** numerically. The local slope-change approximation has median relative error **0.17%**; its worst error is **40%** on a nearly cancelled, very small update, and is retained. A separate double-precision root-refinement check tests 36 locally matched simple-root steps; the median absolute root-motion prediction error is about $4\times10^{-6}$ in log-distance. Unmatched or nearly degenerate roots are reported.

This is a causal sampling intervention plus a verified local creation/movement mechanism in the existing model. It is not a global convergence theorem, a claim that every seed has the same history, or a proof that the three input landmarks determine the final root. These are exploratory same-seed controls, with no new significance or planning claims. The actor's input gain remains fixed.

[Model/optimizer audit](output/verification.json) · [Approximation errors](output/local-attribution-accuracy.json) · [Refined-root check](output/refined-root-motion-check.json) · [Full comparison audit](output/completion-audit.json) · [Protocol](PROTOCOL.md)

From repository root with the parent study's dependencies:

```bash
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/run_mechanism.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/replay_birth.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/verify_mechanism.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/verify_root_motion.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/report_mechanism.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/audit_outputs.py
```

Training resumes saved stages, with partial checkpoints every 500 updates. Full gradients and optimizer states are saved at selected snapshots; every 25 updates, the detailed traces save their effects on the function and slope.

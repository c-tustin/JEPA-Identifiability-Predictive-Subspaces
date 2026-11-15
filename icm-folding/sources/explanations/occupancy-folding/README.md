# Occupancy, folding, and the near-object ICM policy

The occupancy explanation is supported, but the remedy needs to change **which distribution SIGReg matches**, rather than just divide its current gradients by occupancy. Across ten stored gain-4 ICM sources, ordinary SIGReg folds in 5/10 runs; inverse-output-gradient scaling folds in 8/10; inverse-occupancy-weighted SIGReg moments fold in 0/10. The mean affine log-distance R² values on a uniform-log grid are 0.612, 0.609 and 0.982, respectively. These are descriptive results for this configuration, not a guarantee for other encoders or input gains.

Open [the graph gallery](output/index.html), [the numerical results](output/summary.json), or [the generated tables](output/RESULTS.md). [Protocol](PROTOCOL.md) specifies all interventions. The 30 native runs and 30 controlled-occupancy runs use 4,000 updates each. No OU data is used.

## What the occupancy explanation means

Let q=log d and let p(q) be the distribution of observations supplied to SIGReg. SIGReg matches the distribution of h(q) to a standard Gaussian. Its empirical characteristic function averages cos(t h(q)) and sin(t h(q)) over observations. An often-visited region therefore has greater total influence than a rarely visited region.

Your intuition that the frequently visited range must receive more representation space is useful. In an ideal **increasing** Gaussianizing map,

    h(q) = Φ⁻¹(F_p(q)),        h′(q) = p(q) / φ(h(q)).

This gives the precise density-dependent slope: a density peak affects how quickly the representation moves through output quantiles. There is no rule that one distance, such as 2, must be repelled from every other distance. The characteristic-function force depends on the current global output distribution and on each output's phase; it is not a pairwise physical-distance repulsion.

Most importantly, the formula above remains ordered for one, two, or three occupancy peaks. **Multimodality does not require folding.** SIGReg also admits unordered functions with similar output histograms. Shared neural parameters, the input transform, alignment, and optimization determine whether training finds an ordered map or bends the far side back. A frequently sampled near region can change the far-distance slope through those shared parameters.

The earlier seed-131 experiment measures that mechanism directly: the turn first enters near d=15.81 at update 205, then moves to 6.06 at 250, 3.44 at 500, and 2.37 at 4,000. SIGReg's part of the update creates the first observed reversal. The density peak is not the location where that fold is born. See [the earlier finite-update interventions](../../../../behavior-policy-experiments/behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/WHY_LEFTWARD.md).

## What was tested with multiple peaks

The new controlled distributions have peaks at 2; at 2 and 8; or at 2, 5 and 12. Peaks have equal mass and log-width 0.14, with a 10% uniform-log background. A uniform-log control is included. The pool covers the same physical interval [1,16] in every condition.

Only the regularizer's endpoint occupancy changes. Each seed retains exactly the same stored ICM alignment pairs, initialization, learning rate, model, loss weight, batch sizes and budget. This isolates the proposed SIGReg mechanism; these are not newly trained bimodal/trimodal actors. Root histories show every detected sign-changing slope root every 25 updates, including roots that disappear. The ordinary controls need not reproduce the native-ICM fold: these synthetic distributions deliberately have different support density.

The figures include curve snapshots and turn-position histories. They do not assume that two peaks must create two folds, that turns must sit at peaks, or that every trajectory moves inward. An explicit monotone Gaussianization figure shows that the target marginals themselves permit ordered solutions.

Final folded runs in the three diagnostic seeds:

| Regularizer occupancy | Ordinary SIGReg | Inverse gradient only | Inverse-weighted moments |
| --- | ---: | ---: | ---: |
| Peak at 2 | 0/3 | 1/3 | 0/3 |
| Peaks at 2 and 8 | 0/3 | 3/3 | 0/3 |
| Peaks at 2, 5 and 12 | 0/3 | 1/3 | 0/3 |
| Uniform log-distance | 0/3 | Not repeated | Not repeated |

Thus neither a peak at 2 nor multiple peaks is sufficient to recreate the native failure in these controlled conditions. The synthetic background and shape differ from the much more boundary-concentrated native data. Ordinary synthetic curves remain nonlinear but ordered. The bimodal inverse-gradient intervention instead develops two late turns in seeds 130 and 131 and one in seed 136; it does not merely undo the native inward fold. This is evidence against treating inverse local gradient scaling as a substitute for correcting the characteristic-function distribution.

## Why the two inverse-occupancy methods differ

For native ICM, estimate endpoint occupancy in 32 equal log-distance bins from the full stored dataset. Assign each endpoint weight proportional to the inverse of its bin's mass, normalized within each batch. For controlled occupancy, use the known masses in 64 log-distance bins. This targets equal mass per log-distance bin, **not uniform physical distance**. No missing bins are synthesized in native data, and uniformizing bin mass does not uniformize the conditional distribution inside each bin.

**Gradient-only:** retain ordinary characteristic-function moments, then multiply each regularizer output gradient by its normalized inverse-occupancy weight. This changes the local response while the force still depends on the original, concentrated output histogram. It need not be the gradient of a scalar objective. Here it removes some old folds but also introduces folds, ending with 8/10 native runs folded. It is not a reliable repair.

**Weighted moments:** compute

    c(t) = Σ_i w_i cos(t h_i) / Σ_i w_i,
    s(t) = Σ_i w_i sin(t h_i) / Σ_i w_i,

then apply the same Gaussian characteristic-function discrepancy and unchanged batch-size multiplier. Autograd differentiates this corrected objective. Both the aggregate moments and each sample's contribution now refer to the corrected sampling measure. This prevents final observed folds in all ten native runs when used from initialization. It does not establish a universal monotonicity guarantee or guarantee a late intervention will undo an existing fold. Earlier continuations already show that late balancing can leave a fold intact.

Finite-batch inverse weighting can have high variance when occupancy is tiny. These experiments use full-dataset counts, no learned density estimator, no clipping, and self-normalization in each batch. The geometry used for binning is available from the known sensor; generalizing this to observations without a known distance coordinate is a separate problem.

## Why the actor remains near the object

The ICM actor and curiosity predictor are separate from the later SIGReg encoder. The actor is trained and frozen **before** these representation experiments. Consequently the SIGReg fold cannot cause the stored actor's near-object occupancy, and repairing the encoder does not change that actor.

The immediate dynamical explanation is measurable. All ten saved actors prefer approach over retreat throughout the far interior: their mean distance change over d=8–14 lies between −0.086 and −0.048 per action. At d=2, approach probability is roughly 0.55–0.72, retreat probability 0.12–0.20, and stay probability 0.14–0.24. They keep moving and repeatedly return toward the reflecting near boundary; they are not literally immobilized at 2. Their stored fractions below distance 4 range from 77% to 98%. The boundary reflects overshoot, rather than clipping motion into a sticky boundary state.

Why was an inward policy learned? ICM rewards prediction error in a learned action-sensitive feature space, not coverage or novelty counts ([original ICM paper](https://proceedings.mlr.press/v70/pathak17a.html)). In this sensor, landmarks scale as V/d. A fixed physical displacement changes raw landmarks approximately as 1/d², making near transitions visually stronger. This supplies a plausible geometric bias, not a proof about the learned feature error.

The new instrumented training histories provide more direct evidence. For all three diagnostic seeds, the near region has larger expected intrinsic reward than d=8–14 at updates 26, 101, 201 and 401. By update 600 the inward drift is much stronger and fresh frozen-policy rollouts concentrate near the object. Seed 131 illustrates the feedback: fraction below 4 rises from 20% initially to 30% at update 401 and 89% at update 600. Current reward is not the entire explanation: its final far-region mean reward actually exceeds the near mean, while its actor still approaches. Policy learning depends on prior rewards, visited transitions, the critic and finite adaptation, so a final reward map is insufficient to infer how the behavior developed.

The training traces reproduce the original final actor and ICM weights **exactly** for seeds 130, 131 and 136. Diagnostics preserve RNG state. Frozen quarter-distance-lattice Markov chains further show that the learned action probabilities alone generate a near-boundary concentration; these lattice comparisons are descriptive approximations to the continuously offset trajectories, not a claim that the collected finite rollouts are stationary. No reward-normalization or environment intervention was run to uniquely isolate the historical geometric cause.

Finally, if “stuck” refers to the **fold's location**, that has a different explanation. The original representation learning rate reaches zero at update 4,000. Earlier continuation moves the same turn from 2.368 to 1.905, so 2.4 is a training endpoint rather than an established equilibrium. See [the continuation experiment](../../../../behavior-policy-experiments/behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/stopping-point/README.md).

## Reproduction and verification

From the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 python3 explanations/occupancy-folding/run_study.py --audit-only
OPENBLAS_NUM_THREADS=1 python3 explanations/occupancy-folding/run_study.py
OPENBLAS_NUM_THREADS=1 python3 explanations/occupancy-folding/policy_diagnostics.py
OPENBLAS_NUM_THREADS=1 python3 explanations/occupancy-folding/report.py
```

Completed run files are reused. Separate `--shard 0 --shards 2` and `--shard 1 --shards 2` processes can run independent jobs. Saved checkpoints retain all curves and slopes, final model and optimizer states, distributions and per-run metrics. The report recomputes every final curve, slope, root list and R² from model weights; native ordinary models must reproduce the earlier weights exactly. A finite-difference gradient check validates weighted SIGReg, and separate checks validate inverse-gradient scaling and equivalence to ordinary SIGReg for uniform weights. Source and input hashes are retained. Root detection is a finite-grid diagnostic, not a proof of global injectivity.

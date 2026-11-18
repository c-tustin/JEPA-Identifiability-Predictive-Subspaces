# Can occupancy cause folding with the alignment function fixed?

November 17, 2026. Completed [fixed-protocol](PROTOCOL.md) follow-up to the [ICM-matched OU experiment](../icm-matched-ou/README.md), with [code](run.py), forty 4,000-update fits and sixty same-state Adam forks.

| Source pairs / added stays | Native-like stay allocation | Spread-out stay allocation |
| --- | ---: | ---: |
| 50% / 50% | 5/10 final folds | 9/10 final folds |
| 25% / 75% | 5/10 final folds | 0/10 final folds |

All detected final turns lie within the stored source distance range. Compare allocations within each row: only those comparisons have the identical alignment function. At 50% source pairs, spreading stays introduces folds in five sources and removes one (136); at 25%, it removes all five native-allocation folds. Mean measured mass below distance 2 is 88.48% for either native allocation and 57.94%/42.67% for balanced allocation at source fractions 0.5/0.25. Mean histogram TV from the specified target is 0.00083–0.00122. The source fraction changes alignment strength, so the two rows are not an occupancy-only dose comparison.

These paired interventions establish a causal effect of the **sampled regularizer measure** on final folding in this setup. They reject a universal rule that spreading occupancy prevents folds. They do not isolate population-objective effects from finite-batch effects, establish the exclusive cause of every original ICM fold, or validate a universal physical-policy mechanism.

See [causal summary](output/causal_summary.png), [histograms/encoders/slopes](output/occupancy_encoders_slopes.png), [same-state forks](output/same_state_forks.png), [same-histogram counterexample](output/same_histogram_different_order.png), [results](output/RESULTS.md), [all metrics](output/summary.json), [input/alignment audit](output/audit.json), [fork diagnostics](output/forks.json), [verification](output/verification.json), and [provenance](output/provenance.json).

All forty saved float32 curves/slopes reproduce exactly after reload; final slope roots were independently recomputed on a 20,001-point float64 grid. All sixty same-state forks have zero alignment-gradient difference and nonzero actual Adam-update contrasts. Combined-gradient differences agree with the SIGReg channel up to float32 roundoff (maximum absolute residual 4.20e-5). The initial twenty source/fraction audits also report zero alignment value and gradient differences.

The OU comparison recreated folds but changed occupancy and physical transition geometry together. This experiment isolates a training-measure effect while keeping **both losses on the same inputs**.

Every batch retains the same S original ICM pairs and adds the same number H of stay pairs (s,s). Only where the stays occur changes: native ICM endpoint occupancy versus equal mass across occupied log-distance bins. All source observations, pair streams, initial weights, model, optimizer, total batch size, stay count and schedule are matched. Stays use actual source observations; no missing support is synthesized.

For every encoder h in this pointwise deterministic model class:

`alignment = [sum_source (h(x)-h(xp))² + sum_stays (h(s)-h(s))²] / (S+H)`

`          = sum_source (h(x)-h(xp))² / (S+H)`.

Thus changing stay placement leaves the **entire alignment function**, not just its initial value, identical. Each stay is encoded once and its activation reused at both endpoint positions, ensuring exact numeric cancellation. Both losses receive the same full output tensor; SIGReg still sees the repeated stay endpoints and their changed occupancy. Ordinary SIGReg is not reweighted.

For a fixed realized batch, let p be its endpoint empirical measure and A its source-only sum divided by the total pair count. The loss is exactly `0.9*A(theta) + 0.1*R_p(theta)`. Comparing stay allocations at the same parameters gives `L_1-L_0 = 0.1*(R_p1-R_p0)` and the corresponding parameter-gradient identity. This statement does not require independent endpoints or an approximation using histogram bin centers. At different trained parameters, alignment gradients can differ as a downstream consequence; that does not change the fact that the alignment *function* is the same.

The forty-fit matrix covers all ten historical source seeds 130–139, source-pair fractions 0.5 and 0.25, and both allocations. Compare allocations within a fixed fraction; changing the fraction changes alignment strength. Same-state next-step forks at native-path updates 100,200,400 share weights and Adam state to check the update channel explicitly. The input/alignment audit passed all twenty source/fraction cells before training.

These are replay interventions, not freshly collected OU/actor trajectories. This is a test of whether the sampled measure supplied to SIGReg can causally change folding under a fixed alignment objective. It cannot establish that occupancy is the only cause, or that one near-object mass determines folding universally. Effects can include finite-batch gradient noise. The hard guarantees here concern alignment cancellation; fold checks remain empirical.

The complementary [fold-development framework](../fold-development/README.md) derives sufficient opposite-slope conditions for an actual optimizer step, including its finite-step remainder. A rigorous certificate for the trained neural network still needs validated numerical/remainder bounds. That is a different question from identifying an occupancy intervention's causal effect: a histogram by itself does not specify alignment pairs, initial weights, response kernels or Adam state.

There is also an exact obstruction to deciding order from marginal distributions alone. For a uniform percentile U in [0,1], `h_ordered(U)=-cos(pi*U)` is increasing, whereas `h_folded(U)=cos(2*pi*U)` turns at U=0.5. Both outputs have exactly the same arcsine CDF `1-arccos(y)/pi` on [-1,1]. Any population loss depending only on the output marginal, including this characteristic-function discrepancy, assigns them the same value. This example does not achieve an exactly Gaussian target or claim either function is a saved trained MLP; it proves that such a marginal loss cannot distinguish their order. A proposed universal folding criterion needs additional information about the encoder, pairs and optimizer.

From the workspace root, using the [existing dependencies](../shared-occupancy-ou/requirements.txt):

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-causal-dwell/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-causal-dwell/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-causal-dwell/report.py
```

New code lives here and imports canonical earlier studies without copying checkpoints. Plans are reproducible from fixed seeds with retained hashes; checkpoints retain initial/final weights, geometry histories, fork diagnostics, source/protocol/code hashes and expected/actual exposure histograms. Final reports independently reload curves/slopes and check float64 derivatives on 20,001 points. Fresh independent sources, batch-size replication and long-budget stability remain future work.

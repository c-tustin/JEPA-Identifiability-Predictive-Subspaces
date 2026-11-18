# Occupancy intervention with an invariant alignment objective

November 17, 2026. Exploratory protocol fixed before these fits. Earlier OU and concentration results motivated this design; these are diagnostic, not fresh confirmatory sources.

## Identification

The ICM-matched OU experiment changes occupancy and pair geometry together. Here fix a batch of S original ICM pairs and append H stay pairs (s,s), with B=S+H=512. Evaluate the unchanged pointwise gain-4 scalar MLP once on all current and successor observations. Both alignment and ordinary SIGReg consume those exact encoded endpoints. There are no importance weights, detached endpoints or separate per-loss datasets.

For every deterministic pointwise encoder, each stay contributes (h(s)-h(s))²=0 and zero alignment gradient. Each added stay is encoded once and its activation reused at both endpoint positions in the single common loss-input tensor, making this cancellation exact in floating point too. Both losses see that entire tensor. Thus the alignment function for this entire batch equals (1/B) times the sum of squared differences over the S fixed original pairs, independent of where the H stays are placed. Moving the stays changes the endpoint empirical measure supplied to SIGReg but leaves the alignment objective identically unchanged. This is an analytic cancellation for the stated encoder/loss; it is not a no-fold theorem. Batch normalization, dropout or cross-example operations would require a new audit.

Compare two stay allocations: `native` samples the pooled original endpoint measure; `balanced` gives equal probability to each occupied one of 32 equal log-distance bins, preserving the original conditional observations within each bin. Both use exactly the same source support, S source pairs, H added stays, encoder initialization, source index stream, common stay quantiles, optimizer, schedule, regularizer coefficient, batch size and update count. Empty bins are not synthesized. Different source seeds can have different support. The intervention changes the replay measure; these are not freshly collected OU or actor trajectories.

The marginal is p_total=f*p_source+(1-f)*p_stay in the population, where f=S/B and p_source is the original pooled endpoint distribution. Run f=0.5 and 0.25, comparing allocations only within each f. Changing f changes alignment strength and endpoint duplication correlations and is not an occupancy-only contrast. Original source pairs can already contain stay actions: f denotes the fraction of retained source pairs, not the fraction of physically moving transitions.

## Fixed matrix and training

All ten historical sources 130–139, both source fractions, both stay allocations: 40 fits. Same gain-4 MLP and seed-specific initial weights, 4,000 scheduled Adam updates, batch 512, and loss 0.9*alignment+0.1*ordinary SIGReg. No adaptive coefficient selection, source exclusions or post-outcome extensions. Source pairs are sampled uniformly with replacement. Stay allocations use common uniform random numbers transformed through the respective discrete endpoint CDFs. Both endpoint halves include the identical stay observations.

Record expected and realized training endpoint histograms, plan seeds/hashes, source/code/protocol hashes, initial/final weights and geometry every 100 updates. Final folds are derivative sign changes on a 20,001-point float64 log-distance grid. Report in-range roots separately, output scale and normalized backward variation. A zero sampled count is not a continuous injectivity guarantee. Four thousand updates need not be convergence.

## Same-state mechanism check

On the `native` training path at updates 100,200,400, fork the *next* step into both stay allocations using the exact same encoder, Adam moments, scheduled rate, original source pairs and common hold quantiles. At fixed parameters compare alignment values/gradients, regularizer gradients, and combined gradients. Check that the combined-gradient difference equals 0.1 times the regularizer-gradient difference to numerical tolerance. Apply one actual Adam update per fork with identical before-state; retain slope changes at distances 2,4,8,16 and parameter-update differences. These local interventions identify the channel of an update difference; they do not by themselves explain every later turn.

Audit zero stay alignment gradients and one forward call with shared output identity. Independently reload final checkpoints, reproduce stored float32 curves/slopes and regenerate all plans and realized histograms. Verify alignment invariance at independent initial-model/batch probes for every source/fraction.

## Interpretation

If allocations change final folding with the alignment function and all specified controls fixed, this establishes a causal effect of the regularizer's sampled occupancy measure on folding in this training setting. The effect can include finite-batch moment/gradient-noise changes, not only the population SIGReg objective. It does not prove occupancy is the only cause, supply a universal concentration threshold, identify physical behavior occupancy independently of all dynamics, or establish arbitrary-architecture generalization. Shared weights, input gain and Adam history can mediate the effect. Fresh actor/initialization cohorts, alternative batch sizes and long-budget stability remain subsequent tests rather than claims of this protocol.

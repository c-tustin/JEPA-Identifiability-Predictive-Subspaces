# Saved data × initialization × independent batch sampling

Fixed November 19, 2026 before training or inspecting the new crossed fits. This is a follow-up to the historical dataset/initialization crossing, whose dataset rows retained different historical batch plans.

## Fixed design

- Ten unchanged saved datasets, seeds 130–139; ten original encoder initializations, labels 130–139; five fresh batch-plan labels 0–4. Fully crossed: 500 new fits. All initializations are included, rather than selecting those with favorable previous results.
- Each plan uses independent torch.Generator streams: alignment seed 3100000 + plan label; regularizer seed 3200000 + plan label. Draw with replacement from all 16,384 stored pair rows, shape (4000,512). The exact same two index tensors are used across every dataset and initialization at a given plan label. Their generation has no dependency on model construction or dataset seed. Draw indices for alignment and regularizer separately, as in the original qualitative study.
- Initial weights use torch.manual_seed(40000 + initialization label) and the unchanged 6→48→48→1 scalar tanh encoder, input tanh(4*x). Hold weights identical across every dataset/plan for a given initialization.
- Original 4,000 updates, batch 512, Adam, half-constant/half-cosine learning rate from 0.001, objective 0.9 alignment + 0.1 ordinary SIGReg. No new actors, collection, reweighting, geometry in the gradient, decoder, or tuning. Training sees only x and xp.
- Require all datasets to have identical configuration, pair count and layout. Stored rows are time-major with 256 episode indices at each of 64 retained times. Shared indices give matched episode/time draws, not matched physical observations. They do not make data from different policies identical.

## Validation and saved artifacts

Before batch launch, check all source shapes/configurations, regeneration of all ten original initializations, distinct hashes for the five new plan pairs, and plan hashes independent of data/init. Use the new update loop to replay original full diagonal fits for seeds 130 and 131 (ordered and folded examples); require exact final parameter equality to saved original checkpoints. Check interruption/resume equality on a short separate validation fit before running the matrix.

Save source, code/dependency/protocol, plan and initial-weight hashes, initial/final weights, final optimizer state, resumable partials every 500 updates, and geometry histories every 100 updates on a 1,025-point grid. Final checks independently reload weights and use exact autograd derivatives on a 20,001-point float64 log-distance grid over [1,16]. Record roots, backward/total variation, output std and affine log-distance R². Any sampled derivative sign reversal counts as a detected fold; material folds additionally require backward/total variation > 0.001. Distinguish initial folds from final folds. Spatial/temporal grids are empirical diagnostics, not injectivity guarantees.

## Analysis fixed before outcomes

Report the full 10×10×5 cell table, fold/material counts by dataset, initialization and plan, and the number of dataset/initialization pairs whose final binary status changes across the five plans. Report within-dataset/init disagreements between all ten pairs of plan labels. Report between-dataset disagreements at matched init/plan. For context, compute the balanced three-way descriptive sum-of-squares decomposition of binary final-fold outcomes (main effects and all interactions); these are descriptive fractions of this finite matrix, not independent-source significance tests or population causal shares. Include the analogous material-fold decomposition. Five plan draws provide a small sampling-noise assessment, not proof of robustness to every possible stream.

This isolates encoder-training minibatch randomness from the fixed saved-dataset factor conditional on the tested model, initialization bank and training protocol. A dataset remains a bundle of the learned policy and the particular trajectories collected from it. Recollection from the same frozen actors and new policy-training seeds are separate experiments. The 500 fits are not 500 independent policies. Old crossing results remain a separate historical comparison; the fresh matrix does not mix in old plans or count replay validation fits as additional cells.

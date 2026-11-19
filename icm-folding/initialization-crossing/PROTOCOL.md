# Dataset × initialization crossing

Fixed November 19, 2026 before inspecting crossed training outcomes.

## Question and design

Cross the ten stored ICM policy datasets (130–139) with the ten original scalar-encoder initializations (130–139): 100 fits. Each cell starts from initialization and uses the original gain-4 scalar 6→48→48→1 tanh MLP, ordinary SIGReg, 4,000 updates, batch 512, Adam and historical learning-rate schedule. Objective: 0.9 alignment + 0.1 SIGReg. Use all original stored pairs, matching the original qualitative study rather than the later gradient study's trajectory split. No new policies, rollouts, labels, repairs, or hyperparameter tuning.

Within each dataset row, hold both alignment and separate regularizer index streams exactly fixed across all initializations. Across each initialization column, initial weights are identical. Use the historical alignment stream associated with each dataset; generate regularizer indices using seed 1900000 + data seed. Initialize using seed 40000 + initialization seed. The diagonal must reproduce the ten original final encoders bit-for-bit. Historical diagonal streams were originally generated following the matching initial model construction; they are now frozen independently of the initialization used in each crossed fit.

Run diagonal replay checks first. Save source, initial-weight, plan, code and protocol hashes, initial/final parameters, final Adam state, and resumable partials every 500 updates. No geometry labels enter training; only x and xp are used.

## Evaluation and interpretation

At initialization and every 100 updates, evaluate exact autograd slopes on a 1,025-point log-distance grid over [1,16]. Independently reload final weights and evaluate 20,001 float64 grid points. A detected fold is a sampled derivative sign change; material folds additionally require backward variation / total variation > 0.001. Retain roots, scale, affine log-distance R², and history. Temporal and spatial grids can miss reversals; no analytic injectivity guarantee is claimed. Initial turns are distinguished from newly developed turns.

Report the complete binary matrix, fold frequencies by dataset and initialization, and paired contrasts. These ten reused historical sources and initializations are diagnostic; 100 cells are not 100 independent policy sources. A dataset row bundles the learned actor, collection randomness, and its fixed minibatch streams. Row differences cannot uniquely identify occupancy, actor-training randomness, collection randomness, or minibatch randomness. Interactions may prevent an additive explanation. No downstream decoder or planning experiment is included in this first phase.

A subsequent phase should recollect several datasets from each fixed saved actor and cross them with common initializations; separately varying batch plans would isolate that component. Neither phase is implied complete by launching this crossing.

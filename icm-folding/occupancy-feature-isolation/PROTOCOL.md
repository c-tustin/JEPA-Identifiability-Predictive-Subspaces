# Isolating the occupancy feature

Stage 1 fixed November 19, 2026 before hybrid training outcomes.

The previous source swap identified an occupancy pathway, not a distribution
feature. Decompose each empirical endpoint distribution at the previously used
physical distance 2 into (1) mass below 2, (2) the conditional distribution
below 2, and (3) the conditional distribution at or above 2. Cross these three
components independently between each of five fixed source pairs:
(130,131), (134,132), (135,133), (137,136), (138,139). The first member had no
original final fold; the second had one. This is an exploratory mechanism
decomposition on reused sources, not independent confirmation.

All eight component combinations per pair: 40 fits. Use common encoder
initialization 131, original gain-4 MLP, 4,000 updates, batch 512, original
Adam and learning-rate schedule. Hold alignment data and indices fixed to
historical source 131. Loss is 0.9 alignment + 0.1 SIGReg. Draw 1,024 iid
regularizer endpoints from each hybrid by inverse conditional empirical CDF,
using one shared float64 uniform plan (seed 8400131). Preserve actual source
sensor vectors and their multiplicities; do not smooth bins or add states.
No physical labels enter model inputs or loss; known distance is used only to
construct the controlled sampling intervention and evaluate geometry.

Audit component normalization, exact native-corner empirical recovery,
conditional/mass invariance across swaps, input identities and realized
sampling errors. Save hashes of source files, initial state, uniform/index
plans, code, protocol and imported dependencies. Save full models/Adam and
100-update geometry histories. Reload final encoders and evaluate 20,001
float64 log-distance points, reporting any turn, material backward variation
>0.001, and turns inside actual hybrid observed support. Record initial folds.

Report every cell and component-swap contrast, including interactions. Neither
the arbitrary split at 2 nor a fitted feature is assumed to be the answer.
This stage can reject coarse explanations or localize a conditional component;
it cannot establish an exact universal feature. Follow-up intervention doses,
alternative split points, common support, independent initializations/plans
and new sources are required before declaring a feature isolated.

The InfoNCE question remains a separate required phase. Apply the identified
feature intervention to the same-data objectives and test a mechanistic
prediction about their different gradients or collision costs. An absence of
sampled folds in old InfoNCE fits alone is insufficient. Do not mark the overall
goal complete based solely on this first decomposition.

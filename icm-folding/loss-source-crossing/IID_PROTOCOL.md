# Independent-endpoint robustness check

Fixed November 19, 2026 before these outcomes are inspected. Use all ten
historical SIGReg source datasets with alignment data fixed to 131 and
initialization fixed to 131. Use the same original seed-131 alignment index
plan, original 4,000-update training configuration and unchanged loss.
Replace paired endpoint sampling in SIGReg with 1,024 independent uniform
draws with replacement from the full empirical endpoint multiset. This
preserves the exact empirical marginal in expectation, including source
multiplicities; it removes within-pair dependence. Use three independent
regularizer plans, generator seeds 2900131/2900132/2900133, each shared across
all ten sources: 30 fits. Save identity hashes, model, optimizer and history.
Reload and evaluate final 20,001-point float64 geometry.

The three plans test finite-batch robustness of the endpoint-distribution
effect at one alignment source and initialization. They are not new policies,
fresh sources, population-objective training, matched realized histograms, or
a pure single-draw dependence-only contrast. Draw noise and dependence change
together relative to the main crossing. A final fold under iid sampling
disproves that paired-endpoint dependence is necessary for that fit.

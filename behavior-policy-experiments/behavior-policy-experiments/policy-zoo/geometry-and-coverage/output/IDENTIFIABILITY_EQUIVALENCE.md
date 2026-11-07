# Identifiability: equivalence of recovery, and uncertainty about folds

**Recovery is practically equivalent across matched policies at the stated tolerance. Fold-risk equivalence must be evaluated separately from recovery-score equivalence.**

Recovery R² was already included in the previous TOST analysis. This follow-up adds regional recovery checks and a paired exact test for fold risk. It reuses completed runs; no new training is claimed.

Exploratory recovery tolerance: **±0.001 R²**. Every policy pair must pass paired TOST at alpha 0.05. Seeds are independent units; distances and goals are not. Paired-t assumptions apply. These tolerances were set after the experiment, not prespecified.

| Study | Recovery measure | Range of policy means | Global TOST p | Equivalent? |
| --- | --- | ---: | ---: | --- |
| Four surrogates, 10 seeds | Stationary-weighted log-distance recovery | 0.994836–0.994839 | 1.18e-21 | Yes |
| Four surrogates, 10 seeds | Uniform log-distance grid recovery | 0.995355–0.995358 | 7.58e-22 | Yes |
| Seven original policies, 20 seeds | Whole range, uniform log-distance grid | 0.997994–0.998000 | 1.55e-33 | Yes |
| Seven original policies, 20 seeds | Near object: d < 4 | 0.993957–0.993977 | 1.43e-34 | Yes |
| Seven original policies, 20 seeds | Far from object: d ≥ 9 | 0.998025–0.998029 | 5.82e-47 | Yes |

For the seven original policies, the conjunction of all three recovery measures passes: **p = 1.55e-33**. Each endpoint requires all 21 policy pairs to pass. This intersection-union claim needs no additional pair-count correction; individual intervals remain pointwise, not simultaneous.

## What about folded representations?

All seven matched policies have **0 grid-detected folds in 20 runs**. Identical observations do not establish identical population fold probabilities. The grid check uses 1,025 log-spaced distances and its existing numerical threshold; it cannot rule out every between-grid fold.

For each policy pair, bound the absolute fold-probability difference by the probability that their paired runs disagree about folding. An exact binomial test asks whether that disagreement probability is below the tolerance. This is a conservative sufficient test of fold-risk equivalence.

At **±5 percentage points** of fold probability, the seven-policy global p is **0.358**: equivalence is **not established**. With zero disagreements in 20 seeds, the pointwise one-sided 95% upper bound is **13.9%**. The earlier four-policy study has only 10 seeds and is also inconclusive at the same margin.

If we fix this 5-percentage-point tolerance and collect a fresh cohort, this conservative rule needs at least **59 paired seeds with zero disagreements** to pass. That is a best-case requirement, not a general power calculation or a reason to stop sampling once significance appears.

## What this does—and does not—say

- Similar R² scores mean similar empirical linear recovery of the chosen log-distance coordinate. They do not prove the functions themselves are identical, or that either is exactly identifiable up to rotation.
- In one dimension an orthogonal rotation is only a sign. R² allows fitting scale and offset as well; it is an empirical recovery diagnostic, not a direct test of the paper’s full theorem.
- Both policies could perform equally poorly, so the table includes absolute recovery levels. Here these levels are high but below one.
- The seven-policy study matches starting observations and SIGReg exposure, not successor distributions or stationarity. No conclusion extends to the original native-visitation zoo.

[Full pairwise tests and exact fold-risk calculations](IDENTIFIABILITY_EQUIVALENCE.json) · [Earlier recovery/planning equivalence](EQUIVALENCE_SUMMARY.md)

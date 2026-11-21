# Stage 2: mass dose × conditional shape × objective

Fixed November 19, 2026 before this stage's outcomes. This exploratory follow-up
is motivated by the first completed hybrid cells: mass swaps change substantial
fold status, while a long-tail swap at high near mass has already produced one
tiny turn. Do not call near mass a complete feature on that evidence.

Use all four near-shape/far-shape combinations from sources 130 and 136. Hold
each conditional empirical profile fixed and vary near mass over
0.25, 0.5, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.90625, 0.9375, 0.96875.
All observations are actual stored vectors. Use common initialization 131,
4,000 original-schedule Adam updates and a common float64 uniform endpoint
plan (seed 8500131). The source-131 alignment-pair plan remains fixed.

Three objectives, 120 fits total:

1. Original separate-input SIGReg: 0.9 fixed alignment + 0.1 SIGReg on 1,024
   hybrid endpoints, as in stage 1.
2. Raw/own-code InfoNCE using the existing raw-embedding-infonce objective and
   1→48→48→6 critic, 256 hybrid endpoints per update (first 256 of the common
   1,024 draws), with no alignment term. Fix raw normalization to full source
   131 endpoint statistics across all distributions, and fix critic
   initialization to seed 6100131. This is an occupancy dose experiment within
   that objective, not a matched-compute objective ranking. In particular it
   changes sampling count and replaces temporal alignment with own/raw pairs.
3. Separate-input scalar VICReg-style variance control: 25 fixed alignment +
   25 times the mean variance hinge of the first/second 512 hybrid endpoints,
   sqrt(var_unbiased + 1e-4), target std 1. This adapts the earlier scalar
   VICReg formula to fixed alignment data. It is labeled as this control,
   not represented as a replay of the earlier joint-data VICReg recipe.

No model receives distance labels. Audit distributions, fixed initializers,
plans and source identities. Save full states, optimizer, 100-update geometry
and sampled losses; reload every final model for 20,001-point float64 geometry.
Retain any turns, material turns, support-restricted turns and output scale.
Report all cells, including collapse, unexpected folds and boundary turns.

This stage tests a mass/shape interaction and whether it differs by objective.
It cannot establish a universal one-number occupancy criterion, eliminate
architecture/optimizer dependence, or prove why a loss avoids every possible
fold. Robustness to new initializations/plans/sources and an explicit
collision-cost or gradient mechanism remain required. Temporal InfoNCE is
not silently substituted for raw/own-code InfoNCE; its separate test remains
pending.

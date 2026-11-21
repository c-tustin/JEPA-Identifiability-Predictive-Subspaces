# Temporal InfoNCE and original joint-data loss controls

Fixed November 19, 2026 before these encoder outcomes. Use all ten new sources
9100–9109, initialization 131, critic initialization 6100131, original 4,000
updates and learning-rate schedule. The full frozen-source native joint law
is tested, plus endpoint near masses 0.5,0.75,0.9375: 160 intended fits across
SIGReg, temporal distance-score InfoNCE, raw/own-code InfoNCE and scalar VICReg.

Group actual source transition rows into both endpoints below 2 (NN), one
endpoint below 2 (cross), and neither below 2 (FF). Keep conditional transition
distributions within all three groups fixed and keep cross probability gamma
equal to native. At desired endpoint near mass alpha use probabilities
(alpha-gamma/2, gamma, 1-alpha-gamma/2). This gives exactly alpha endpoint near
mass in expectation and exactly recovers the native empirical joint law at
its original alpha. Audit all feasibility constraints and report unavailable
cases instead of changing sources. Draw real pairs via one shared float64
uniform plan (seed 9000131), 512 pairs/update.

Both losses use exactly one shared encoder-output tensor for each sampled
pair batch. SIGReg, temporal InfoNCE and VICReg use the unchanged earlier
mlp-gradient-study.base_loss formulas; raw/own-code InfoNCE uses the existing
quadratic raw critic, first 128 endpoints from each view and fixed source-131
raw normalization. The temporal loss uses its original symmetric 128-candidate
scalar distance score and batch-code standardization. These formulas are
explicitly different objectives and candidate counts, not matched-compute
rankings. No privileged distance enters their losses or model inputs.

This joint-data experiment necessarily changes transition sampling as alpha
changes. Conditional endpoint shapes also vary because cross transitions
contribute a fixed absolute mass. Report the analytic conditional-mixture
change bounds gamma/2*abs(1/alpha-1/alpha_native) for near and the analogous
1/(1-alpha) bound for far. Do not claim this is the same pure-marginal
intervention as the earlier endpoint-only study. Its role is a joint-data
bridge for temporal InfoNCE and original scalar VICReg, combined with the
already isolated endpoint-mass results.

Preserve every native/intervention cell, exact source and plan identities,
full weights/Adam/critic states and 100-update geometry. Independently reload
all final geometry on 20,001 float64 points, retaining tiny turns and support
diagnostics. New-world generality and analytic injectivity remain unproved.

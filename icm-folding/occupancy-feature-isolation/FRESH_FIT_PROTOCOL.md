# Implementing the predeclared new-source controlled fits

Fixed November 19, 2026 after all ten new sources/predictions were saved and
before fitting any new-source encoder. FRESH_PROTOCOL.md and BANK.json remain
unchanged. Freeze all ten prediction-file hashes in PREDICTIONS_LOCK.json.

Use dose.fit unchanged with fresh source conditional profiles, common encoder
131 and critic 6100131, original fixed source-131 alignment plan, fixed raw
normalization to source 131, and a new common uniform plan seeded 8900131.
For each source use its native near mass and 0.5,0.75,0.9375, SIGReg and raw/
own-code InfoNCE: 80 intended distinct fits. Keep any coincident native/preset
mass explicitly identified, without double-counting an identical fit.

Predictions from the frozen risk bank concern material folds for SIGReg at
native mass. Report any-turn accuracy separately, all margins, all incorrect
predictions, and the coarse 0.6–0.91 mass-window rule as a separate comparator.
Report mass intervention outcomes without replacing sources or refitting the
predictor. The wrapper/protocol/prediction hashes are governing scope metadata;
the unchanged dose trainer also records its original implementation dependencies.

Independent reloads must regenerate source/plan/initialization identities and
all final dense geometries. This confirms a causal feature only within this
same renderer/model/optimizer setting; it does not establish a universal
mass threshold or replace temporal InfoNCE/joint-data testing.

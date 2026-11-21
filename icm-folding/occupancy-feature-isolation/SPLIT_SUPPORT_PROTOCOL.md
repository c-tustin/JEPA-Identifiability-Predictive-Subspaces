# Split-location and identical-support checks

Fixed November 19, 2026 before these outcomes. The first-stage protocol
required checking that the feature was not an artifact of splitting at 2 or
of different source supports. Keep all previous stages unchanged.

Split check: pairs (130,131) and (137,136), thresholds 1.5,2,2.5,3, all eight
mass/near/far combinations: 64 fits. Recompute empirical components and mass
at each split, use one new common uniform plan 9200131, common initializer 131
and fixed source-131 alignment. Reuse dose.fit, explicitly recording the
governing split protocol and threshold in scope sidecars. Its implementation
receives conditional arrays and mixture mass; its original protocol remains
a dependency, not a claim that these components are split at 2.

Identical-support check: restrict all ten original endpoint sets to distances
<=10, form the union pooled actual observations, and keep only the 32 log bins
occupied in every source. All target distributions use this identical finite
pool with strictly positive weights. Within each bin use the same pooled
conditional distribution. Each source supplies only its conditional near/far
bin masses on this common set; retain its original full-source near mass.
Repeat all eight component combinations for all five original pairs: 40 fits.
This changes the conditional profiles and discretizes their density shape;
it is not called an exact native replay. It supplies a strict common-support
control without generating new sensor vectors. Recompute and audit every
point's positive probability and the exact target regional mass.

Both checks use original 4,000-step SIGReg training and shared initialization,
alignment plan, optimizer and uniform stream. Save target/index/source hashes
and full states, and independently reload every final dense geometry. Report
all tiny/material turns and departures from the original mass-factor rule.
A split at 2 is not assumed to be a privileged physical threshold. These
tests do not establish a distribution-only guarantee for arbitrary models.

# Remove temporal alignment

Fixed November 19, 2026 before these outcomes are inspected. For all ten
historical endpoint sources, train the same scalar MLP from common
initialization 131 with only 0.1 SIGReg, setting the alignment term exactly to
zero. Use the main crossing's common regularizer pair-index plan 1900131,
unchanged batch size, Adam, 4,000 updates and rate schedule. Compare with
the main crossing's alignment-source-131 row. Save checkpoints and provenance;
independently reload final 20,001-point float64 geometry.

This tests whether temporal alignment is necessary for folding in this fixed
setup. It does not isolate all regularizer/optimizer interactions or show
alignment never changes geometry. It is a one-initialization diagnostic using
reused sources, not a method ranking or a new independent policy cohort.

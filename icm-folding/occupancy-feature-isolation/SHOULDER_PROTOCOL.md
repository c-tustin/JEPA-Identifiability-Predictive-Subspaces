# Resolve the within-[2,3) feature

Fixed November 20, 2026 before these outcomes. The eight within-interval
patches show that source 9109's conditional shape within [2,3), rather than
within [3,6) or [6,16], switches material folding in the remaining fixed-50%
case. This corrects the earlier speculative deep-tail account.

Keep source-9104 near profile, 50% near mass, and all coarse far interval
probabilities fixed to source 9104. Keep [3,6) and [6,16] profiles at 9104.
Split [2,3) at 2.5. Independently swap its conditional CDF(2.5), lower-half
profile and upper-half profile between 9104/9109: eight fits. Reuse exact
uniform plan 8900131, initialization/alignment and training helper. The two
native-shoulder corners must replay deep-shape 000 and 100 exactly. Native
CDF shortcuts are explicitly used to avoid purely numerical partition drift.

Retain every case. If within-half shape still changes the outcome, do not
claim CDF(2.5) has isolated the feature. This is a local exploratory follow-up,
not a prevalidated universal physical threshold or predictor.

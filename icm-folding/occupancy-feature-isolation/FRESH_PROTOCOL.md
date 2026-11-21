# New-source prediction and intervention validation

Fixed November 19, 2026 before generating or inspecting these sources. Train
ten new ICM policy sources with original configuration, seeds 9100–9109, and
original collection seeds 10000 + source seed. No outcome-driven replacement.
Save policy, ICM, histories, parameter change, observations/actions and config.

Freeze all twelve reference SIGReg encoders from the four stage-2 templates
at near masses 0.5,0.75,0.9375. Record their hashes in BANK.json before new
source generation. No bank selection or refitting on new source outcomes.
For each new empirical endpoint distribution, calculate exact expected iid
1,024-endpoint SIGReg risk plus original source-131 alignment risk for every
bank encoder. Record min folded risk minus min ordered risk and its predicted
sign before fitting any new encoder. Also record the coarse near-mass window
prediction 0.6 < mass below 2 < 0.91; that rough window is deliberately tested
as a provisional rule, not treated as an exact theorem. Keep all disagreements.

Required validation fits, specified now but implemented/run separately:
common initialization 131 and shared sampling plans; original source-131
alignment held fixed; SIGReg and raw/own-code InfoNCE on each source's native
endpoint measure and the same conditional profiles reweighted to near masses
0.5,0.75,0.9375. There are 80 planned controlled fits. Exact native-mixture
recovery, full provenance and independent dense final reloads are required.
If either conditional component is absent, retain that native result and
report the unavailable intervention; do not silently replace the source.

Temporal InfoNCE and original joint-data VICReg must also be tested rather
than inferred from raw/own-code results. A joint-data intervention protocol
and its unavoidable changes to transition sampling must be made explicit
before that experiment; those tests are not declared covered here.

The frozen risk-bank comparison is an operational distribution functional,
not proof that the bank contains a global minimum or that every optimizer
selects its best candidate. New source validation tests that claim directly.
All sources share the same toy renderer; other worlds and continuous-domain
injectivity are outside the evidence supplied by this cohort.

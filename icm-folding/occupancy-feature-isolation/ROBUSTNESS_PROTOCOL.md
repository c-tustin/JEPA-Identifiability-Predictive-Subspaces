# Independent initialization and sampling replication

Fixed November 19, 2026 before these outcomes. Replicate the core mass-window
contrast at 0.5,0.75,0.9375 with both diagonal conditional templates (130,130)
and (136,136), encoder initializations 130–134 inclusive, and two new common
uniform plans seeded 8800131/8800132. All five initializations are included in
this contiguous bank, without choosing by prior success. Compare SIGReg and
raw/own-code InfoNCE as specified in DOSE_PROTOCOL.md. Total: 120 new fits.

Reuse dose.fit unchanged, passing each explicit encoder state and uniform
plan and writing into separate init/plan subdirectories. Initialize critics
with 6100000 + encoder initialization label; common within each initialization.
Raw normalization and source-131 alignment data/index plan remain fixed.
The new governing protocol, wrapper hash, initialization/critic hashes and
plan hashes are recorded in preflight and per-cell scope sidecars. The older
dose protocol remains a dependency of the unchanged fit routine, not a claim
that these use its original single initializer or uniform stream.

Require independent final reload and identity verification. Report every fit,
any turn, material turn, support and scale, including failures of the window
prediction. These are new training randomness conditions on reused data, not
new actors or a universal sufficient-statistic proof. New source validation,
boundary refinement, temporal InfoNCE and mechanism convergence remain open.

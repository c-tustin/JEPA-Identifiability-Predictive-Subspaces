# Objective mechanism: exact risks and fixed-encoder critic checks

The finite-iid-batch SIGReg risk formula agrees exactly with an enumerated
two-atom/two-draw example, and with direct moment calculations for all twelve
fixed candidates at three additional mixture masses (maximum error
8.53e-14). Each candidate's risk is quadratic in near mass.

Comparing the candidates trained at masses 0.5,0.75,0.9375 yields crossings
between the low-mass ordered and middle-mass folded candidate at
0.5834–0.5977, and between that folded candidate and the high-mass ordered
candidate at 0.8825–0.9110 across four conditional profiles. This supplies an
objective-level reason for an intermediate mass window: the relative
near/far weights change the relative Gaussian-matching costs. All five old
substantial-fold masses lie inside every one of these candidate windows;
the five original nonfolded masses lie above every window.

This comparison is restricted to three fitted candidates per profile. It is
not a proof of global optimality or exact training-branch prediction. In
particular the dose study has an ordered 87.5% case despite the three-candidate
ranking preferring the folded candidate there; optimizer path and additional
candidate solutions matter near boundaries.

For raw/own-code InfoNCE, identical codes give identical critic score rows.
With two distinct positive labels the average loss is at least log 2. The
checked two-point example has loss 0.693147 at collision, 0.598139 after a
small correctly associated separation, and 0.798139 with the reversed
association. SIGReg assigns exactly the same loss to the two permutations.
This is a structural collision-cost example, not an injectivity certificate
for finite real-data fits.

All 54 frozen-encoder critic probes passed independent checkpoint reload and
validation-loss reconstruction. At two profiles, three target masses and three
critic initializations, the middle-mass folded encoder has higher InfoNCE
loss than both ordered candidates in **36/36** paired comparisons.
The excess ranges from 0.3938 to 1.1299 loss units.
Codes are affinely normalized using the target distribution to control
critic input conditioning; raw normalization, critic class, update budget,
initial states and candidate plans are paired. The maximum absolute change
between the last and third-last validation checkpoints is
0.0047; these finite optimizations do
not prove the critics reach their global optima.

Together these tests support a specific mechanism: intermediate near/tail
mass can favor a folded Gaussian-matching solution in the tested model,
whereas raw/own-code discrimination penalizes its ambiguous association with
observations and can retain a non-Gaussian ordered code. They do not establish
that InfoNCE can never fold, that occupancy alone determines every optimizer
path, or that temporal InfoNCE has already been tested by this raw/code probe.
New initializer/plan replication and new-source predictions remain pending.

[Exact-risk plots](../mechanism_risks.png) · [Exact-risk values](../mechanism.json) ·
[Critic comparisons](critic_comparison.png) · [All critic measurements](summary.json) ·
[Audit](verification.json) · [Protocol](../../MECHANISM_PROTOCOL.md)

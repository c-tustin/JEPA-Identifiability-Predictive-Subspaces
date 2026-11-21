# Stage 3: test an objective-level explanation

Fixed November 19, 2026 before the following comparisons. Stage-2 low/middle
doses already show strong SIGReg folds at intermediate mass and no raw/own-code
InfoNCE folds. That is not yet an explanation of their difference.

For each of the four stage-2 conditional-profile templates, compare the three
SIGReg encoders trained at near mass 0.5, 0.75 and 0.9375. Evaluate all three
fixed encoders under every mixture mass, with alignment fixed to source 131.
Derive the exact expected finite-iid-batch SIGReg objective from near/far
characteristic functions. For N endpoints and real target g,

    E[N |mean exp(i*w*y)-g|^2]
       = (N-1)|phi|^2 - 2N*g*Re(phi) + N*g^2 + 1.

Here phi = alpha*phi_near + (1-alpha)*phi_far, so each fixed encoder's risk
is exactly quadratic in alpha. Report all pairwise risk crossings, including
cases where they fail to explain the trained branch. A comparison among three
trained candidates does not establish a global optimum. Check this formula
against exact enumeration of a small discrete example, and retain conditional
output ranges, moments and code overlap diagnostics.

For the raw/own-code InfoNCE mechanism, verify the exact two-input collision
bound: identical codes yield identical categorical score rows, so with two
distinct positive labels their mean loss is at least log 2. Verify that a
separation of codes in the correct association lowers the existing quadratic
critic loss in a concrete two-point example, while code permutations leave
SIGReg exactly unchanged. For two point-mass input components, the probability
of a cross-component two-candidate batch is 2*alpha*(1-alpha); report the
resulting excess-collision cost under the ideal distinguishing comparator.
This is a deliberately analytic example, not a claim that finite samples of
the real encoders collide exactly or that InfoNCE guarantees injectivity.

A further frozen-encoder critic test is planned separately: use the same three
SIGReg candidates at each of near masses 0.5,0.75,0.9375, templates (130,130)
and (136,136), three fixed critic initializations and 4,000 critic updates.
Normalize fixed codes by their target-distribution mean/std to control input
conditioning; raw normalization remains fixed to source 131. Use the existing
critic class and 256 candidates, identical plans across encoder candidates.
Evaluate on separately drawn candidate batches. This is a restricted-critic
optimization comparison, not a proof of Bayes-optimal discrimination. Include
all seeds and losses; do not infer general InfoNCE behavior from a failed
critic fit. Temporal InfoNCE still requires its own joint-data experiment.

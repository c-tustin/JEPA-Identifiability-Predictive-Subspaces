# Interpretation of the frozen protocol

The protocol phrase “necessity of transition coupling” should be read narrowly as **necessity of the explicit squared-difference alignment term**. It does not mean the intervention removes every effect of the joint pair law.

Even without alignment, endpoints of a sampled transition co-occur in the same finite SIGReg minibatch. For any fixed batch, permuting its endpoints leaves the regularizer unchanged. Across randomly sampled batches, however, the variance of its empirical characteristic function can depend on endpoint correlations. The existing [finite-pair loss calculation](../mechanism-tests/MATH.md) makes this distinction explicit.

Therefore unchanged folds after alignment removal establish that the explicit alignment term was unnecessary for those intervention fits. They do **not** establish that the marginal distribution alone is sufficient, that transition laws never matter, or that SIGReg uniquely causes the folds. The numerical intervention and prescribed tests are unchanged; this clarification limits interpretation rather than changing the frozen protocol.

Earlier collection is a complementary test using the same behavior actor. It changes occupancy, boundary encounters, state/action frequencies and the empirical pair law together. Any repair demonstrates sensitivity to collection phase; it does not isolate all these changes. Isolating residual finite-batch pair effects would require independently sampling pooled endpoints in a separate, prospectively specified control.

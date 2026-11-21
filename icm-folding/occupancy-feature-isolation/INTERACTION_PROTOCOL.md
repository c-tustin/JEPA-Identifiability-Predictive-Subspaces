# Does the discrimination gradient actively oppose the SIGReg fold?

Fixed November 20, 2026 before outcomes. Use sources 131,136,9109, the joint
near-mass-0.75 construction, initializations 131/132, common uniform pair plan
9300131, matching critic seeds 6100000 + initialization, and original 4,000
updates/schedule. Shared raw normalization is source 131. One encoder forward
supplies all losses on each batch.

Train 0.9 alignment + 0.1 SIGReg + lambda raw/own-code InfoNCE for lambda
0,0.1,1,10,100. Add a lambda-100 control with codes detached only in InfoNCE,
so the critic is trained but contributes no encoder gradient. Total 36 fits.
No outcome-driven lambda selection or omission. Raw/code scoring uses the
same first 128 endpoints from each view, as in the preceding joint study.

The detached control's final encoder must exactly reproduce lambda 0 at the
same source/init/plan. If attached InfoNCE changes folding while detached
InfoNCE does not, this isolates an effect of its encoder gradient rather than
extra parameters/compute. It does not prove every contrastive gradient is
anti-folding or that InfoNCE guarantees order. If no tested weight prevents
folding, retain that result rather than extending the grid adaptively.

Save full states, critics, Adam, sampled losses and geometry. Independently
reload final 20,001-point float64 geometry. Evaluate final expected SIGReg
risk under the actual controlled joint pair law, including finite-pair-batch
variance, to determine whether any prevented folds trade off Gaussian matching
or instead improve both objectives. Do not assume the outcome beforehand.

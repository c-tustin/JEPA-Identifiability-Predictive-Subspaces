# Direct temporal InfoNCE gradient control

Fixed November 20, 2026 before these outcomes. Repeat the attached/detached
intervention with the original temporal distance-score InfoNCE formula rather
than infer its mechanism from raw/own-code InfoNCE. Same three sources,
initializers 131/132, near mass 0.75, joint law, uniform plan 9300131 and
4,000-step schedule. Use lambda 0,1,10,100 plus detached lambda 100: 30 fits.

The unchanged earlier base_loss computes its symmetric 128-candidate score
using code standardization over all 1,024 endpoints. Add it to 0.9 alignment
+ 0.1 SIGReg. Detach only its code inputs in the control. There is no learned
critic for this temporal score. Require each detached final encoder to equal
the matching lambda-zero encoder exactly. Native lambda-zero fits also check
against the preceding raw-interaction lambda-zero baseline at identical inputs.

Retain all failures and costs, independently reload geometry, and evaluate
the exact expected original objective under the controlled joint law. A
successful encoder-gradient intervention identifies a mechanism in this
setup, not a universal guarantee or a matched-compute method ranking.

# Alignment source × SIGReg source

Fixed November 19, 2026 before running this experiment.

Cross all ten historical alignment-pair datasets (130–139) with all ten
historical SIGReg endpoint datasets (130–139), using one common initialization
(131), one common alignment index plan (historical 131), and one common
regularizer index plan (1900131). All datasets have the same number of pairs.
Use the original gain-4 scalar MLP, 4,000 updates, batch 512, original Adam
and learning-rate schedule, and loss 0.9 alignment + 0.1 SIGReg. The (131,131)
cell must reproduce historical encoder 131 bit-for-bit before other cells run.
No physical labels enter either loss. Historical training already used
separate alignment and regularizer batches; this experiment changes their
source independently. It is a diagnostic intervention, not a new actor.

Save initial/final weights, optimizer state, data/plan/code/protocol hashes,
and 100-update geometry histories. Independently reload every final checkpoint
and evaluate autograd slopes on 20,001 float64 log-distance points in [1,16].
Report both any derivative sign change and material backward variation >0.001.
Report turns restricted to each source's observed support separately.

Row changes isolate alignment dataset with fixed regularizer data, initial
weights and index plans; column changes isolate regularizer dataset with
fixed alignment data and those same controls. A column includes endpoint
distribution and within-pair endpoint dependence, not just a coarse occupancy
histogram. Reused datasets are ten sources, not 100 independent replications.
One initialization and one batch plan cannot establish universal attribution.
Finite grids do not prove injectivity. Final fold status does not measure all
transient folds or uniquely explain actor learning.

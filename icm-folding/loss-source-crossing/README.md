# What causes folding in these ICM-source runs?

**The endpoint occupancy distribution supplied to SIGReg causally controls the final folding split in this fixed setup.** The unconstrained scalar MLP can satisfy a distribution-matching pressure while reversing physical order. Accumulated SIGReg momentum triggers several inspected fold-creation steps, but is not necessary for every turn.

This concerns the scalar representation trained after the ICM actor is frozen. The actor's earlier 32-dimensional curiosity features are a separate network. No actor is retrained in this experiment.

## Controlled evidence, completed November 19, 2026

All 100 alignment-source × regularizer-source fits, 30 iid-endpoint fits, and ten alignment-removal fits completed. All 140 final checkpoints pass independent dense float64 reload and identity checks; 130 crossing/IID fits also pass derivative finite differences. The common (131,131) control reproduces the historical final encoder bit-for-bit. All sources use the same renderer and configuration; sensor reconstruction errors are at most 1.2e−7.

| SIGReg source datasets | Folds across ten alignment sources, per dataset | IID endpoint plans, per dataset | Alignment removed |
| --- | ---: | ---: | --- |
| 131, 132, 133, 136, 139 | 10/10 | 3/3 | Folded |
| 130, 134, 135, 137, 138 | 0/10 | 0/3 | No detected turn |

Changing only the alignment source flips **zero of 100** final binary outcomes. Changing the regularizer source switches between the folded and unfolded groups with the alignment source, initialization, both index plans, optimizer and schedule fixed. The main crossing contains 50 material final folds; all their turns lie inside both associated sources' observed support.

Independent endpoint draws retain each empirical endpoint distribution in expectation while removing within-pair dependence. All three plans preserve the same source split, so paired endpoints are not necessary for these folds. Removing alignment entirely also preserves the split, demonstrating that temporal alignment is not necessary in this setup. These controls identify the regularizer's data pathway more precisely than the earlier bundled dataset/initialization crossing.

[Complete results](output/RESULTS.md), [full matrix](output/source_crossing.png), [audited measurements](output/audited-summary.json), [verification](output/verification.json).

## How this produces a fold

SIGReg compares global sine/cosine averages of scalar outputs to Gaussian targets. Those averages depend on the output distribution and are invariant to permuting which physical states receive which codes. They supply no physical-order constraint. Shared MLP parameters transmit these occupancy-dependent gradients across the whole distance range; a parameter update can take a previously positive local derivative through zero.

The [occupancy/slope integral](../occupancy-integral/README.md) evaluates that force through the encoder's actual parameter sensitivity. At seed 131's update 205, the current regularizer gradient opposes the reversal, while accumulated regularizer momentum drives it. An actual optimizer-state intervention removes that history while keeping the same before-weights, current gradients and second moments: the post-step slope at distance 16 changes from −0.000832708 to +0.005567166 and the turn disappears.

The [five-source replication](../occupancy-integral/output/BIRTH_RESULTS.md) retains all eight detected birth events through update 500. Removing regularizer history prevents six; removing alignment history prevents none. Two events still turn without first-moment history, including a first-update transient. Momentum is therefore a demonstrated local trigger, not a universal prerequisite or permanent-prevention rule.

## What is established and what remains open

The causal conclusion is conditional on these ten saved sources, the original gain-4 MLP, initialization 131 for the source interventions, Adam, and the fixed budget. The complementary [100-cell initialization study](../initialization-crossing/README.md) shows initialization sensitivity on some other dataset conditions. Full occupancy shape, model response, current gradients and optimizer history all matter; a single concentration threshold or number of modes does not explain the prior interventions.

The exact density feature that predicts branch choice across arbitrary new sources is not identified here. New actor training and recollection, other renderers, other optimizers and an all-input injectivity guarantee remain separate questions. Finite derivative grids can miss narrow turns. Crossed fits reuse ten sources and must not be counted as 140 independent policy replications. These results establish a causal mechanism for the inspected runs, without claiming that every possible ICM run must fold for the same reason.

## Reproduce

The [main protocol](PROTOCOL.md), [iid protocol](IID_PROTOCOL.md), and [alignment-removal protocol](ABLATION_PROTOCOL.md) were fixed before their respective outcomes. Run from the workspace or portable repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/loss-source-crossing/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/loss-source-crossing/iid_endpoints.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/loss-source-crossing/alignment_ablation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/loss-source-crossing/report.py
```

Original sources remain in the existing linked qualitative study. Completed cells resume without altering their identity; derived model checkpoints remain local and are inventoried in the publication manifest.

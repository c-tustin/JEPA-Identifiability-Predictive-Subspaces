# Optimizer-history interventions across five historical failures

All five original 4,000-update paths replay bit-for-bit. Scanning each update through 500 detects 8 ordered-to-folded events, all passing independent dense before/after geometry gates. All 32 fork checkpoints reproduce saved geometry exactly after reload on 20,001 float64 points.

| Source | Update | Original turn distance | Remove SIGReg history | Remove alignment history | Remove all history |
| --- | ---: | ---: | --- | --- | --- |
| 131 | 205 | 15.8112 | No turn | Turn | No turn |
| 132 | 363 | 15.3150 | No turn | Turn | No turn |
| 132 | 368 | 15.8394 | Turn | Turn | Turn |
| 133 | 446 | 15.4268 | No turn | Turn | No turn |
| 136 | 346 | 15.3227 | No turn | Turn | No turn |
| 136 | 354 | 15.2257 | No turn | Turn | No turn |
| 139 | 1 | 2.2463 | Turn | Turn | Turn |
| 139 | 376 | 15.6966 | No turn | Turn | No turn |

Removing accumulated SIGReg first moment prevents 6/8 inspected birth steps; removing alignment first moment prevents 0/8. Every one of the five sources has at least one prevented birth. Current gradients, second moments, step count and learning rate are identical within each event. These are actual finite Adam steps, rather than only gradient or linearized slope attributions.

Two counterexamples limit the momentum claim. Seed 139 develops a transient turn on its first update, when no momentum history exists. Seed 132 at update 368 still turns when all first-moment history is removed. Therefore accumulated momentum is a causal trigger at several inspected events, but is not necessary for every turn. A fork starts from the historically trained before-state; it does not erase the influence of momentum on that earlier path or on second moments.

The initial seed-136 model already has a turn; that is not counted as a birth. Multiple births in one run reflect intervening disappearance, so eight events are not eight independent sources. The seven later turns first enter near the far boundary; five of those seven begin beyond their source’s observed range. Both seed-136 events begin inside its wider stored support. All five original final folds lie inside observed support. These intervention checks establish local causes, not permanent prevention. Finite temporal inspection stops at 500, finite spatial grids can miss narrow turns, and selected historical failures do not establish generality.

[Protocol](../BIRTH_PROTOCOL.md) · [Trainer](../birth_replication.py) · [Measurements](birth_replication.json) · [Figure](birth_replication.png)

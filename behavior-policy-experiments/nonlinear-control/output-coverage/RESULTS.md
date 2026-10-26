# Matched source-state and action coverage

Simulator-reset controls, not trajectory policies. Same source observations/actions/noise across worlds; only drift differs. Paired seeds 23–25, unchanged encoder and fitting budget. [Protocol](../EXTENSION_PROTOCOL.md).

| Drift | SIGReg | Runs | Distance $R^2$ | Log-distance $R^2$ | Worst reversal (% range) |
| --- | --- | ---: | ---: | ---: | ---: |
| Zero | raw | 3 | 0.9774 | 0.9323 | 0.0000 |
| Zero | balanced | 3 | 0.8602 | 0.9913 | 0.0000 |
| Nonlinear | raw | 3 | 0.9758 | 0.9403 | 0.0000 |
| Nonlinear | balanced | 3 | 0.8381 | 0.9857 | 0.0000 |

The source distribution is matched; the endpoint mixture seen by SIGReg can still differ because successors depend on dynamics. Reset queries change both state and action coverage relative to trajectories. Complete per-seed metrics, query hashes and checkpoints are retained.

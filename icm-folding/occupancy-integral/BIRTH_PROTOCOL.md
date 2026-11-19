# Replicate the local optimizer intervention

Fixed November 19, 2026 before inspecting replicated events. Use all five
historically final-folded original runs: 131,132,133,136,139. Replay their
unchanged 4,000-update paths with separate alignment/SIGReg first-moment
histories. Require exact original final weights. Scan every update through
500 on a 1,025-point derivative grid; retain every ordered-to-folded
transition, independently checking before/after on 20,001 float64 points.
Initial folds are recorded separately, not counted as created folds.

At each verified event fork the actual before-state four ways: unchanged;
remove accumulated SIGReg first moment; remove accumulated alignment first
moment; remove all accumulated first moment. Keep current gradients, second
moments, step counter, batch and learning rate fixed. Independently evaluate
actual post-step dense geometry. This tests first-moment necessity for that
particular event, not permanent prevention, counterfactual second-moment
history, all future births, or universal necessity of Adam. Report failed
verification gates and all detected events without outcome-driven exclusion.
Historical selected failure cases are diagnostic replications, not fresh
source validation. Finite grids may miss narrow turns; later births beyond
500 are outside this scan.

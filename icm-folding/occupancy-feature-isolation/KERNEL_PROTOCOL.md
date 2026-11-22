# Prospective test: does tangent-kernel locking of the near cluster predict the fold?

Fixed November 22, 2026, before any fit below was trained.

## Background

In the [framework](FRAMEWORK.md), the fold happens when SIGReg's force can bend the tail of the code curve faster than
it can stretch the heavy near cluster. Both speeds are set by the encoder's tangent kernel. The locking ratio is

  L = decoupling(d 4 vs 16) / decoupling(d 1 vs 1.9),

with decoupling(a, b) = 1 − Θ(a,b)/√(Θ(a,a)Θ(b,b)) at initialization ([kernel.py](kernel.py)).

Calibration data known before this protocol:

| Input map | Initializations | L | Fold at near mass 0.625–0.875? |
| --- | --- | ---: | --- |
| `tanh(4x)` | 131, 133 | 30, 58 | yes, all fits |
| `x`, `tanh(x)` | 131, 133 | 0.4–0.8 | no fits |

The threshold is set at the geometric midpoint of 0.8 and 30, so **L* = 5**.

## Negative exploratory result, recorded before this test

A first-order statistic failed. It takes the instantaneous velocity of the code under SIGReg's force filtered through the
initial kernel (gradient descent or Adam's sign step) and compares the rate at which the curve closes on itself with the
rate at which skewness falls ([kernel_race.py](kernel_race.py)). On 418 audited fits it made 73–133 errors, against 44
for near mass alone. The fold develops nonlinearly over about 300 steps, so the initial velocity does not settle it.

## Design (52 fits)

| Setting | Values |
| --- | --- |
| Input maps | `tanh(kx)` with k = 1.5, 2, 2.5, 3, 6, 8 (new), and `tanh(4x)` |
| Initializations | 131, 133, 135, 137 (`tanh(4x)`: only 135 and 137, which are new) |
| Laws | the geometry-test laws m0.75 and m0.875 (uniform within d < 2 and within d ≥ 2) |
| Training | identical to the [geometry test](GEOMETRY_PROTOCOL.md) |

L for every (map, initialization) pair is computed by `kernel_test.py --design` and stored in the locked predictions.

## Predictions

1. Per cell: junction fold (backward fraction > 0.1) if L > 5, no junction fold if L < 5.
2. Ordinal: the fold fraction does not decrease across the L bins < 5, 5–20 and > 20.
3. Spearman correlation between log L and backward fraction is positive across the 52 fits.

All cells are reported, and failures are kept. One encoder architecture; an empirical check, not an analytic guarantee.

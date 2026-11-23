# Causal test: with the starting histogram held fixed, does letting the spike stretch remove the fold?

Fixed November 22, 2026, before any fit below was trained.

## Why this test

The [framework](FRAMEWORK.md) claims that the fold happens because the encoder can bend the tail of its code curve
faster than it can stretch the heavy near-wall spike. Every earlier comparison changed the input map. That changes both
how easily the spike stretches and the initial code (and therefore the starting histogram), so the two were confounded.

## Design (36 fits)

The baseline `tanh(4x)` encoder gets one extra input channel c(x), computed from u = 1/d (the sensor's second
coordinate). The first-layer weights for the new channel are initialized to zero. All other weights are copied from the
usual initializer. The initial code is therefore identical to the baseline at every distance (checked to be exactly equal
before training), so the starting histogram is identical.

| Condition | Extra channel c | Nonzero for | What it adds |
| --- | --- | --- | --- |
| near-only | max(0, 2u − 1) | d < 2 only | lets the near spike stretch; nothing at far states |
| far-only | max(0, 1 − 2u) | d > 2 only | lets the tail bend more; nothing at near states |
| duplicate | tanh(4u) | all states | parameter-count control with the same geometry as the baseline |

| Setting | Values |
| --- | --- |
| Laws | geometry-test m0.625, m0.75, m0.875 (uniform within d < 2 and within d ≥ 2) |
| Initializations | 131, 133, 135, 137 |
| Training | identical to the [geometry test](GEOMETRY_PROTOCOL.md): 0.9 alignment + 0.1 SIGReg, 4,000 Adam updates, plan 9600131 |

Baseline `tanh(4x)` fits on these laws already fold at every initialization tested (backward 0.31–0.34).

## Predictions (junction fold means backward fraction > 0.1)

1. near-only: no fold in at least 10 of 12 cells.
2. far-only: fold in at least 10 of 12 cells.
3. duplicate: fold in at least 10 of 12 cells.

If (1) fails while (2) and (3) hold, stretchability of the spike is not the cause, and the starting shape is doing the
work. If (3) fails, adding capacity alone changes the outcome and the test is uninformative.

We also report the tangent-kernel decoupling of the near cluster (d 1 vs 1.9) and the tail (d 4 vs 16) at initialization
for each condition.

All cells are reported, and failures are kept. One architecture; an empirical check, not an analytic guarantee.

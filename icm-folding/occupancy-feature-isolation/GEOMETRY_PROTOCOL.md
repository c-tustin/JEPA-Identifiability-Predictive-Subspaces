# Does the fold follow the encoder's input geometry rather than physical distance?

Fixed November 22, 2026, before any of the fits below were trained.

## Framework under test

The [framework](FRAMEWORK.md) says the fold is caused by an asymmetric code law at initialization: a heavy cluster that
the encoder cannot spread, next to a spread tail. SIGReg then pulls the tail back over the cluster. Which states form the
unspreadable cluster is set by the encoder's input map, not by physical distance.

The baseline encoder feeds `tanh(4x)` to its MLP, with x ∝ 1/d. That squashes distance 1–2 into 0.035 of input range,
against 0.72 for distance 2–16. If the input map is changed to `x` (linear) or `tanh(x)`, the near and far regions get
similar input spans (0.08 vs 0.07 in f0 at initialization 131), and the asymmetry of each occupancy changes:

| near mass m | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 0.9375 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| γ, tanh(4x), init 131 | −0.54 | −0.24 | 0.10 | 0.48 | 0.92 | 1.53 | 2.64 | 3.95 |
| γ, x, init 131 | −1.61 | −0.93 | −0.42 | 0.02 | 0.46 | 0.98 | 1.71 | 2.21 |

The sign of γ is taken with the code oriented to increase with distance. Negative γ means the far region is the cluster.

## Design (48 fits)

| Setting | Values |
| --- | --- |
| Input map | `tanh(4x)` (baseline), `x`, `tanh(x)` |
| Near mass m | 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375 |
| Law | uniform over the near atoms (d < 2) with mass m, uniform over the far atoms with 1 − m (atoms of sources 130 and 136) |
| Initializations | 131, 133; every input map uses the identical initial weights |
| Training | identical to Part A of the [skewness protocol](SKEW_PROTOCOL.md): 0.9 alignment (source-131 pairs) + 0.1 SIGReg, 4,000 Adam updates, uniform plan 9600131 |

## Predictions

Per cell, the earlier skewness rule is applied to |γ| in that input map's own f0: junction fold if 0.75 ≤ |γ| ≤ 2.6, no
fold if |γ| ≤ 0.45 or |γ| ≥ 3.8, and no prediction otherwise. Junction fold means backward fraction > 0.1.

Specific geometry hypotheses (H_geom):

1. Native m = 0.9375 does not fold under `tanh(4x)` at either initialization. It folds under `x` and under `tanh(x)`
   for at least one initialization each, because the cluster is no longer squashed into ejection range.
2. m = 0.125 does not fold under `tanh(4x)` at either initialization. It folds under `x` for at least one
   initialization, with the roles reversed: the far region is the cluster and the near branch turns back, so the
   turning point lies below distance 3.
3. m = 0.5 does not fold under `x` or `tanh(x)` at either initialization (|γ| ≤ 0.13).

All cells are reported. Failures are kept as falsifications. This is one encoder class and an empirical check, not an
analytic guarantee.

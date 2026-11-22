# Does a cluster that is compact in the encoder's input restore the fold?

Fixed November 22, 2026, after scoring the [geometry test](GEOMETRY_PROTOCOL.md) and before any fit below was trained.

## What the geometry test showed

With the input map changed from `tanh(4x)` to `x` or `tanh(x)`, none of the 32 fits folded, at any near mass from 0.125 to
0.9375, even though the initial codes had skewness up to 2.7. So 19 of 38 skewness-rule predictions failed, and so did
H1 and H2. Without folding, the trained codes reached skewness about 0 monotonically, by stretching the near region over
most of the code range.

Refined claim (H_compact): the fold needs a heavy region whose span in the encoder's input is tiny, so that the
Gaussianizing (quantile) map would require a very steep encoder there. Skewness of the initial code mattered only
because under `tanh(4x)` the whole near region (d < 2) is such a region.

## Design (32 fits)

The window mass is m, spread uniformly over the atoms in window W. The tail carries 1 − m, uniform in distance (atoms
reweighted by the inverse count in 0.25-wide distance bins) over the stated range.

| Law | Window W | x-span of W | Tail range |
| --- | --- | ---: | --- |
| wall | [1, 1.02) | 0.020 | [2, 16] |
| wide | [1, 1.25) | 0.200 | [2, 16] |
| far | [8, 9) | 0.014 | [1, 8) |

| Setting | Values |
| --- | --- |
| Masses m | 0.75, 0.875 |
| Input maps | `x` and `tanh(x)` for all three laws; `tanh(4x)` for wall and far |
| Initializations | 131, 133 |
| Training | identical to the geometry test (geometry.fit) |

## Predictions (junction fold means backward fraction > 0.1)

1. wall under `x` and `tanh(x)`: fold at m = 0.75 for both initializations, and at m = 0.875 for at least one
   initialization per map.
2. wide under `x` and `tanh(x)`: no fold for any mass or initialization.
3. far under `x`: fold at m = 0.75 for at least one initialization. Under `tanh(x)`: same prediction.
4. wall under `tanh(4x)`: fold at m = 0.75 for both initializations. far under `tanh(4x)`: no prediction.

All cells are reported, and failures are kept. One encoder class; an empirical check, not an analytic guarantee.

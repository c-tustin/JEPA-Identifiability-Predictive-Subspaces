# Does the near cluster's position (wall vs junction) control the fold?

Fixed November 21, 2026, after scoring the [skewness protocol](SKEW_PROTOCOL.md) and before any of the fits below were
trained.

## What the skewness test showed

- Skewness ≤ 0.45 never folded.
- Skewness inside the pre-registered band was not sufficient. Laws that raised skewness by piling far mass at distance
  2–3 did not fold at m = 0.5, 0.9375 or 0.96875.
- At m = 0.75, moving both regions toward distance 2 removed the fold.

The remaining candidate is where the near cluster sits. Near the reflecting wall (distance 1), the encoder's
tanh-compressed input cannot spread the cluster, and nothing lies below it in distance.

## Design (24 fits)

| Setting | Values |
| --- | --- |
| Atoms | as in the skewness protocol (sources 130 and 136) |
| Near mass m | 0.5, 0.75, 0.9375 |
| Near window W (near mass placed uniformly over the actual atoms in W) | [1, 1.25), [1.25, 1.5), [1.5, 1.75), [1.75, 2) |
| Far conditional | unchanged: uniform over all far atoms |
| Initializations | 131, 133 |
| Uniform plan | 9600131 |

Training is identical to Part A of the skewness protocol.

Hypothesis H_wall predicts a junction fold (backward > 0.1) at m = 0.75 for W = [1, 1.25) with both initializations, and
none for W = [1.75, 2) with both initializations. At m = 0.5 and 0.9375 it makes no prediction.

H_skew is scored on the same cells using the earlier rule, with γ computed in each initializer's own f0.

All cells are reported. If H_wall fails, that is retained as a falsification. This is an exploratory follow-up with one
encoder class, not an analytic guarantee.

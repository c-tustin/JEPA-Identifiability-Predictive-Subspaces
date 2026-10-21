# Learning distance from a shrinking triangle

This experiment recreates a simple version of Theorem 1 from [*When Does LeJEPA Learn a World Model?*](https://arxiv.org/abs/2605.26379), by Klindt, LeCun and Balestriero.

**The question:** Can a network work out a useful distance scale without being given distance labels?

## The setup

An agent looks straight at a fixed triangle. When the distance doubles, the triangle’s apparent width and height halve. Its shape stays the same. The network receives the six coordinates of its three corners, rather than a full image.

We train it on pairs of nearby views with two goals:

1. Give the two views similar output numbers.
2. Keep the outputs spread out in a standard bell-shaped distribution, rather than letting every view produce the same number. **SIGReg** is the method we use for this second goal.

The correct distances are used only to generate the scene and check the result afterward.

## Why we use log-distance

Distance must be positive. The paper’s result assumes that the hidden quantity being learned follows a bell-shaped, or **Gaussian**, distribution that extends in both directions. We therefore choose that quantity to be log-distance.

Log-distance treats equal ratios as equal steps: moving from distance 2 to 4 changes it by the same amount as moving from 4 to 8.

Pairs follow a particular random movement rule called **Ornstein–Uhlenbeck (OU)**. It tends to keep successive log-distances close while preserving the required distribution. Ordinary random walking does not automatically have these properties.

## What the paper predicts

Under these ideal assumptions and perfect training, the network’s output must equal the chosen log-distance scale, possibly with its sign reversed. Increasing distance might make the output increase or decrease; either result contains the same information.

That is the identifiability result we test here. It recovers log-distance. Turning that number into physical distance still requires an exponential calculation and a known distance scale.

The [mathematical notes](THEORY.md) contain the equations, proof, exact training settings and SIGReg formula.

## What happened

We repeated training three times with different random numbers. A straight line fitted to the learned output predicted log-distance on new examples with **R² scores from 0.9890 to 0.9906**. Using apparent size directly gave scores from 0.7661 to 0.7776. A score of 1 would be a perfect fit.

Recovery was close, especially in the middle of the distance range, but less accurate at the extremes. The learned outputs also did not quite match the intended bell-shaped spread.

![Original reproduction](output/reproduction.png)

See the [recorded numbers](output/metrics.json) or the [detailed results table](THEORY.md#observed-results).

## Run it

After installing the dependencies as described in the [main README](../../README.md#try-it), run this from the repository’s top folder:

```sh
python behavior-policy-experiments/identifiability/reproduce.py
```

The default is three runs of 10,000 training steps. It saves the scores, a figure and trained models in `output/`, replacing the existing files there.

For a short check that leaves the saved results alone:

```sh
python behavior-policy-experiments/identifiability/reproduce.py --steps 20 --seeds 0 --output /tmp/jepa-smoke
```

Run its four numerical checks with:

```sh
python -m unittest discover -s behavior-policy-experiments/identifiability -p 'test_*.py' -v
```

## What this does and does not show

The triangle has a fixed physical size. If both its size and distance could change, a small nearby triangle could look exactly like a large distant one. These observations alone would not distinguish them.

The result also depends on the special movement rule and ideal visibility. It does not automatically apply to an agent choosing its own movements, objects disappearing from view, or low-resolution images. The [curiosity experiment](../curiosity/README.md) tests one change to the movement rule.

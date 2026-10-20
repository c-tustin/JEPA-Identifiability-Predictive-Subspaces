# JEPA Identifiability and Predictive Subspaces

A minimal unsupervised experiment studying identifiability from an object whose apparent size scales as **1/distance**. The original experiment adapts Theorem 1 of [*When Does LeJEPA Learn a World Model?*](https://arxiv.org/abs/2605.26379), by David Klindt, Yann LeCun and Randall Balestriero.

An agent observes the projected vertices of a fixed triangle. A neural encoder learns from pairs of observations using alignment and **SIGReg**, without distance labels. With stationary Gaussian dynamics in log-distance, the ideal representation recovers log-distance up to sign.

This is a one-dimensional toy adaptation, not the authors' full experimental pipeline. It uses continuous vertex coordinates rather than raster images. Predictive subspaces beyond this scalar setting are future work.

![SIGReg identifiability experiment](behavior-policy-experiments/identifiability/output/reproduction.png)

## Behavior-policy extension

The new [behavior-policy experiments](behavior-policy-experiments/README.md) keep the original reproduction alongside an ICM curiosity-policy test based on Pathak et al.'s official GitHub code. The comparison uses OU, random-action and learned curiosity trajectories, with separate on-policy and common-grid evaluation.

```sh
python behavior-policy-experiments/curiosity/experiment.py
```

See the [curiosity implementation and protocol](behavior-policy-experiments/curiosity/README.md) and its [results](behavior-policy-experiments/curiosity/output/RESULTS.md). All original code, write-up and outputs now live under `behavior-policy-experiments/identifiability/`.

Across three seeds, common-grid log-distance recovery is R² = **0.9946 ± 0.0012** for OU, **0.9609 ± 0.0004** for random actions, and **0.8832 ± 0.0548** for learned ICM. The ICM condition also has incomplete coverage in two seeds and substantial failures to maintain unit embedding variance. These results show degraded recovery in this configuration; they do not isolate a failure of the population theorem from distribution changes and optimization errors.

## Quick start

Use Python 3.12 (the pinned PyTorch version supports Python 3.8–3.12):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python behavior-policy-experiments/identifiability/reproduce.py
```

The default run trains three seeds for 10,000 steps each on CPU and writes metrics, a figure and checkpoints under `behavior-policy-experiments/identifiability/output/`. Training generates fresh data locally; it does not download datasets or pretrained models.

For a quick smoke run that preserves the checked-in results:

```sh
python behavior-policy-experiments/identifiability/reproduce.py --steps 20 --seeds 0 --output /tmp/jepa-smoke
```

Run the numerical checks:

```sh
python -m unittest discover -s behavior-policy-experiments/identifiability -p 'test_*.py' -v
python -m unittest discover -s behavior-policy-experiments/curiosity -p 'test_*.py' -v
```

## Results

Across three seeds, a held-out linear probe recovers the Gaussian log-distance coordinate with **R² = 0.9890–0.9906**. A probe directly from apparent size reaches R² = 0.7661–0.7776. The learned raw embedding variance is 0.9250–0.9401, so Gaussianity and recovery remain approximate.

The encoder observes only triangle vertices during optimization. Latent coordinates are used for evaluation and resolving the unavoidable global sign. Raw and independently calibrated metrics are both reported.

## Files

- [Derivation and experimental details](behavior-policy-experiments/identifiability/README.md): assumptions, Hermite proof, SIGReg formula and limitations.
- [Experiment](behavior-policy-experiments/identifiability/reproduce.py): renderer, encoder, SIGReg and evaluation.
- [Numerical checks](behavior-policy-experiments/identifiability/test_reproduce.py): reference-loss equivalence, gradients and sensor geometry.
- [Recorded metrics](behavior-policy-experiments/identifiability/output/metrics.json) and [figure](behavior-policy-experiments/identifiability/output/reproduction.png).
- `behavior-policy-experiments/identifiability/output/encoder_seed{0,1,2}.pt`: model state, calibration, configuration and training history.

## Scope and assumptions

Distance is positive, so the Gaussian coordinate is **log-distance**, not physical distance. Paired states follow a stationary Ornstein–Uhlenbeck transition in that coordinate. The object has fixed physical size and remains visible with unchanged orientation. Unknown variable physical size introduces a size–distance ambiguity. Ordinary walking policies and finite raster images require additional analysis.

The numerical implementation uses the scalar specialization of the [authors' SIGReg code](https://github.com/klindtlab/lejepa-identifiability/blob/main/experiments/lejepa_id/losses.py). See [third-party notices](THIRD_PARTY_NOTICES.md) for SIGReg and the curiosity port, including their respective licenses. This repository's license is [MIT](LICENSE).

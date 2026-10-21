# Curiosity-policy implementation and experiment details

For the plain-language overview and main findings, start with the [README](README.md).

## Question

Does a SIGReg encoder still linearly recover log-distance when its positive pairs come from a curiosity-trained agent instead of the Gaussian OU process?

The observation remains a fixed triangle's six projected vertex coordinates, proportional to 1/distance. Representation fitting uses only observations. Ground-truth distance and log-distance are used by the simulator and evaluation, not as training labels or policy inputs.

## Sources and actual code reuse

The curiosity source is Pathak, Agrawal, Efros and Darrell, [*Curiosity-driven Exploration by Self-supervised Prediction*, ICML 2017](https://pathak22.github.io/noreward-rl/), supplied locally as `curiosity_driven_exploration.pdf`. The official implementation is [pathak22/noreward-rl](https://github.com/pathak22/noreward-rl).

The port is based on commit **3e220c2177fc253916f12d980957fc40579d577a**. The relevant unmodified source files, original README and complete license are retained in [`upstream/noreward-rl`](upstream/noreward-rl), with SHA-256 hashes in [`PROVENANCE.json`](upstream/noreward-rl/PROVENANCE.json). Every experiment verifies these hashes before training. No upstream TensorFlow code is executed or imported.

The PyTorch implementation in [`experiment.py`](experiment.py) ports these parts:

| Upstream code | Toy implementation |
| --- | --- |
| `src/model.py: StateActionPredictor` | Shared feature encoder phi; inverse action classifier; action-conditioned forward feature predictor |
| `StateActionPredictor.forwardloss` | Half the squared feature-error **sum**, averaged over transitions; the feature dimension is factored out of the mean, as upstream |
| `StateActionPredictor.pred_bonus` | Intrinsic reward from the forward error, evaluated before updating ICM |
| `src/a3c.py: process_rollout` | Bootstrapped discounted returns; terminal masks stop returns at reset |
| `src/a3c.py: A3C` | Actor loss, value loss and entropy regularization; separate policy and predictor optimizers |
| `src/constants.py` | Gamma 0.99, rollout length 20, gradient clipping 40, forward/inverse mixture beta 0.2 |

This is an attributed **source-derived port**, not an exact run of the legacy TensorFlow/game stack. Four game CNN layers become a small MLP over six landmarks; feature width is 32 instead of 288. The policy is feedforward because the ideal visual sensor is Markov, rather than an LSTM. Training is synchronous vectorized advantage actor–critic, not distributed A3C. There is no game action repeat. The worker's batch-size/predictor gradient scale factors are replaced by mean reductions and separate optimizer learning rates. The policy and ICM are separate from the SIGReg encoder.

An important implementation detail is preserved: the official source does **not** stop the forward-loss gradient through phi(s_next); the port likewise trains both feature branches. The intrinsic reward is detached for the actor–critic update, so policy optimization cannot backpropagate through the simulator or manipulate the predictor directly.

## Environment and conditions

The controlled physical environment has distance d in [1,16], initial distances sampled uniformly in physical distance, and three discrete actions: approach by 0.25, stay, retreat by 0.25. Overshoot reflects at the boundaries; no observations are resized to erase apparent size. There is no external reward and no injected observation or actuation noise.

1. **OU:** independent stationary Gaussian pairs in z=log(d/4)/0.7 with rho=0.9, rendered through the original inverse-distance mapping. This condition is unbounded in physical distance, as required by the original Gaussian construction.
2. **Random:** uniformly sampled physical actions in the bounded environment.
3. **ICM:** actions sampled from the learned curiosity policy in that same bounded environment.

Random and ICM conditions share dynamics, sensor, bounds, start distribution, episode lengths, transition budgets and representation-learning settings. The random-action control helps distinguish curiosity learning from changing the environment and data-generating process. OU versus either behavior condition changes multiple assumptions, so it does not isolate the effect of curiosity alone.

ICM and policy train jointly, then **freeze**. Fresh train, calibration and test trajectories are collected with different seeds and disjoint episodes. Each behavior trajectory discards 128 burn-in steps and records 64 transitions. Resets are never treated as positive pairs, and no time steps within an episode are randomly split between train and test. Burn-in does not establish stationarity; distribution diagnostics are reported rather than assuming it.

Defaults per seed:

- Policy: 600 updates × 20 steps × 16 parallel environments = 192,000 interactions; independent training episodes have length 256.
- Policy Adam learning rate 0.0001; ICM Adam learning rate 0.001; gamma 0.99; entropy weight 0.0005; intrinsic reward scale 0.2 and clipping at 1. These are toy settings drawing on the upstream constants/Mario settings, not a claim to reproduce a particular game run.
- Representation data: 256 training trajectories × 64 recorded steps = 16,384 pairs; 64 calibration and 64 test trajectories (4,096 pairs each). OU uses matching independent pair counts.
- Each representation is initialized from the same seed across conditions and trained from scratch for 4,000 minibatches of 512 pairs, with alignment + SIGReg, lambda 0.1, and the same cosine-decay schedule as the original toy implementation. No batch whitening is used.
- Seeds: 0, 1 and 2. The ICM feature encoder is never reused as the representation encoder.

## Evaluation and interpretation

Two tests answer different questions:

- **Behavior-distribution R²:** fit a linear embedding-to-log-distance probe on separate calibration trajectories and score it on new test trajectories under the same frozen policy.
- **Common-grid R²:** fit a separate evaluation-only probe on independently sampled log-distances, then score on a shared log-distance grid covering physical distances [1,16]. This reveals nonlinear geometry or missing coverage that a narrow behavior distribution can conceal. The common-grid probe is not used for training, model selection or early stopping.

Additional metrics include raw and calibration embedding means/variances, raw alignment, held-out characteristic-function discrepancy from the Gaussian target (SIGReg divided by sample count), latent skewness and excess kurtosis, distance from a standard Gaussian CDF after standardization, pair correlation, paired marginal mean shift, fraction of samples in the common range, and occupancy of 32 physical-distance bins. No Gaussianity p-values are reported for correlated trajectory samples. The learned policy's weight change and action-distribution difference from uniform are recorded, along with inverse-model accuracy relative to the majority-action baseline.

**The mathematical guarantee does not automatically transfer.** The bounded behavior conditions cannot have an exact non-degenerate Gaussian log-distance marginal. Their action-driven transitions also differ from isotropic additive Gaussian OU noise. The experiment therefore tests empirical linear recovery under those policies; it does not apply the original theorem's alignment lower bound or approximate-identifiability bound to them. A high R² is evidence of near-linear recovery on a tested distribution, not a proof of identifiability over all encoders. A lower R² is evidence of failure of this trained model/configuration, not a counterexample to the theorem under its original assumptions.

There is a direct incompatibility here: an affine function of bounded log-distance is bounded, so it cannot have an exact standard-Gaussian distribution. Thus exact affine recovery plus exact Gaussian matching cannot hold simultaneously in either bounded behavior condition. This follows from the support we chose, before learning any curiosity policy. The useful comparison is how well approximate recovery survives, and how learned curiosity changes it relative to random actions in the same bounded environment.

One-dimensional monotonic Gaussianization is also a simple problem: these experiments do not demonstrate disentangling several independent latent factors. Coverage, bounded geometry and optimization can each affect the outcome. The random-action control, raw variances, common-grid test and multiple seeds make those limits visible, but do not isolate every mechanism.

## Run and inspect

From the repository root with `requirements.txt` installed:

```sh
python behavior-policy-experiments/curiosity/experiment.py
```

For a short execution check, keeping recorded results intact:

```sh
python behavior-policy-experiments/curiosity/experiment.py --quick --seeds 0 --output /tmp/behavior-policy-smoke
```

The quick mode has only 3 actor–critic updates and 10 representation updates. Its R² is **not** an identifiability result; a nearly collapsed random network can still have a high calibrated R² in one dimension.

```sh
python -m unittest discover -s behavior-policy-experiments/identifiability -p 'test_*.py' -v
python -m unittest discover -s behavior-policy-experiments/curiosity -p 'test_*.py' -v
```

Results are in [`output/RESULTS.md`](output/RESULTS.md), [`output/metrics.json`](output/metrics.json) and [`output/comparison.png`](output/comparison.png). Checkpoints retain the behavior policy, ICM and each representation. The compressed evaluation arrays support replotting without retraining. Data generation can be reconstructed from the policy checkpoint, configuration and collection seeds in the source.

To reconstruct all held-out trajectories and verify the recorded recovery scores and variances from the saved checkpoints, without training:

```sh
python behavior-policy-experiments/curiosity/experiment.py --evaluate-only
```

This also regenerates the report and figure. The checked-in evaluation was verified in this way for all nine seed/condition combinations.

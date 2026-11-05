# Does movement matter when coverage is the same?

**Main question:** Can two behavior policies see the same distances equally often, yet learn different representations or reach goals differently?

[Results and figures](output/RESULTS.md) · [Fixed protocol](PROTOCOL.md) · [How slow features are calculated](MATH.md)

## First result

Completed 40 encoder fits and 80 planning evaluations. Every policy has mean log-distance R² about 0.99484 and no detected folds. Mean shared-data goal error is 0.0586–0.0590; policy-native errors vary more, but none of the corrected comparisons is significant. The full paired tables and all curves are in the [results](output/RESULTS.md).

Changing these movement rules at the fixed marginal does **not** reproduce the original zoo's large ICM–OU difference. This is a result for these kernels and training settings, not a proof that behavior policies never affect recovery or planning. In particular, correcting ICM's occupancy also changes its dynamics.

## What the zoo already tells us

The [original zoo](../README.md) found large recovery differences: mean pure-policy log-distance R² was 0.603 for ICM, 0.961 for random actions and 0.995 for the ideal OU pair source. But these policies also produced different visited-distance distributions. The reported correlation between Gaussian-looking occupancy and recovery is descriptive; it does not establish that occupancy caused the differences.

The [expanded zoo](../new-policy-results/README.md) found that guarded biased walk → OU and size-feedback bias → OU beat fixed OU on fresh seeds. Neither established an improvement over the best old curriculum. Those are finite-training **representation** results; they do not yet show better goal-reaching.

This follow-up separates two questions:

1. **Representation:** If the visited-distance distribution is held fixed, does the movement rule change recovery of log-distance or cause folds?
2. **Planning:** Does any difference affect actually reaching a goal, and does the result depend on which action data train the predictor?

## What stays the same

Same triangle, inverse-distance observation, one distance coordinate, and approach/stay/retreat actions of size 0.25. Same encoder, SIGReg, loss weight, training schedule and budget. No regularizer ablation.

All four policies have exactly the same distance counts at **both** ends of the training pairs. Their stationary distributions, global action counts, and mean physical movement also match. This removes differences in histogram shape as well as differences in how many bins are occupied.

The collection protocol uses a fixed reachable 61-state distance grid, replacing the earlier continuously sampled resets. Stationary pairs are enumerated with exact counts, rather than collected from a single trajectory. These choices make the controls exact; the test does not measure how long a rollout takes to achieve coverage.

## What changes

| Policy | Movement rule |
| --- | --- |
| Reference local walk | Ordinary local crossings, adjusted to the common stationary distribution |
| Slow crossing near 2.5 | Crossings near 2.5 are less frequent; crossings elsewhere compensate |
| Slow crossing near 8 | The same idea, with the slowdown around 8 |
| ICM-shaped, coverage corrected | Relative crossing preferences from one frozen zoo ICM actor, corrected to the same occupancy |

The ICM-shaped rule is a **new controlled policy**, not the original ICM. The reference is a physical local walk, not the old ideal Gaussian/OU pair source. We preserve the original zoo results and use these follow-ups to test a mechanism.

## What is a “slow feature”?

It is a number computed from distance that changes little under a particular movement rule, while still varying across the visited distances. If the agent rarely crosses distance 8, “which side of 8 am I on?” can change more slowly than distance itself.

We calculate these features from the policy's exact transition matrix, before training the encoder. They show what the alignment term rewards under centering and unit variance. The encoder's unchanged SIGReg means they are a benchmark, not a promised final representation. See the [short derivation](MATH.md).

## How we test planning

After training each encoder, freeze it and train the same action-conditioned predictor two ways:

- **Shared data:** every encoder gets the same observation/action transitions. Differences better isolate representation quality for this predictor and planner.
- **Native data:** each encoder gets its own policy's transitions. This additionally tests where the policy sampled informative actions.

Both planners receive current and goal embeddings. We score actual terminal distance error on identical reachable goals, not just error in embedding space. All goal cases have a zero-error physical oracle. We also show errors for starts near, in the middle, and far from the object.

Ten paired seeds vary encoder initialization and training randomness. They do not vary the fixed policy kernels or the single frozen ICM actor. All seeds are included in statistical comparisons.

## What remains open

- Does the result persist for other shared stationary distributions and independently trained ICM actors? This test fixes one broad marginal and one old actor.
- Does pairing observations several actions apart make movement rules matter more for recovery and reaching, while keeping both endpoint marginals matched?
- Which native-data planning differences repeat with more predictor seeds, and are they localized to rarely sampled actions at particular distances?

These follow-ups keep the emphasis on behavior policies, identifiability and planning. The current result does not identify the cause of the original ICM folds.

## Reproduce

From the repository root:

```sh
python3 -m unittest discover -s behavior-policy-experiments/policy-zoo/matched-coverage -p 'test_*.py' -v
python3 behavior-policy-experiments/policy-zoo/matched-coverage/run.py --workers 6
python3 behavior-policy-experiments/policy-zoo/matched-coverage/analyze_matched.py
```

Dependencies are the repository's existing NumPy, PyTorch and Matplotlib versions. No downloads are needed. The runner resumes completed fits and encoder checkpoints. An interrupted predictor fit reruns deterministically. A changed code/settings hash requires a separate output directory. Run `verify.py` to replay all saved encoder/planning evaluations; this optional verifier also needs SciPy (see `requirements.txt`).

Verification: all three controlled-kernel checks pass. All 40 saved encoder evaluations and 80 physical planning rollouts reproduce exactly; all nine statistical comparisons were checked. A complete reference training replay reproduces encoder and both predictors with zero parameter difference. [Verification record](output/VERIFICATION.json).

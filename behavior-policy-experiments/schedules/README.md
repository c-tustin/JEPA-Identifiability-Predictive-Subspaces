# Starting with curiosity, then switching toward OU

This experiment asks whether curiosity is a useful starting point if we later train with the more even distance sampling used by OU. It also tests whether changing OU settings over time helps.

## Why was ICM worse before?

The saved runs show three problems:

- Two agents chose to move closer about 56–58% of the time. They spent much of their time near the object and covered less of the distance range.
- They scored better on the distances they usually visited than on the full test range. Their learned distance scales did not transfer well to unfamiliar views.
- Their learned outputs had variances around 14 and 18, although the training target was 1. That is a training problem as well as an exploration problem.

There is also a mismatch between the two learning goals. Curiosity seeks changes that its prediction model finds surprising. It does not try to keep log-distance bell-shaped, which is what makes the original proof work. SIGReg still asks the distance network for bell-shaped outputs. If the visited log-distances are strongly clustered or uneven, meeting that request can require stretching the learned distance scale nonlinearly.

Equal physical moves cause larger apparent-size changes near the object, which could make those moves attractive to a curiosity system. But a check of the final curiosity models does not support that explanation consistently across runs. Their rewards also changed as they learned, and the saved snapshots cannot tell us the entire history. We therefore treat the exact cause as an open question.

## What does “decreasing entropy toward OU” mean here?

**Action entropy** measures how random the agent’s action choices are. The **entropy bonus** is the training incentive to keep those choices random. Reducing the bonus does not guarantee that actual randomness falls at the same rate.

OU uses continuous random changes to distance, rather than the same three discrete actions. Its randomness is a different quantity. We cannot directly turn one entropy value into the other.

We therefore test two separate changes, and then combine them:

1. **Reduce the action-entropy bonus while the curiosity agent learns.** Start at the original value, 0.0005, and reduce it to zero. Afterward, stop updating that policy and collect its training pairs.
2. **Gradually replace curiosity pairs with OU pairs while the distance network learns.** Use only curiosity pairs for the first quarter of training, increase the OU share during the middle half, and finish with only OU pairs for the last quarter.

The second change is a rule for choosing training pairs. It does not claim to be one uninterrupted three-action walk. The policy-training schedule and the later distance-network schedule run in separate stages.

## The eight comparisons

| Training rule | What changes |
| --- | --- |
| Fixed OU | Original OU settings throughout; reference result |
| Fixed ICM | Original curiosity policy throughout; reference result |
| ICM, entropy bonus decreases | Less incentive for random actions during policy learning |
| ICM → OU | Gradual handover of distance-network training pairs |
| ICM entropy decreases → OU | Both changes above |
| OU correlation changes | Correlation rises from 0.5 to 0.9 |
| OU spread changes | Log-distance spread falls from 1.0 to 0.7 |
| OU correlation and spread change | Both OU changes above |

Higher OU correlation means that two paired views tend to be closer together. OU spread controls how widely physical distances are sampled. All schedules keep their starting settings for the first quarter, change them during the middle half, and hold their final settings for the last quarter.

Changing correlation keeps the same Gaussian distribution of log-distances in each epoch. Changing spread changes the distance distribution itself. This distinction matters when comparing the results with the paper’s assumptions.

In the code, correlation is `rho` and spread is `sigma`. For log-distance L, a pair is generated as `L_next = rho * L + sigma * sqrt(1 - rho**2) * noise`, with starting L sampled at that epoch’s spread. The schedule changes these paired samples each epoch, rather than modifying a continuous trajectory that spans all epochs.

An epoch here means 32 randomly sampled minibatches, matching the 16,384-pair training pool and batch size of 512. Sampling is with replacement, so it is not an exact pass through every pair. Settings change only at epoch boundaries.

## What stays the same?

Every distance network gets 4,000 training updates with the same architecture, learning rate schedule and SIGReg weight. All schedules are chosen before looking at their scores. The network still receives only observations, without distance labels.

The fixed ICM and fixed OU results reuse the original saved models when their settings match exactly. New conditions train from the same initial weights for each run. OU settings are applied to a fixed pool of Gaussian starting values and noise, so a changing parameter does not also introduce a different random pool.

The earlier [behavior-policy comparison](../curiosity/README.md) and its saved results are preserved.

## What happened?

On the same test distances, averages across the three runs were:

| Training rule | R² |
| --- | ---: |
| Fixed ICM | 0.8832 |
| ICM with decreasing entropy bonus | 0.8844 |
| Fixed OU | 0.9946 |
| ICM → OU | 0.9991 |
| Decreasing ICM entropy bonus, then OU | 0.9991 |
| Changing OU correlation | 0.9947 |
| Narrowing OU spread | 0.9974 |
| Changing both OU settings | 0.9977 |

**The handover helped in all three runs.** Its output variance on the fixed OU test also ended much closer to the target: about 0.96 on average, versus about 15.77 for fixed ICM on that same test.

Reducing the entropy bonus alone did not solve the problem. Measured action randomness changed very little, and adding that reduction to the handover gave almost no extra benefit. This result concerns the original small bonus decreasing to zero, not every possible entropy setting.

For OU, beginning with a wider range of log-distances and then narrowing it helped more than changing correlation alone. These are small experiments with three runs, so the results are evidence for this setup rather than a general rule.

## Read the results

The [results report](output/RESULTS.md) includes every run, the schedules, and measurements of actual action entropy. It reports two tests:

- **Same distance range:** the same test distances from 1 to 16 for every model.
- **Fixed OU test:** the same Gaussian test states for every model, including some physical distances outside that range.

Both scores ask how well a straight line reads log-distance from the learned number. R²=1 is perfect. Output variance and Gaussian-matching error are also recorded, since a good straight-line score alone does not show that the training target was met.

For OU-to-OU changes, each correlation setting retains the linear optimum described by the original proof. That does not establish that a neural network will find it. The curiosity mixture and changing spread also change the data distribution over time; no new mathematical identifiability guarantee is claimed for those schedules.

## Run it

After following the [installation instructions](../../README.md#try-it), run this from the repository’s top folder:

```sh
python behavior-policy-experiments/schedules/run_schedules.py
```

The full comparison trains three more curiosity agents and 18 new distance networks. It reuses six matching reference models. Saved results go in this folder’s `output/`; use `--output /tmp/my-schedule-run` to keep a separate run.

Check the program quickly without overwriting saved results:

```sh
python behavior-policy-experiments/schedules/run_schedules.py --quick --seeds 0 --output /tmp/schedule-smoke
```

Check the saved scores without training again:

```sh
python behavior-policy-experiments/schedules/run_schedules.py --evaluate-only
```

Run the schedule checks:

```sh
python -m unittest discover -s behavior-policy-experiments/schedules -p 'test_*.py' -v
```

## Why does the handover help?

The [curriculum-control study](../curriculum-controls/README.md) follows up in the same distance-only world. It compares random→OU with ICM→OU, checks the original runs exactly, and tests optimizer resets, learning-rate restarts, and matched OU-pair exposure.

# What changes when the agent learns where to move?

The [original experiment](../identifiability/README.md) uses a special random movement rule chosen to match the paper’s assumptions. Here we replace that rule with a learned **behavior policy**: a rule for choosing the next action.

The agent still sees a triangle that shrinks with distance. It can move closer by 0.25, stay, or move farther away by 0.25. In the two action-based tests, distance stays between 1 and 16; attempts to move past an end bounce back.

## How curiosity works

We use the **Intrinsic Curiosity Module (ICM)** from Pathak et al.’s [curiosity-driven exploration paper](https://pathak22.github.io/noreward-rl/).

It learns two things from the agent’s experience:

- Given the view before and after a move, which action did the agent take?
- Given the current view and the action, what should the next view look like in the model’s learned description?

The agent receives a curiosity reward when the second prediction is wrong. It learns to choose actions that earn more of that reward. There is no goal reward and no distance label.

We adapted the [authors’ official code](https://github.com/pathak22/noreward-rl) to PyTorch and the triangle scene. Their original code uses TensorFlow and game images. The [technical notes](TECHNICAL_NOTES.md#sources-and-actual-code-reuse) explain the changes; a fixed copy of the relevant original files and license is kept in [`upstream/`](upstream/noreward-rl).

## The comparison

We test three movement rules:

| Rule | How the next view is chosen |
| --- | --- |
| OU baseline | The special random rule from the original experiment |
| Random actions | Equal chances of moving closer, staying or moving away |
| Curiosity policy | A network chooses actions after learning from curiosity rewards |

After learning a curiosity policy, we stop changing it and collect fresh trips through the scene. A separate network then learns a single output number from pairs of views, using the same SIGReg training method as the original experiment. It does not reuse the curiosity model’s learned description.

Every rule gets the same number of training pairs. Training and testing use different trips, and a jump back to a starting position is never counted as a real movement pair.

## What happened

We ran each comparison three times. We measured how well a straight line could turn the learned output into log-distance on the **same test distances from 1 to 16**.

| Movement rule | Average R² | Variation across runs |
| --- | ---: | ---: |
| OU baseline | 0.9946 | ± 0.0012 |
| Random actions | 0.9609 | ± 0.0004 |
| Curiosity policy | 0.8832 | ± 0.0548 |

A score of 1 means a perfect straight-line fit. The variation column is the sample standard deviation across the three runs.

The curiosity agents did worse across the shared test range. Two spent much of their time near the object and covered less of the distance range. In those runs, the output numbers also spread out far more than the training target intended.

When we tested only on distances each curiosity agent tended to visit, the average score was higher: **0.9589**. Doing well on familiar views can therefore hide problems elsewhere in the scene.

The [full report](output/RESULTS.md) includes every run, checks on the output spread, and evidence that the policy actually learned. The [comparison figure](output/comparison.png) shows the results visually.

## Does identifiability still hold?

**The learned distance scale was less reliable under curiosity in this experiment.** We cannot claim that the original mathematical guarantee applies to these agents.

There are two reasons to be careful about the conclusion. First, both action-based tests change the distribution of distances and movements assumed by the theorem. Second, some networks did not meet the intended training target, so training problems may explain part of the drop.

There is also a simple mathematical limit in our bounded scene: a straight-line function of a quantity limited to a fixed range is still limited to a fixed range. It cannot have an exact Gaussian distribution, which has unbounded tails. This issue already affects the random-action test; curiosity is not its only cause.

These results show how well approximate recovery survived in this particular setup. They do not show that curiosity always prevents learning distance, or that the paper’s theorem is wrong.

The [follow-up schedule experiment](../schedules/README.md) tests a possible remedy: start with curiosity, then gradually switch the distance network’s training pairs to OU. It also tests decreasing the action-entropy bonus separately and changing OU settings over epochs.

## Run it

Install the dependencies using the [main README](../../README.md#try-the-code). Then run commands from the repository’s top folder.

Full comparison:

```sh
python behavior-policy-experiments/curiosity/experiment.py
```

This trains three curiosity agents and nine distance-learning networks. It saves new reports, figures and models in `output/`, replacing the existing files there. Use `--output /tmp/my-behavior-run` to keep a separate run.

To check the saved scores without training again:

```sh
python behavior-policy-experiments/curiosity/experiment.py --evaluate-only
```

That command loads the saved models, recreates the test data, checks the recorded scores, and regenerates the report and figure.

For a quick check that the program runs:

```sh
python behavior-policy-experiments/curiosity/experiment.py --quick --seeds 0 --output /tmp/behavior-policy-smoke
```

The quick run is too short to judge learning quality. Run the six numerical checks with:

```sh
python -m unittest discover -s behavior-policy-experiments/curiosity -p 'test_*.py' -v
```

For exact settings, equations, code changes and measurement details, see the [technical notes](TECHNICAL_NOTES.md).

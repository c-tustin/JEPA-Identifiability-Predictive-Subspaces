# JEPA Identifiability and Predictive Subspaces

**Can an agent learn how far away an object is just by watching it change size?** This project tests that question with a triangle that looks half as big when the agent moves twice as far away.

The agent is a small neural network. It sees the positions of the triangle’s three corners, rather than a full image. During training, it never receives the correct distance.

We ask whether the number it learns has a simple straight-line relationship with **log-distance**. This is what we mean by *linear identifiability*. Log-distance measures changes in ratios: doubling the distance adds the same amount wherever the agent starts.

## What we found

We tried three ways of choosing the agent’s movements:

| Movement rule | Recovery score on the same test range |
| --- | ---: |
| A special random rule chosen to match the paper’s assumptions, called OU | 0.995 |
| Random choices to move closer, stay, or move farther away | 0.961 |
| A learned curiosity rule that seeks hard-to-predict changes | 0.883 |

These are average **R² scores** across three runs, tested at the same distances between 1 and 16. A score of 1 means a straight line predicts log-distance perfectly from the learned number; lower scores mean a worse fit.

The curiosity-trained agents did worse on that shared test. In two runs they spent much of their time near the object. Their learned numbers also failed to have the bell-shaped spread that training was meant to encourage. These results mix changes in exploration with training problems; they do not overturn the original paper’s mathematical result.

The [experiment guide](behavior-policy-experiments/README.md) explains the experiments. The [full results](behavior-policy-experiments/curiosity/output/RESULTS.md) include every run and the extra checks.

The [schedule experiment](behavior-policy-experiments/schedules/README.md) follows up by gradually switching from curiosity to OU, reducing the incentive for random actions, and changing OU settings over training epochs. It keeps these changes separate so we can see which ones help.

In that follow-up, gradually switching from ICM to OU raised the shared-range score to **0.9991** across three runs. The handover with entropy-bonus decay scored almost identically; these two settings are effectively tied. Reducing the entropy bonus alone did little. Starting OU with a wider distance spread and narrowing it also helped. The [schedule results](behavior-policy-experiments/schedules/output/RESULTS.md) include the full comparison and training checks.

For why the movement rule matters, start with [the plain-language explanation and research ideas](behavior-policy-experiments/analysis/README.md). It links to precise arguments, a controlled counterexample, and proposed extensions to nonlinear dynamics and action-dependent policies. The [SIGReg guide](behavior-policy-experiments/identifiability/SIGREG.md) explains the regularizer step by step.

The larger [curriculum-control study](behavior-policy-experiments/curriculum-controls/README.md) revisits that three-run gain in the same one-distance world. It finds that the gain is seed-dependent: two of five new ICM→OU runs develop folded distance curves and fail badly, even with optimizer and learning-rate resets. See the [updated interpretation](behavior-policy-experiments/curriculum-controls/INTERPRETATION.md) before treating ICM→OU as the best choice.

## Try it

Set up Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a short check that the code works:

```sh
python behavior-policy-experiments/curiosity/experiment.py --quick --seeds 0 --output /tmp/behavior-policy-smoke
```

That short run is too small to draw conclusions. For the full comparison, run:

```sh
python behavior-policy-experiments/curiosity/experiment.py
```

To check the saved results without training again:

```sh
python behavior-policy-experiments/curiosity/experiment.py --evaluate-only
```

The code runs on CPU and creates its own training data. A full run writes new results over the files in its output folder; add `--output /tmp/my-behavior-run` to save a separate run.

## Where to look

- [Original paper reproduction](behavior-policy-experiments/identifiability/README.md): the simple setting where the mathematical result applies.
- [Curiosity experiment](behavior-policy-experiments/curiosity/README.md): what changes when the agent learns where to move.
- [Scheduled exploration](behavior-policy-experiments/schedules/README.md): curiosity-to-OU handovers and OU settings that change over time.
- [Comparison figure](behavior-policy-experiments/curiosity/output/comparison.png) and [detailed results](behavior-policy-experiments/curiosity/output/RESULTS.md).
- [Credits and licenses](THIRD_PARTY_NOTICES.md): the original papers and the code we adapted. This repository uses the [MIT license](LICENSE), with separate notices for borrowed code.

The title also mentions predictive subspaces, but the current experiments study just one changing quantity: distance.

# Behavior policy experiments

A **behavior policy** is the rule an agent uses to decide its next move. These experiments ask whether changing that rule changes what the agent learns about distance.

The scene is always the same: a fixed triangle looks smaller as the agent moves away. The learning system sees the triangle’s corner positions, without being told the distance.

## Three experiments

| Experiment | What it asks | Read more |
| --- | --- | --- |
| Original paper reproduction | Can the network learn log-distance when movements follow the paper’s special random rule? | [Simple explanation](identifiability/README.md) |
| Curiosity comparison | What happens with random actions or a learned curiosity rule instead? | [Setup and findings](curiosity/README.md) |
| Scheduled exploration | Can starting with curiosity and gradually switching toward OU improve recovery? What if OU settings change over epochs? | [Schedules and results](schedules/README.md) |

The original reproduction’s code and saved results are kept in `identifiability/`. The first behavior comparison is in `curiosity/`. The follow-up schedules are in `schedules/`.

## Main finding

On the same test distances, the average recovery score was **0.995 for the paper’s random rule**, **0.961 for random actions**, and **0.883 for curiosity-driven actions**. Closer to 1 is better.

The curiosity agents explored unevenly, and some networks did not learn the intended spread of output values. The [results report](curiosity/output/RESULTS.md) shows these problems alongside the scores.

The follow-up handovers from ICM to OU scored **0.9991**, essentially tied with and without entropy-bonus decay. Fixed OU scored 0.9946 in the matched comparison. These are results from three runs in one world, rather than a general ranking of exploration methods.

See [why policies change recovery and what to test next](analysis/README.md) for the explanation, mathematical arguments, and research directions.

## Run the experiments

First follow the [installation instructions](../README.md#try-it), then run commands from the repository’s top folder:

```sh
# Original reproduction
python behavior-policy-experiments/identifiability/reproduce.py

# Compare all three movement rules
python behavior-policy-experiments/curiosity/experiment.py

# Test curiosity-to-OU handovers and changing OU settings
python behavior-policy-experiments/schedules/run_schedules.py
```

The comparison gives all three rules the same number of training pairs. Its OU run is a fresh comparison run, so its scores differ slightly from the original reproduction, which keeps generating new pairs throughout training.

# Why did starting with curiosity help?

This study keeps the same triangle, its inverse-distance size change, and the single distance coordinate. It tests explanations for the earlier ICM→OU gain. No coordinates or actions are added.

## Questions fixed before running

1. Does ICM→OU beat random→OU? If both help similarly, curiosity may be unnecessary.
2. Does an OU→OU phase change itself help? Its unchanged-training version must exactly equal fixed OU.
3. Does erasing Adam's memory or restarting the learning rate at the final all-OU phase change the ranking?
4. Does the handover still beat OU-only controls that receive exactly the same OU observations, with the same repetitions?

The primary comparison is ICM→OU versus random→OU on five new seeds (3–7), using the common-distance-range R². The original three seeds (0–2) are replayed as a separate historical group. The broader Gaussian OU test is a second outcome. We also report all eight seeds together. Settings and seed counts are fixed before inspecting these results; no early stopping or best-checkpoint selection is used.

## What stays the same

- Scene, encoder, loss, batch size 512, OU correlation 0.9 and log-distance spread 0.7.
- 600 updates to learn each ICM policy; the policy is then frozen. Historical policies are reused only after checking their configuration. Random and OU do not require policy learning.
- 4,000 representation updates in each source/reset condition. The first quarter uses the starting source, the middle half gradually increases the OU fraction, and the final quarter uses only OU. Schedule settings change at epoch boundaries, exactly as in the earlier experiment.
- Per seed, identical initial encoder weights, indices, OU observations, and handover masks. All final all-OU batches are identical across sources and reset conditions.
- Identical independent calibration and test data. Distance labels are used only to evaluate readouts; the fitting function receives observations only.

## The controls

Each starting source—ICM, uniform random actions, and OU—is crossed with four choices at the first all-OU update:

| Tail setting | What changes at that boundary? |
| --- | --- |
| `carry` | Keep Adam's state and the original learning-rate clock |
| `reset_adam` | Erase Adam's running moments and step count; keep the learning-rate clock |
| `restart_lr` | Keep Adam's state; restart the hold-then-cosine learning-rate rule over the remaining updates |
| `reset_both` | Reset both |

All four tails from a source begin with the exact same saved encoder weights. The reset-both comparison across sources removes inherited optimizer state and gives the models identical subsequent data and learning rates. Any remaining difference is due to their different starting weights, under this fixed tail protocol. This is evidence about initialization in this experiment, not a universal convergence claim.

OU→OU uses the *same OU pool* on both sides of the nominal handover. Therefore its carry version is exactly fixed OU, not a switch between two independently sampled OU datasets. The new runner checks the original fixed-OU and ICM→OU encoder weights bit for bit for seeds 0–2.

Two additional OU-only controls discard the non-OU pair slots from the handover and pack the remaining OU occurrences, in order, into batches of 512, plus one possible final partial batch:

- **Compressed clock:** apply the original learning-rate rule to this shorter optimizer budget.
- **Original clock:** each packed batch uses the learning rate at its last pair's original scheduled step.

These controls match the OU observations and their repetitions exactly. They have fewer updates and different batch grouping and regularizer composition. The two clocks expose a timing sensitivity; they do not magically hold every aspect of training equal. Restarting a tail's learning rate also increases total learning-rate exposure, deliberately.

This produces 14 conditions per seed, or 112 final models. Shared prefixes are trained once and forked, saving computation without changing the experiment.

## Run or check it

From the repository root, after the [usual installation](../../README.md#try-it):

```sh
python -m unittest discover -s behavior-policy-experiments/curriculum-controls -p 'test_*.py' -v
python behavior-policy-experiments/curriculum-controls/run_controls.py
python behavior-policy-experiments/curriculum-controls/run_controls.py --evaluate-only
```

The full run replaces the output folder. For a short execution check that preserves the saved study:

```sh
python behavior-policy-experiments/curriculum-controls/run_controls.py --quick --seeds 0 --output /tmp/curriculum-control-smoke
```

Use `--resume` to skip fully completed seeds after an interruption. A partially completed seed is retrained; the resume command requires the same configuration and seed list. Final models, prefix weights and Adam states, policy checkpoints, learning histories, sampling-plan hashes, evaluation arrays, and paired comparisons are saved. The training plan is reconstructed from its seed and verified by its hash.

## Read the outcome

[Results and individual runs](output/RESULTS.md) · [Interpretation](INTERPRETATION.md)

Report averages include sample standard deviations. Paired seed differences and 95% bootstrap intervals are descriptive; only five new seeds are available, and the many exploratory comparisons are not corrected for multiple testing. Results concern this environment and finite training budget. No inference is based on comparing selected best seeds.

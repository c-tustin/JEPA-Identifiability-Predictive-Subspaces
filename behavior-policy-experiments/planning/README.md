# Can the learned coordinates support planning?

The agent receives a current view and a **goal view**, predicts candidate action sequences in its frozen learned coordinates, executes the first action, and repeats. Its planner never receives true distance.

The action-conditioned predictor is trained on observations and physical action labels:

$$
\widehat F(y_t,a_t)\approx y_{t+1},\qquad y_t=h(x_t).
$$

Two tests separate representation quality from exploration coverage:

| Predictor training data | What it tests |
| --- | --- |
| Common random-action data for every encoder | Can the representation support the same planning problem? |
| The behavior policy's own physical data | Does the full exploration-to-planning pipeline work? |

OU pairs have no physical action labels, so native-OU planning is omitted. No artificial labels are invented. Predictor training uses 1,000 updates of 256 transitions. Encoders are frozen and normalized using common observation data, not distance labels. Calibration and test transitions are independent.

## Fixed planning task

There are 32 reachable start/goal pairs per seed. Starts lie between distances 3 and 14; goals differ by one to eight physical steps of 0.25. Every controller has 12 decisions, horizon 8, and the same bank of 128 candidate action sequences. The bank includes a direct path to each reachable integer displacement followed by stays. Candidate cost is terminal squared goal error plus 0.01 times accumulated intermediate squared goal error. An exact-dynamics planning test must succeed before learned-model results are interpreted. A known-dynamics reference checks reachability; a stay-put reference measures the benefit of acting.

Primary score: actual terminal distance error,

$$
e_{\rm goal}=|d_T-d_{\rm goal}|.
$$

Also report success within 0.25 and held-out one-step latent prediction error. Low latent prediction error is not sufficient for good physical planning, especially when the encoder folds different distances together.

```sh
python -m unittest discover -s behavior-policy-experiments/planning -p 'test_*.py' -v
python behavior-policy-experiments/planning/evaluate_planning.py --source legacy
python behavior-policy-experiments/planning/evaluate_planning.py --source policies
python behavior-policy-experiments/planning/evaluate_planning.py --source repairs
```

Each source has its own output directory. `legacy` checks the original three encoders across eight seeds. `policies` checks the four alternative exploration rules across five new seeds. `repairs` checks OU/random/ICM, the two ICM-policy changes, and the monotone diagnostic. Use `--resume` for completed predictors or `--evaluate-only` to verify saved results. `--quick --output /tmp/planning-smoke` is only an execution check.

**Planner validation:** an initial terminal-only random-search pilot failed even with exact dynamics. That pilot was superseded before interpreting learned-policy comparisons; version 2 uses the structured candidates and intermediate cost described above.

This first test uses local goals, point predictions, a fixed search budget and a short horizon. It does not establish arbitrary-distance or partially observed planning. Cases share a trained model, so seeds—not individual trajectories—are the replication units. [Original encoders](output-legacy/RESULTS.md) · [Alternative policies](output-policies/RESULTS.md) · [Repair study](output-repairs/RESULTS.md).

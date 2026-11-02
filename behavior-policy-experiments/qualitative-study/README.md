# Ten trials: what ICM learns, and whether it helps planning

This study plots what ICM, OU and ICM→OU learn in the same distance-only world, then tests whether they help the agent reach physical goals. **ICM + OU means the existing ICM→OU curriculum:** start with ICM pairs, gradually replace them with OU pairs, and finish with OU. There are ten paired trials per main policy, plus ten trials for each of two replay binnings.

**Follow-up:** [Why do the ICM curves turn or taper near distance 2–4?](fold-location/README.md) Input-scale interventions, exact derivatives and sampling diagnostics.

**Deeper mechanism:** [the first reversal and its subsequent movement](fold-location/training-mechanism/README.md), including pure-ICM regularizer-sampling controls and a replay of the actual optimizer update.

## What happened

| Representation training | Folded runs | Mean physical goal error ↓ |
| --- | ---: | ---: |
| ICM | 5/10 | 0.871 |
| OU | 0/10 | 0.099 |
| ICM→OU | 5/10 | 0.366 |
| ICM→OU, balanced log-size SIGReg | 0/10 | 0.100 |
| ICM→OU, balanced raw-size SIGReg | 0/10 | 0.102 |

These scores use the **same physical action data** to train every planning predictor. ICM→OU beats OU only in seeds **130 and 134**, and loses in the other eight. Its adjusted two-sided sign-test $p=0.219$: no significant advantage. The separate validation screen selected seeds **130 and 137**; they give one held-out win and one loss, $p=1.000$. We cannot get a valid benefit test by selecting only seeds 130 and 134 because they won on the final test.

Balancing SIGReg repairs the observed folds, but does not establish better planning than OU. It also does not solve native action-data coverage: balanced models' mean errors rise to **1.24** when the predictor learns from ICM's own physical pairs. Ordinary ICM→OU rises from **0.366 to 1.906** under that change.

Three useful cases to inspect:

- **Seed 136:** ICM visits all 32 bins, yet has a broad reversal and goal error **1.959**. Occupied-bin count alone misses the problem.
- **Seeds 130 versus 131:** seed 130 has no detected fold but flattens at far distances and has error **1.416**. Seed 131 folds below about 2.37 but has error **0.223**. A fold flag alone also misses the geometry and task dependence. Seed 139 is a reminder that a short reversal can still be very damaging; branch collisions matter too.
- **Coverage removals:** removing distances **11.31–16** gives the largest observed average error increase (**+0.028**), followed by **2.83–4** (**+0.022**). Removing **1–1.41**, which almost none of these goal routes require, improves the average by **0.052**. None of the eight adjusted region sign tests is significant, so this is an exploratory ranking, not a proven list of essential bins.

The practical explanation is **ordering and usable action examples, not just more visits**. The results identify specific sampling repairs and data-removal effects; they do not prove which neural optimization mechanism creates every fold. All saved-model and physical-trajectory audits pass: 50 encoders, 180 predictors, 11,520 held-out goal rollouts, and five focused tests. Goals within a seed are repeated evaluations, not extra independent training trials.

## Start with the pictures

Every representation panel plots physical distance against the learned scalar output. A reversal means increasing distance makes the output move backward. Red marks these reversed segments. The gray dashed curve is a reference, not a fitted correction; all learned curves receive only an affine display normalization.

The panel's $R^2$ measures how well a straight-line readout recovers log-distance; closer to 1 is better. It is not a planning score. In the coverage plots, higher histogram entropy means visits are spread more evenly across bins; it does not tell us which bins are visited.

- Ten separate panels each: [ICM](output/figures/representations_icm.png), [OU](output/figures/representations_ou.png), [ICM→OU](output/figures/representations_icm_to_ou.png).
- [All three overlaid for each trial](output/figures/representations_overlay.png).
- [Which curriculum runs win or lose](output/figures/curriculum_wins_and_losses.png), [shared versus ICM-native predictor data](output/figures/planning_by_trial.png), and [concrete goal-reaching trajectories](output/figures/planning_failure_examples.png).
- [ICM sampling in each trial](output/figures/icm_sampling_by_trial.png), [coverage and geometry versus planning](output/figures/coverage_and_geometry_vs_planning.png), and [state/action coverage](output/figures/physical_state_action_coverage.png).
- [Binning changes the curves](output/figures/representations_binning.png) and [the replay distribution and planning errors](output/figures/binning_exposure_and_planning.png).
- [Which regions hurt when removed](output/figures/controlled_region_importance.png), [per-bin associations and goal-route demand](output/figures/bin_associations_and_task_demand.png), and [coverage/error heatmap](output/figures/coverage_heatmap.png).

Open the PNGs to zoom; matching PDFs are in the same folder. [Results and every trial's scores](output/RESULTS.md) · [Plain-language explanation with the math](EXPLANATION.md).

## What the comparison controls

[Protocol](PROTOCOL.md) fixes the seeds, budgets and tests before the new results. The main comparison uses all ten trials. A separate success screen uses different goals from the final evaluation, so we do not select the runs because their final test scores look good.

Each seed shares its encoder initialization, learning budget, physical predictor-training data and evaluation goals across policies. Every encoder gets 4,000 updates. Each frozen encoder gets a separately trained action predictor, then 64 held-out goal tasks. Reported error is the mean remaining physical distance to the goal after 12 decisions. The agent never receives physical distance labels. All goals are reachable by the oracle.

The success screen requires no substantial grid reversal and lower error on 24 separate validation goals. Statistical tests use seeds, not individual goals, as independent observations. The exact sign test asks whether the curriculum wins more often than it loses; paired bootstrap intervals describe its mean error change. The all-trial and screened-subgroup sign tests receive a Holm adjustment. [SciPy's exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) documents the test used here.

For the binning comparison, only the examples sent to SIGReg are reweighted; alignment and the final OU phase stay fixed. For the coverage-removal comparison, the OU encoder stays fixed and one region is removed from predictor training. Removing a region changes the available examples while keeping the number of training pairs and updates constant. These are two different interventions.

ICM costs an additional 192,000 actor-training transitions per seed, followed by 49,152 collection transitions including burn-in, of which 16,384 pairs are retained. Its frozen data are reused across the ICM-containing arms. The common physical predictor data contain another 16,384 retained pairs. OU is a passive Gaussian pair generator, so these comparisons do not match actor-training compute or physical transition dynamics. OU-native action prediction is undefined and is not fabricated.

## Reproduce or inspect

From repository root:

```bash
pip install -r behavior-policy-experiments/qualitative-study/requirements.txt
# Rebuild the published figures and audit the saved models and trajectories:
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/report.py
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/verify.py

# Run a fresh cohort in a new output directory:
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/run_trials.py --output /tmp/distance-policy-study
OPENBLAS_NUM_THREADS=1 python behavior-policy-experiments/qualitative-study/report.py --output /tmp/distance-policy-study
```

Use `--quick --output /tmp/qualitative-smoke` for a reduced execution check; it is not part of the scientific results. Rerun the same training command in the same checkout/output directory to resume completed stages; partial encoders save every 500 updates. The source hashes and protocol hash guard against silently changing a resumed experiment. The recorded environment used Python 3.12.3 on CPU.

`run_trials.py` trains and saves the data/models. `report.py` produces the plots and tables. `trial_stats.py` implements the seed-level comparisons. `verify.py` independently replays saved physical actions, checks equal goals and data-removal budgets, and recreates curves from saved encoders. Five focused tests cover pairing, subgroup selection, multiplicity adjustment, replay and region removal; run them with `pytest behavior-policy-experiments/qualitative-study/test_study.py` after installing pytest.

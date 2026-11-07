# Fresh-seed confirmation: policy ties and ICM exposure

**Completed and verified:** 60 fresh paired seeds, 240 newly trained exploration actors, 540 encoder fits and 1,080 planning evaluations. A separate untrained-encoder check adds 60 predictor fits.

**Start here: [concise results and explanation](output/SUMMARY.md).** [Graphs](output/FIGURES.md) · [Fixed protocol](PROTOCOL.md) · [Published code/time lock](output/LOCK.json)

## What held up?

All seven matched policies pass practical equivalence for whole-range/near/far recovery and shared-data physical goal error. We fixed **±0.001 R²** and **±0.005 distance units** before training. The latter is 2% of one action step. The joint claim passes the four-claim Holm correction: **p = 1.08 × 10⁻¹⁹**. TOST's paired-mean assumptions apply; this does not prove exact equality.

## The stronger mechanism result

Keep ICM's actor, transition pairs and initial encoder weights fixed. Change only which observations SIGReg sees; keep its formula and weight unchanged.

| SIGReg observations | Recovery R² | Folds / 60 | Physical goal error |
| --- | ---: | ---: | ---: |
| ICM's native observations | 0.6611 | 30 | 0.8739 |
| Broad common observations | 0.9975 | 0 | 0.0760 |

Recovery and reaching improve in all 60 seeds; all three planned mechanism tests pass correction. The broad pool supplies additional real observations through controlled resets. It is a data intervention, not a free autonomous exploration improvement.

The secondary untrained baseline has recovery 0.8421, 26/60 folds and goal error 0.4479. Matched training therefore improves beyond the starting geometry. [Baseline results](output/untrained_baseline/RESULTS.md).

## What remains open?

Policy-specific predictor-data planning equivalence is not established at the same tolerance. The next useful test keeps one encoder fixed while changing predictor action data. The present result identifies a finite-training exposure effect; it does not extend the paper's general identifiability guarantee to nonlinear systems.

Starting observations and SIGReg exposure match; successor distributions and stationarity do not. These are physical OU action controllers, not Gaussian-latent OU dynamics. [Research value](RESEARCH_VALUE.md) · [One-dimensional caveat](ONE_DIMENSION_CAVEAT.md) · [Statistical assumptions](ASSUMPTIONS.md) · [Geometry/planning link](PLANNING_BRIDGE.md)

## Reproduce

With Python 3.12, from the repository root:

```sh
python3 -m pip install -r behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/requirements.txt
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/run_confirmation.py --run --workers 6
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/finish_confirmation.py --workers 6
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/untrained_baseline.py --train --analyze --verify --workers 4
python3 behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/write_summary.py
```

The published lock is already present; do not refreeze a completed cohort. Encoder/Adam resumes every 500 updates. Completed actor/predictor states are reused; interrupted actor/predictor fits restart deterministically. Regenerable source caches are ignored by Git. See the protocol for the historical gates and pre-training freeze procedure.

[Main verification](output/VERIFICATION.json) covers all 420 sources, 540 geometries, 1,080 physical-planning replays and 84 primary TOST boundary checks. [Fresh full/resumed gates](output/FRESH_REPLAY_GATES.json) · [Baseline verification](output/untrained_baseline/VERIFICATION.json)

# Targeted additions to the behavior-policy zoo

Fixed before any full-size result from these policies. The old zoo results have been inspected; policy design is informed by them. Discovery reuses seeds 8–27; confirmation uses untouched seeds 28–47 with a selection rule fixed below. No policy is required to win.

## Unchanged experiment

Same triangle, one physical distance, bounds 1–16, actions approach/stay/retreat by 0.25, uniform initial physical distance, 128 burn-in steps, 64 recorded steps, 256 trajectories (16,384 pairs). Same encoder, loss, coupled 512-pair alignment/SIGReg batches, 4,000 Adam updates and learning-rate schedule. Original seed-specific initialization and pair-index streams. No distance labels reach the encoder. Common-grid R² is the primary outcome; folds and data statistics are secondary. This is not a planning evaluation.

New controllers read only the observation's height and their own memory/randomness. They exploit the known height = 1/d renderer and Gaussian target, an explicit engineering prior. All reject actions whose proposed displacement would leave the bounds, substituting stay rather than deliberately inducing a bounce. The actual environment is unchanged.

All seven first spend 64 existing burn-in steps moving toward a randomly drawn target apparent size (truncated log-normal distance, center d=4). This is an active collection policy, not a changed reset distribution or extra interaction. The remaining 64 burn-in and all 64 recorded steps use the rules below.

## Seven fixed policies

| Name | Recorded behavior | Intended improvement |
| --- | --- | --- |
| gaussian_walk | Symmetric ±0.25 Metropolis walk, target log d ~ N(log 4, 0.50²), truncated to bounds | Match SIGReg's Gaussian preference without reflection cycles |
| gaussian_log_diffusion | Same target, proposal activity min(1,(d/8)²); include activity ratio in Metropolis acceptance | Approximately equalize movement variance in log distance |
| gaussian_wide_diffusion | Same mobility, target log standard deviation 0.65 | More distant views with a broader target |
| gaussian_target_fast | Track an internal OU log-size target, SD 0.55, correlation 0.98 | Combine Gaussian-like occupancy with frequent movements |
| gaussian_target_slow | Same, correlation 0.995 | Test slower target motion and steadier tracking |
| gaussian_deficit | Softmax over feasible actions using desired Gaussian log-bin mass / accumulated bin counts; temperature 0.25, 10% feasible random actions; target SD 0.55 | Coverage shaped toward a Gaussian rather than uniform bins |
| gaussian_start_random | After the same initial target approach (SD 0.55), uniform feasible actions | Check how much the burn-in steering alone buys |

For state-dependent Metropolis proposals, log acceptance is log π(d') − log π(d) + log m(d') − log m(d), with π the log-normal physical-distance density (including its 1/d Jacobian). The target is stationary on each reachable displacement lattice, not a claim of exact continuous Gaussian occupancy after finite burn-in.

## Discovery comparisons: paired seeds 8–27

Run each new policy pure and with the original 25%→75% OU handover: 280 fits. Compare with all existing physical policies in the same protocol and the fixed-OU reference. References for targeted tests: fixed OU; random in the same protocol; the best old pure policy (count) or best old handover policy (approach-biased), chosen from the already known old tables.

Report mean R², SD, fold count, paired differences and percentile paired-seed bootstrap 95% intervals (20,000 resamples, seed 20261105). Two-sided exact sign tests target win probability among non-ties (difference tolerance 1e-10), with Holm over all 14 new-policy/protocol contrasts within each of the three reference families. Mean intervals are unadjusted exploratory summaries, not tests of the same null as sign tests. Report every candidate, including failures. Retain visitation shape, stay fraction and movement statistics to detect trivial near-static successes.

## Fresh confirmation: seeds 28–47

Select the top **two new policies per protocol** by discovery mean R²; ties are broken by policy name. Do not filter on folds, p-values or selected winning seeds. Freeze the selected names before running confirmation. Run those selected policy/protocol combinations on the 20 fresh seeds, plus fixed OU, pure count, handover approach-biased, and random in both protocols (nine conditions total).

Confirmatory primary family: the four selected policy/protocol contrasts against fixed OU, exact two-sided sign tests with Holm across four. Require a positive mean difference and adjusted p < 0.05 to say a candidate consistently outperforms OU on this representation metric. Also report paired mean intervals and every seed. Secondary comparison families: selected candidates against random, and against the best old source in the same protocol, Holm within each four-contrast family. These test families do not confer planning superiority or universal identifiability.

The original 20-seed baseline means cannot replace fresh-seed baselines. Both pure and handover variants use the same physical source pairs within each seed. Policy/source randomness and encoder initialization remain coupled by seed as in the zoo; there is no new crossed-variance claim.

## Verification and publication

Before discovery, original fixed OU and random/handover paths must reproduce a prior seed-8 checkpoint exactly. Unit checks cover Metropolis detailed balance, proposal feasibility, deterministic regeneration, data shape and no observed boundary reflections. Persist partial encoder/optimizer checkpoints every 1,000 steps and frozen learned count policies for confirmation; do not retrain sources on resume.

Publish policies, all fitted encoders, per-seed metrics, source hashes, selection record, concise README, comparison plots and verification. Preserve the original zoo's protocol/catalog and add this as a linked extension. No hyperparameter changes after discovery results; any later redesign requires a separately labelled follow-up.

# Broader-coverage follow-up

Designed after inspecting discovery results from the Gaussian-targeted controllers: those policies produced near-Gaussian but narrow data and did not beat fixed OU in the completed discovery conditions. This follow-up is separate; their controller settings and original confirmation on seeds 28–47 are unchanged. No confirmation results from that study inform this design.

## Fixed candidates

Same physical environment, observations, actions, initialization, 128-step burn-in, 64 recorded steps and 4,000-update encoder protocol as the existing zoo. The new policies do **not** perform the 64-step Gaussian-target approach. All policies inspect only views and policy memory, infer 1/height as needed, and avoid proposing boundary overshoot. Bounds and movement size remain 1–16 and 0.25. No extra interactions or world coordinates.

| Policy | Fixed rule |
| --- | --- |
| guarded_bias | Approach/stay/retreat probabilities (0.36, 0.33, 0.31); invalid movement becomes stay |
| gentle_bias | Same guard, probabilities (0.35, 0.33, 0.32) |
| adaptive_bias | Stay probability 0.33; approach−retreat bias 0.07 tanh((log d−log 4)/0.8), pointing toward d≈4 |
| guarded_sweep | Persistent sweep, turn at d≤1.25 or d≥15.75; 15% guarded uniform actions |
| bias_sweep_mix | At each step choose equally between the guarded sweep and guarded_bias; shared sweep direction memory |
| no_escort_log_diffusion | The earlier SD-0.65 Gaussian Metropolis/log-diffusion controller, but skip its initial target approach; ordinary uniform world resets remain |

These target broad early training support and ordered movement, informed by the old zoo's biased/sweep successes. They are not claimed to dominate OU in advance. The no-escort comparison is a discovery-stage ablation of the earlier controller, not a fresh paired confirmation of that ablation.

## Discovery, selection and untouched confirmation

Discovery: seeds 8–27; all six sources pure and with the original 25%→75% OU handover (240 fits). The earlier zoo and Gaussian-targeted discovery are known; discovery comparisons are exploratory. Compare the full original zoo and the earlier seven additions on these same seeds.

Select the top two new policies within each protocol by discovery mean shared-range R², ties by name. No fold filtering, winner-subset tests or hyperparameter changes after discovery. Record the selected names before confirmation.

Confirmation: **fresh seeds 48–67**, disjoint from both prior blocks. Run the four selected policy/protocol combinations, fixed OU, pure count, handover approach-biased, random in each protocol, and the preceding Gaussian-targeted extension's discovery-best controller in each protocol: eleven conditions, 220 fits. Choose those two reference names solely from its frozen discovery selection, not from confirmation performance.

Inference matches the first extension: paired mean differences with 20,000 paired-seed bootstrap resamples (seed 20261105), win/loss counts, two-sided exact sign tests among non-ties (tolerance 1e-10), Holm over the 12 discovery or four confirmation contrasts within each reference family (fixed OU, same-protocol random, best old same-protocol policy, best preceding addition). A claimed consistent improvement over OU requires positive observed mean, a majority of positive paired differences, and Holm p<0.05. Sign tests concern win probability, not the mean; intervals are unadjusted summaries. Report effect sizes, SD, minimum, fold counts, broader-OU R² and visitation/motion statistics. This is not a planning claim.

## Checks and reproducibility

Original pure OU and handover random at seed 8 must reproduce stored weights exactly. Regenerate sources deterministically; preserve source hashes and partial checkpoints. Check physical feasibility, mixture rules and the no-escort ablation. Reuse the first extension's fitting/statistical implementation with this fixed catalog, own output folder, and disjoint confirmation block. Publish failed candidates and compare confirmed winners with old best policies, not just OU.

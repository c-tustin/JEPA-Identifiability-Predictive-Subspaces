# Why do curves bend near distances 2–4?

Exploratory follow-up to the ten plotted seeds, specified before the new gain controls on 2026-11-02. These are the same diagnostic seeds 130–139, not new independent validation seeds.

The existing encoder applies `tanh(4*x)` before two tanh hidden layers. With inverse-distance observations, nonzero absolute input coordinates are 0.5/d, 0.866/d and 1/d. Their tanh arguments cross magnitude 1 at d=2, 3.46 and 4. These are smooth scale landmarks, not hard thresholds or mathematical fold points.

Measure all saved ICM, OU and balanced-log curriculum curves: turning points, peak sensitivity to log-distance, far-distance half-sensitivity point for nonfolded curves, hidden-layer saturation and cancellation between signed contributions to the output derivative. Compare these with source visit density. Distinguish actual extrema from monotone flattening.

Retrain pure ICM and pure OU at preprocessing gains **2 and 8**, for all ten seeds. Reuse gain-4 checkpoints after checking an exact full-training reproduction for folded ICM seed 131. Keep saved ICM/OU data, network weight initialization, pair indices, independent regularizer batch indices, optimizer, 4,000 updates and all other settings fixed. Never retrain the behavior actor. The intervention changes input preprocessing and therefore initial function/optimization conditioning; it does not isolate hidden-layer saturation alone.

Store each gain in a separate output directory and preserve the original models and data. Plot all ten gain comparisons, including controls that contradict the hypothesis. No planning predictors need refitting for this location diagnostic. Report fold occurrence and location separately: losing a fold is not a shifted fold. Interpret these as exploratory interventions, without post-hoc significance claims or selecting favorable trials.

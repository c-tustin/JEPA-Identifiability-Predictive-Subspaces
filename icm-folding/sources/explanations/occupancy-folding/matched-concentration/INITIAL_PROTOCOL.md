# Concentration-matched secondary-mode comparison

Fixed before inspecting new training outcomes, 2026-11-14. The previous goal turn made progress by identifying the confound: native seed 131 has 84.6% of pooled endpoints below distance 2, while previous synthetic bimodal/trimodal distributions have 25%/17.5%. Those comparisons did not isolate mode structure.

## Primary question

With the near-object mass, near-object profile, support, background, model, alignment pairs and stochastic draws held fixed, does splitting the far tail into additional bumps change folding? Separately, does increasing near-object mass change folding for the same component shapes?

## Data and controlled measures

Use all ten existing stored ICM sources, seeds 130–139. Reuse the gain-4 scalar encoder, 4,000 updates, original cosine schedule, 512 alignment pairs, 1,024 regularizer endpoints and SIGReg weight 0.1. There is no OU phase and no new actor training. These are previously inspected diagnostic sources, not independent confirmatory seeds.

Pool both endpoints of each stored dataset. Partition log-distance [0, log 16] into 128 bins. All interventions use **only these actual stored endpoints** and retain every occupied bin and every stored point in their population support. Within each bin, the conditional distribution over stored endpoints is uniform and unchanged.

Let P be the native bin distribution; U be uniform mass over the occupied bins; N be the native conditional distribution below distance 2. The target bin measure is

    R_(a,k) = epsilon U + (a − epsilon U_near) N
                            + (1 − a − epsilon U_far) T_k,

with epsilon = 0.02 and a in {0.25, 0.50, 0.875, 0.9375}. Thus every target at a fixed a has exactly the same near-bin probabilities, a mass a below distance 2, identical background epsilon U, and identical stored-point support.

Four normalized far components T_k, all supported at distances >=2:

1. Native: the source's original conditional far-bin distribution.
2. Diffuse: uniform over occupied far bins (no deliberately added far bump).
3. One bump: a single log-concave discrete Gaussian-shaped bump, moment-matched to component 4.
4. Two bumps: equal nominal log-Gaussian bumps at physical distances 4 and 8, log-width 0.12, restricted to occupied far bins.

For the one/two-bump pair, solve the one-bump exponential-quadratic coefficients to match the **actual endpoint** mean and second moment of log-distance under the two-bump tail. This accounts for uneven stored observations within bins. Require errors <1e-9 and a negative quadratic coefficient (a unimodal envelope). Consequently the complete one/two target measures have matched first two log-distance moments as well as identical near-object mass/profile, support and background. Source sampling may have holes and local histogram fluctuations: labels describe designed bump envelopes, not a claim of an exact count of every tiny empirical local maximum.

The U background is identical across conditions for each source; supports may differ between sources. No nonexistent bins are filled. Native near profiles are preserved as the N component; at fixed a, the entire resulting near profile is identical. Across concentration levels only the mixture coefficients change, while N, U and T_k stay fixed.

## Primary estimator and bridge controls

All primary cells use exactly the same native paired endpoint minibatches as the established experiment. Importance weight each endpoint by R_bin/P_bin and use normalized weighted sine/cosine averages inside SIGReg. Alignment pairs and both minibatch streams remain matched across cells. Record the actual average weighted bin masses and effective sample size. The target population masses are exact; self-normalized finite batches need not attain them exactly.

Per source, add (i) an ordinary native-paired reproduction, whose final weights must agree exactly with the preceding native baseline, and (ii) a floor-only control R=0.02 U+0.98 P. These distinguish changing concentration/modes from simply adding the modest background.

## Replay robustness at the decisive concentration

For all four tails at a=0.875, repeat training using unweighted SIGReg on **stratified replay**, with exactly 896 near and 128 far stored endpoints in each batch. Common random-number streams ensure the near observation sequence is identical across all tail structures. Far quantile draws are also matched, with only their target CDF changing. The near/far population conditionals are the same as the corresponding weighted targets.

Also replay the native target P using a rounded native near count per batch, documenting the <1/1024 concentration rounding. This checks the replay estimator against native paired batches. Replay changes finite-batch estimation and endpoint correlations; it does not add observations.

## Planned run counts and readouts

Ten sources × [16 concentration/tail weighted cells + 2 bridge controls + 4 concentration-0.875 replay cells + 1 native replay control] = **230 new encoder runs**. All seeds/cells are retained. No selecting seeds or concentration levels based on observed fold outcomes.

Save raw curves, analytic log-distance slopes, every sign-changing slope root, sign-invariant backward variation, uniform-log-grid R², and output standard deviation every 100 updates. Save selected model states for reconstructing curves. Reevaluate final roots independently on 10,001 points, including outside the source support up to distance 16; distinguish in-support and extrapolation turns. A finite-grid no-turn result is not a global injectivity certificate.

Verify population mass, near-profile equality, background/support, matched first two moments, same initialization/index streams, native reproduction, and replay near-draw equality. Plot actual controlled distributions on a logarithmic probability axis so small far bumps remain visible; plot fold rates versus concentration; show every source's decisive-condition outcome, representative curve snapshots/root histories, estimator agreement, and effects of adding only the background. Report all failures and distribution-estimation deviations.

## Interpretation limits

The one/two-bump contrast isolates higher-order tail shape from near concentration and first two log-distance moments in this fixed setup. It does not prove a universal law about the abstract number of modes. Concentration interventions alter the regularizer's target measure; alignment dynamics remain native. Weighted-moment versus replay agreement strengthens a measure-based explanation; disagreement implicates finite-batch estimation/optimization as well. The native/floor bridge controls must be inspected before claiming that the controlled family reproduces native folding. High near concentration is not assumed sufficient to produce a fold.

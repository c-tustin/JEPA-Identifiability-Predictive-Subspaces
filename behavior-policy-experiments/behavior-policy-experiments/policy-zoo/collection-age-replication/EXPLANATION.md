# What would this prove?

The experiment tests a claim; it is not designed to make early ICM win.

## Three different clocks

| Clock | Meaning | What changes here? |
| --- | --- | --- |
| Policy training | Updates that teach the ICM/count actor how to act | Fixed at 600; actor is then frozen |
| Collection age | Actions since an environment reset | 0, 16, 32, 64 or 128 actions discarded |
| Encoder training | Updates that teach JEPA a representation from stored pairs | 4,000, then 4,000 more in continuation controls |

These are experiment settings, not extra coordinates in the world. Physical distance remains the only varying scene coordinate. In this study “early ICM” means early **collection**, using an already trained actor.

## Comparing windows asks whether their data are useful

Let $D_a$ contain 64 steps beginning at collection age $a$, from the same 256 parallel trajectories. Each window has 16,384 real transitions. We fit

$$
\theta_a=\operatorname{Train}(D_a,\theta_{\mathrm{init}},I),
$$

where the initial encoder and minibatch-index stream $I$ are shared across conditions within a seed. The loss, architecture and training budget are fixed.

If early data give better recovery and reaching than late data, that supports a **collection-phase rescue** in this setup. It does not show that early ICM beats random, count or OU. Those are separate planned comparisons. It also does not show that one good encoder becomes folded later: these encoders were trained separately.

“Same budget” refers to the retained data and representation/predictor training. A later window requires more preceding environment actions if collected separately. Learned ICM/count actors also have an exploration-training cost that random actions do not have. This is not an equal-total-compute algorithm benchmark.

## Continuing one encoder tests a stronger claim

First train on $D_0$, obtaining encoder $\theta_E$ and Adam state $s_E$. Then create two branches:

$$
\theta_{E\to E}=\operatorname{Continue}(D_0,\theta_E,s_E,I'),
$$

$$
\theta_{E\to L}=\operatorname{Continue}(D_{128},\theta_E,s_E,I').
$$

Both use the same saved weights, Adam state, additional training budget, learning-rate restart and index stream. Only the supplied dataset changes. All seeds enter both branches, including any imperfect early encoders.

If switching to late data introduces folds or worsens reaching relative to continued early data, this supports an effect of late data on the same initially shared representation. If late-window fits from scratch fold but the continuation does not, the claim “late data creates folds in a good encoder” is not established; initialization or training history could protect it.

## Identifiability and planning are different tests

- **Affine recovery:** is the learned number nearly a straight-line function of log-distance? This is closest to the paper's linear-identification target.
- **Order and aliases:** is the map monotone, or do distinct distances receive the same code? Nonlinear monotone recovery can preserve information despite lower affine R².
- **Goal-reaching:** after freezing the encoder, can the action-conditioned predictor and planner reach the physical goal?

The primary task score is actual terminal distance error on 64 reachable goals:

$$
E=\frac1{64}\sum_{j=1}^{64}|d_{\mathrm{end}}^{(j)}-g^{(j)}|.
$$

Shared predictor data and identical goals hold the task setup fixed across encoders. Native predictor data adds each policy's state/action sampling effects. The planner receives observations through their encodings, not distance labels. Every evaluated goal has zero oracle error. These scores assess this particular predictor/planner, not the best theoretically possible controller. A nonlinear but one-to-one code might plan well with a different predictor.

The inherited shared dataset samples proposals that stay inside the distance bounds; outward reflected actions at the two grid endpoints are not included in that predictor source. Rollouts still use the original reflecting environment. The same source is used for every encoder, so it remains a controlled comparison, but it is not complete boundary-action coverage.

## What remains a hypothesis

Late ICM may concentrate observations near the object, and the finite network may distort less-constrained distances while fitting the frequently visited region. But collection age also changes reflection encounters, action frequencies and pair correlations. Those are measured together, not separately randomized. A successful collection intervention does not isolate concentration as the sole cause.

These bounded, transient physical datasets are outside the exact stationary Gaussian-OU world of the paper's forward guarantee. Empirical recovery and task improvements are not a new population identifiability theorem. The fresh-seed study tests reproducibility and the stronger continuation claim for this implementation, with one correction across all 19 planned tests.

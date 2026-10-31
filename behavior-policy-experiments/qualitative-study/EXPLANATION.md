# Why coverage can improve while planning gets worse

There are three separate questions: does the encoder preserve distance order, does the predictor learn the effects of all actions, and does the planner need the regions that were sampled? The plots in this study check each question separately.

## What the representation loss rewards

Write the learned coordinate as $h(d)$. The alignment part of our actual training objective is

$$
\mathcal E(h)=\mathbb E_{(d,d')\sim q_\pi}\left[(h(d')-h(d))^2\right].
$$

Here $q_\pi$ is the distribution of collected transition pairs under the behavior policy. This loss encourages similar outputs for connected observations. SIGReg prevents a constant output by encouraging a Gaussian output distribution. It checks the distribution of outputs, rather than the physical order of observations. Neither term directly asks the agent to distinguish two physical goals or to predict the effect of each action.

Changing the behavior policy changes **which transitions connect states, how often those connections appear, and the output distribution being regularized**. A histogram only records state visits. Two datasets can have identical state histograms but very different transitions. Staying still at every distance, for example, gives perfect state coverage while making the alignment loss zero for every encoder.

A useful mathematical approximation replaces SIGReg with exact zero-mean, unit-variance constraints. Minimizing this energy then becomes a weighted graph eigenvector problem. The edge weights come from the symmetrized transition-pair distribution, rather than just the node histogram. This explains why occupancy alone cannot determine what is learned. It is an interpretation of a simplified objective, not a proof that our neural optimizer must learn a fold.

For a stationary standardized Gaussian OU latent $z$, Hermite modes have one-step correlations $\rho^k$. With $0<\rho<1$, the first nonconstant mode, $h(z)=z$, has the smallest alignment energy, $2(1-\rho)$, among zero-mean, unit-variance functions. This follows by expanding $h$ in the orthonormal Hermite basis: its energy is $2\sum_{k\geq1}a_k^2(1-\rho^k)$ with $\sum_{k\geq1}a_k^2=1$. The Hermite spectrum is described in [Andersson and Sjögren, Chapter 1](https://kadamandersson.github.io/OU.pdf). This gives OU a particularly clear ordering signal. Our finite datasets, approximate SIGReg and neural optimization do not satisfy an exact population theorem. The ICM physical trajectories also differ from OU in dynamics, boundaries and sampling distribution, so this experiment does not isolate curiosity as the sole cause.

## What a fold breaks

Suppose two distinct goals satisfy $h(g_1)=h(g_2)$. A deterministic planner that receives only the current encoding and goal encoding starts with the same inputs for either goal. In this deterministic world, it follows the same trajectory and ends at the same point $D$. Therefore

$$
\frac{|D-g_1|+|D-g_2|}{2}\geq\frac{|g_1-g_2|}{2}.
$$

It cannot reach both goals accurately. This is an exact ambiguity argument for equal encodings. Our plotted near-collisions suggest a practical problem but are not exact equalities, and finite numerical planners can respond differently to nearby encodings. The grid reversal check is a diagnostic, not a proof of every possible collision or planning failure.

Folding is also not necessary for poor planning. A monotone encoder can squeeze a large physical interval into a tiny output interval. The predictor and planner must then resolve small latent differences to choose the right action. The sensitivity plot and actual trajectories help reveal this failure.

## Why binning changes more than a coverage score

The observation's apparent height is $s=1/d$. Uniform sampling in log-size bins approximates a density proportional to $1/d$ over physical distance. Uniform sampling in raw-size bins approximates a density proportional to $1/d^2$. These follow from the change-of-variables derivatives $|d\log s/dd|=1/d$ and $|ds/dd|=1/d^2$.

Thus raw-size balancing favors nearby distances more strongly. The finite replay implementation only equalizes occupied bins in expectation, and both endpoints of each sampled pair enter SIGReg. Its actual exposure is plotted rather than assumed to match either ideal density exactly. Replay adds repeated examples, not new states or missing actions.

These are changes to the measure under which SIGReg is applied. They can alter the learned geometry even with the same alignment pairs. A more uniform histogram is not automatically a more useful physical coordinate.

## How the figures distinguish explanations

- Compare **ICM representation panels** with **ICM sampling by trial**. A fold inside a poorly sampled region is consistent with weak constraints there; a fold inside a well-sampled region shows why occupancy alone is insufficient. Association does not establish which optimization step created the fold.
- Compare **shared and native predictor errors**. Shared data holds action training data fixed. Native ICM data includes the policy's state/action preferences. A large native-data penalty points to this second problem even when the encoder is useful.
- Compare **replay binnings** under shared predictor data. This isolates their effect through the learned encoder, while retaining the same training budget and evaluation tasks.
- Inspect **controlled region removals**. They freeze the OU encoder and remove predictor examples from one region at a time. These estimate the effect of those removals for our particular goals and finite predictor training; they do not rank individual bins universally.

We can diagnose geometry, sampling and action coverage with this study. Establishing why the optimizer chooses a particular fold would additionally require interventions on transition connectivity and optimization while holding state occupancy fixed.

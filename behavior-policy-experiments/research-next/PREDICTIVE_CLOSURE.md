# Next experiments: Predictive Closure and nonlinear distance

**Status:** research directions and a derivation, not completed neural-training results. Execution has moved to the cloud task **Predictive Closure and nonlinear JEPA extensions**, thread `01a1761e-d72b-767b-af9b-a567f0c40011`. This handoff preserves the same triangle sensor and one physical distance coordinate.

## What the new paper adds

[Levy, Lidayan and Goldberg, Predictive Closure](https://arxiv.org/abs/2610.27364v1) asks which *sets* of observation features JEPA retains. An unpredictable feature can still be useful because its current value helps predict another retained feature. The paper also studies supplying information directly to the predictor. [Official experiment code](https://github.com/Mosh0110/predictive-closure).

The mathematical qualifications matter. Appendix A.1–A.2 fixes latent dimension, numerical precision and Gaussian marginal distribution, and assumes a representable conditional mean with isotropic conditional Gaussian error whose scalar variance is independent of the current state. Under those conditions, optimized squared error and predictive information rank encoders equivalently. Without the conditional Gaussian assumption, the paper gives an information lower bound, not that equivalence. Finite-frequency SIGReg does not certify the ideal marginal constraint.

## Priority 1: information that a mean predictor misses

Keep the original log-distance coordinate and renderer:

$$
d=4\exp(0.7z),\qquad x=V/d.
$$

Compare two scalar transition laws, both with stationary standard-Gaussian $z$:

$$
\text{Ordinary:}\quad z'=az+\sqrt{1-a^2}\epsilon,\qquad a=0.5,
$$

$$
\text{Random sign:}\quad z'=Srz+\sqrt{1-r^2}\epsilon,\qquad r=0.95,
$$

where $S$ is an independently chosen uniform sign and $\epsilon$ is independent standard Gaussian. This sign is an action/input, not another physical world coordinate. Each sign-conditioned transition preserves the Gaussian marginal, so their mixture does too.

Without the sign, the best mean prediction in the second world is zero. Its optimized squared error is 1, although the next magnitude depends on the current magnitude. In the ordinary world, squared error is $1-a^2=0.75$ and mutual information is

$$
I_{\rm ordinary}=-\tfrac12\log(1-a^2)\approx0.1438\text{ nats}.
$$

For the random-sign world, independence of $S$ and $z$, followed by the information chain rule, gives

$$
I(z;z',S)=I(z;z'\mid S)=-\tfrac12\log(1-r^2),
$$

$$
I(z;z')=I(z;z',S)-I(z;S\mid z')
\ge-\tfrac12\log(1-r^2)-\log2
\approx0.4708\text{ nats}.
$$

Thus more information can coexist with worse mean prediction outside the paper's conditional Gaussian assumption. This derivation compares **two scalar transition laws**. It is not a counterexample to encoder selection within one fixed world and does not refute the qualified theorem. Continuous-information calculations also need a separate resolution check before identifying them with the paper's finite-precision quantities.

**Experiment:** verify marginal moments and information using stable Gaussian-mixture log densities on independent seeds; then compare mean-only, sign-conditioned and mixture-distribution predictors. Use matched budgets, held-out likelihood, distance recovery and checks for folds/compression. Compare prediction of magnitude as well as signed distance. A model can retain useful uncertainty even if its conditional mean is uninformative.

## Priority 2: actions as side information in the existing nonlinear world

[A sharper control and derivation](ACTION_INFORMATION.md) separate the information carried by a state-dependent policy's action from its physical effect. The actuator-off control remains unrun.

Train the encoder jointly with predictors that receive either current action, a shuffled action, or no action. Use matched source states/actions to separate information from coverage. The shuffled-action control distinguishes helpful action information from extra model inputs. No additional physical state is needed. Test whether representation geometry and physical planning improve, rather than reporting only latent prediction error.

This is related to the paper's side-information proposal, but our single fully observed physical coordinate does not reproduce its competition among independent image features. Do not label the experiment a full paper reproduction.

## Priority 3: learn drift at multiple observation intervals

Use independently noisy endpoints to learn an action-structured drift model at several short intervals. Compare its finite-time bias and sampling noise with the [existing derivation](../nonlinear-control/DRIFT_ESTIMATION.md). Evaluate physical control as well as coordinate recovery. An explicit information-preservation condition is still required for the [identification lemma](../nonlinear-control/PROOF.md); small empirical loss alone does not establish one.

## Additional direction: make capacity pressure explicit

[A four-symbol distance representation](FINITE_CAPACITY_DISTANCE.md) provides an exact finite-capacity reference for feature-set competition and side-information steering. It is proposed work, with analytic reference values; no training result is claimed. Unlike the current continuous scalar pilot, it deliberately prevents the encoder from retaining every distance.

## Cloud handoff and evidence

The cloud task was launched from branch `behavior-policy-experiments`, after commit `eb5ba66`; the simplified planning summary was subsequently pushed as `937018a`. Fetch the latest branch before creating `codex/predictive-closure-extensions`. Read the upstream source/license and record the exact revision before adapting code. Keep the paper's assumptions, our hypotheses, simulator privileges and actual results separate.

The original local task verified that the cloud continuation exists and received the research instructions. Repository access, experiment execution and results must be verified from that task's actual outputs; launch alone does not establish completion. Keep a cloud `RUN_STATUS.md` with exact commands, saved evidence and remaining questions. Continue the broader research objective after the first diagnostic rather than treating one analytic example as completion.

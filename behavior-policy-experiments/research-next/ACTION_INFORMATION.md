# Next control: does the action help prediction, or actually control the world?

**Status:** derivation and proposed controls, not completed experiments. Keep the same triangle sensor and one physical distance coordinate. This complements the [signed-OU test](SIGNED_OU_EXTENSION.md) and the cloud predictor experiments.

## Why this matters for Predictive Closure

The paper's Appendix A.6 correctly writes the ideal side-information objective as minimizing $H(Z'\mid Z,A)$, or maximizing $I((Z,A);Z')$ when its assumptions hold. The information chain rule gives

$$
I((Z,A);Z')=I(A;Z')+I(Z;Z'\mid A).
$$

Thus it is not generally enough to compare only conditional mutual information $I(Z;Z'\mid A)$. Even with a fixed unconditional latent marginal, $H(Z'\mid A)$ can change across encoders. This distinction is already compatible with the paper; it is not a correction to its stated A.6 objective.

A behavior policy can make actions informative about the current observation. Supplying such actions may help prediction by revealing state information omitted by the encoder, separately from any physical effect of the actuator.

## A squared-error identity requiring no Gaussian assumption

For any square-integrable scalar prediction target $Y'$, let

$$
R_Z=\min_g E[(Y'-g(Z))^2],\qquad
R_{Z,A}=\min_g E[(Y'-g(Z,A))^2].
$$

The conditional expectation is the optimal mean predictor. Applying the total-variance identity conditionally on $Z$ gives

$$
\boxed{R_Z-R_{Z,A}
=E\!\left[\operatorname{Var}(E[Y'\mid Z,A]\mid Z)\right]\ge0.}
$$

This gain establishes predictive usefulness, not a causal action effect. An action that reveals an omitted part of the current state can reduce prediction error even with the actuator turned off.

## One-distance example to test

Use the ordinary Gaussian AR log-distance world $W'=\rho W+\sqrt{1-\rho^2}\epsilon$, with $0<\rho<1$, and the usual $d=4\exp(0.7W)$ triangle. The simulator ignores the action. Choose a state-dependent policy label $A=\operatorname{sign}(W-c)$ with $c\ne0$, and a folded representation retaining $|W-c|$ (optionally Gaussianized by its continuous CDF).

The representation alone cannot distinguish $W=c+m$ from $W=c-m$. The action label distinguishes these branches. The next folded magnitude has folded-normal centers $(\rho-1)c+\rho m$ and $(\rho-1)c-\rho m$, whose absolute values generally differ. Hence the label can improve prediction even though intervening on it changes no transition. This is a diagnostic example with a deliberately supplied folded encoder, not a claim about what unconstrained JEPA will learn.

## Matched cloud controls

1. **Actuator off:** compare true state-dependent action labels, independently shuffled labels, and no labels. Use identical observation pairs and parameter/update budgets. Any population gain is information from the policy, not physical control.
2. **Actuator on:** repeat with the known physical action effect, preserving state coverage and conditional action variation. Randomized actions distinguish intervention effects from observational correlations.
3. Evaluate both held-out prediction and actual physical goal error. Supplying an action label during training does not guarantee that a planner can reconstruct a missing state feature when choosing its next action.
4. State which information is available before action selection. A policy-provided label computed from the raw observation is privileged side information if a downstream planner sees only the latent; do not silently give it to one controller but not another.

This control is a concrete bridge between the paper's side-information analysis, behavior-policy differences, and controlled-world-model identification. It does not add a world coordinate or establish a new identification theorem.

[Paper, Appendix A.6](https://arxiv.org/abs/2610.27364v1) · [Existing planning interpretation](../nonlinear-control/CONTROL_INTERPRETATION.md)

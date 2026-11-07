# What determines the turning point?

The numbers 2, 3.46 and 4 are the distances where particular input tanh arguments reach magnitude 1. They are **not measured universal fold roots**, and no theorem makes the learned output derivative vanish there. The observed five original roots lie at about 1.91, 2.19, 2.37, 2.65 and 3.14.

## Exact input-scale and weight-adaptation decomposition

Let $q=\log d$ and write the encoder as

$$
h_{g,\theta}(q)=F_\theta\!\left(\tanh(gVe^{-q})\right)
=H_\theta(q-\log g).
$$

A simple turning point is a zero of the full learned derivative, not a zero of an individual tanh derivative. If $r_*(\theta)$ is a root of $H_\theta'$, its physical location satisfies

$$
\log d_* = \log g+r_*(\theta).
$$

With frozen weights, doubling $g$ doubles $d_*$. After retraining, weights change too. For a corresponding single root,

$$
\log\frac{d_{*,1}}{d_{*,0}}
=\log\frac{g_1}{g_0}
+\left[r_*(\theta_1)-r_*(\theta_0)\right].
$$

For seed 131, gain 4→8 gives a measured log movement of **0.74529 = 0.69315 + 0.05215**. The frozen prediction is 4.7359; retraining puts the turn at 4.9894. For gain 4→2, the decomposition is **−0.09638 = −0.69315 + 0.59676**: adaptation opposes most of the direct scale shift. This is an exact coordinate identity for matched roots, not a statistical mediation percentage. If roots appear, disappear or cannot be uniquely matched, there is no such single-root comparison; those cases are retained as unmatched.

## Why changing the data changes this result

The same fixed physical samples correspond to different positions $q-\log g$ in the encoder's input response. Gain changes the geometry through which the network receives those samples. Sampling determines where loss gradients are generated and how often. A neural update affects other distances as well, because the parameters are shared.

For example, the parameter-response kernel at fixed weights is

$$
K_g(q,s)=\nabla_\theta h_{g,\theta}(q)^\mathsf{T}
\nabla_\theta h_{g,\theta}(s).
$$

Changing gain translates both arguments inside $H_\theta$; changing sampling changes the gradient forcing applied through that response. This identity explains why a state histogram alone does not predict the learned curve. It does not identify a unique final optimum or prove that a tanh landmark must become a fold.

The new intervention holds alignment pairs and physical dynamics fixed and changes **only SIGReg sampling**, with pure ICM throughout. It tests this sampling interaction without the final OU phase present in the previous repair experiment.

More explicitly, let $C_t$ and $S_t$ be the batch means of $\cos(ty)$ and $\sin(ty)$, $T_t=e^{-t^2/2}$ the target, and $w_t$ the implemented integration weights. The scalar SIGReg loss is $N\sum_t w_t[(C_t-T_t)^2+S_t^2]$. Its force on output $y_j$ is

$$
\frac{\partial L_R}{\partial y_j}
=2\sum_t w_t t\left[-(C_t-T_t)\sin(ty_j)+S_t\cos(ty_j)\right].
$$

The batch-size factor cancels the derivative of the mean. Sampling changes the means and which views receive these forces. This force adjusts a distribution of outputs; it does not directly penalize reversing physical order. For a plain gradient step its first-order effect on the slope at another distance is proportional to

$$
-\sum_j\frac{\partial L_R}{\partial y_j}\,
\frac{\partial K_g(q,q_j)}{\partial q}.
$$

Thus mostly near-object examples can change a far-distance slope through shared parameters. Gain changes this neural response, and reweighting changes the forces applied through it. This explains the two pathways in the intervention. Adam additionally incorporates past gradients and its adaptive denominator, as checked below; the plain-gradient expression is not substituted for its actual update.

## The actual optimizer's local effect

Write the weighted loss gradients as $a_t=\nabla_\theta(0.9L_A)$ and $r_t=\nabla_\theta(0.1L_R)$. Adam's first moment is linear, so starting from zero,

$$
m_t=m_t^A+m_t^R.
$$

Using Adam's actual combined second-moment denominator $D_t$, the parameter update splits into

$$
u_t=u_t^A+u_t^R,
\qquad
u_t^j=-\eta_t\frac{\widehat m_t^j}{D_t}.
$$

This does **not** split Adam into two causal optimizers: $D_t$ depends on both losses and their history. We verify the numerical parameter-update reconstruction and independently replay saved optimizer steps.

For $s(q,\theta)=\partial h/\partial q$, the first-order slope change is

$$
\Delta s(q)
\approx\nabla_\theta s(q,\theta)^\mathsf{T}u_t^A
+\nabla_\theta s(q,\theta)^\mathsf{T}u_t^R.
$$

Nested differentiation gives both terms. We compare their sum with the actual finite update and check the mixed derivatives against double-precision finite differences. Approximation errors and nearly zero updates remain in the outputs.

Near a simple root, its one-step movement follows the implicit-root relation

$$
\Delta q_*
\approx-\frac{\Delta s(q_*)}{\partial s(q_*,\theta)/\partial q}.
$$

This connects the optimizer update to root motion. It applies locally to a nondegenerate root; it does not describe the birth of a new root or justify forced matching across disappearing branches. The recorded training histories show when roots first enter the evaluated range, while the matched-root comparison checks their subsequent motion.

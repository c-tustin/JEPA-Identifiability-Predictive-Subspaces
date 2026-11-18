# Verifying fold development from occupancy and training dynamics

November 17, 2026.

A fold develops when training makes a scalar encoder turn backward along the observation manifold. We can verify a forthcoming fold by showing that the actual optimizer update drives the slope negative somewhere while it remains positive elsewhere. Occupancy determines which output forces are averaged; the encoder determines how those forces change a slope; optimizer state determines the update that is actually applied. A sufficient certificate also bounds the error of linearizing that update.

The central sufficient condition, derived below, is

$$
\boxed{\eta_t C_t(q_-) > u_t(q_-)+\rho_t(q_-),\qquad
u_t(q_+)-\eta_t C_t(q_+)>\rho_t(q_+).}
\tag{1}
$$

Here $u_t=\sigma\partial_q h_{\theta_t}$ is the slope in one fixed orientation $\sigma\in\{-1,+1\}$, $C_t$ is the optimizer's slope-decreasing force, and $\rho_t$ bounds the finite-step remainder. The two inequalities guarantee opposite slope signs after the step. The derivation concerns the posthoc scalar encoder trained on frozen ICM data, rather than the original ICM feature encoder or actor.

## Geometry and the information needed

Let $q=\log d\in Q=[q_{\min},q_{\max}]$, let $x(q)$ be the simulator's sensor map, and define

$$
h_\theta(q)=f_\theta(x(q)),\quad
J_\theta(q)=\nabla_\theta h_\theta(q),\quad
u_\theta(q)=\sigma\partial_qh_\theta(q).
$$

Since $\partial_qh=d\,\partial_dh$, changing between distance and log distance preserves the derivative sign. Fix $\sigma$ once from the initial orientation; changing it separately at each query would hide folds.

Assume $h_\theta$ is continuously differentiable in $q$ on $Q$. If $u_\theta(q_- )<0$ and $u_\theta(q_+)>0$ at two points in $Q$, the representation has a slope reversal and is noninjective on $Q$. Indeed, a continuous injective real-valued function on an interval must be strictly monotone; derivatives of either sign exclude both possible monotone orientations. Continuity of the slope also supplies a zero between the witnesses. A zero slope alone does not prove a fold: $h(q)=q^3$ is injective. Negative slope everywhere can simply be the other global orientation.

For the update expansions, also assume $u_\theta(q)$ is twice continuously differentiable in parameters along the proposed parameter segment. Smooth tanh encoders satisfy these differentiability assumptions. For ReLU encoders, activation crossings require piecewise analysis or direct value witnesses; a smooth Taylor bound cannot be used across a kink without justification.

To determine the next training step, retain:

- Current parameters, sensor map, and loss implementation, including coefficients and normalization.
- The regularizer sampling measure $p_R$ and alignment pair measure $\mu_A(dq\,dr)$, or the actual sampled batches and their weights.
- Optimizer moments, step counter, learning rate, epsilon, and any decay, clipping, target-network, or projection state.

A marginal occupancy histogram by itself does not determine these quantities. With a fixed stationary actor, $\mu_A(dq\,dr)=p_A(dq)P_\pi(q,dr)$; the transition kernel remains necessary even when $p_A=p_R$. Actual finite trajectories can have different endpoint marginals. For an online actor or EMA target, their states must be included in the evolving training state.

## How occupancy produces a slope force

Write the objective as

$$
L(\theta)=\lambda_A L_A(\theta)+\lambda_R R_{p_R}(\theta),\qquad
L_A=\int[h(q)-h(r)]^2\,\mu_A(dq\,dr).
\tag{2}
$$

For the symmetric alignment used here, both endpoints are differentiated. Detached targets require their actual gradient instead. In the historical baseline, $\lambda_A=0.9$ and $\lambda_R=0.1$.

For a finite-frequency characteristic-function regularizer, define

$$
C_k=\int\cos(\omega_k h(s))p_R(ds),\qquad
S_k=\int\sin(\omega_k h(s))p_R(ds),\qquad
T_k=e^{-\omega_k^2/2},
$$

$$
R_{p_R}=\kappa\sum_k w_k[(C_k-T_k)^2+S_k^2].
\tag{3}
$$

Here $w_k$ includes all quadrature factors from the implementation, and $\kappa$ is its fixed loss multiplier. The existing integral diagnostic uses $\kappa=1024$ and 17 frequency knots; increasing the number of observations in a diagnostic must not silently change $\kappa$.

Treat the data measure and sample weights as fixed when differentiating, and assume derivatives may be interchanged with the integrals. The output force before multiplying by occupancy is

$$
G_{p_R}(s)=2\kappa\sum_k w_k\omega_k
\left[-(C_k-T_k)\sin(\omega_k h(s))+S_k\cos(\omega_k h(s))\right].
\tag{4}
$$

Therefore

$$
\nabla_\theta R_{p_R}=\int G_{p_R}(s)J_\theta(s)p_R(ds).
\tag{5}
$$

For an empirical measure $p_R=\sum_i\alpha_i\delta_{q_i}$, normalized nonnegative weights $\alpha_i$ replace the integral by a sum. In particular $\partial R/\partial h(q_i)=\alpha_iG_{p_R}(q_i)$; do not multiply by $\alpha_i$ twice.

Define the encoder response kernel

$$
K_\theta(q,s)=J_\theta(q)^\top J_\theta(s).
$$

Differentiating the regularizer-induced function update with respect to the query gives

$$
I_{p_R}(q)=\int G_{p_R}(s)\partial_qK_\theta(q,s)p_R(ds)
=\nabla_\theta\partial_qh(q)^\top\nabla_\theta R_{p_R}.
\tag{6}
$$

The corresponding alignment contraction is

$$
A_{\mu_A}(q)=2\int [h(s)-h(r)]
[\partial_qK_\theta(q,s)-\partial_qK_\theta(q,r)]\mu_A(ds\,dr).
\tag{7}
$$

Thus ordinary gradient flow obeys the exact instantaneous identity

$$
\partial_tu(q)=-\sigma[\lambda_A A_{\mu_A}(q)+\lambda_R I_{p_R}(q)].
\tag{8}
$$

An SGD step has the same expression multiplied by its learning rate, plus a finite-step remainder. Positive $\sigma I_{p_R}$ decreases the oriented slope through the regularizer. Whether it crosses zero also depends on the current slope, alignment, and the size of the step.

Equations (4)–(6) expose three interactions: occupancy changes the outer weights, changes the global moments inside $G_{p_R}$, and is filtered through the signed response $\partial_qK_\theta$. Even the bin with the most occupancy need not have the largest slope-decreasing contribution. Input gain, architecture, and current weights change the response kernel. Multiplying an old output gradient by inverse occupancy changes its outer weights while retaining the old global moments; recomputing weighted moments changes both.

The full measures are needed for exact integrals. A histogram with bin masses $\pi_b$ gives the identity

$$
I_{p_R}(q)=\sum_b\pi_b\,\mathbb E_{p_R}[G_{p_R}(S)\partial_qK(q,S)\mid S\in B_b],
\tag{9}
$$

but masses alone do not specify the conditional factors. Evaluating bin centers is an approximation unless within-bin errors are bounded. Empty bins supply no empirical force; parameter sharing can still change the encoder there.

## Finite batches change the objective being averaged

A full-data plug-in force is not generally the expected minibatch force. Let $Z_1,\ldots,Z_N$ be iid from $p_R$, with equal weights and the same fixed $\kappa$. Since $|e^{i\omega h}|=1$,

$$
\mathbb E\widehat R_N
=\left(1-\frac1N\right)R_{p_R}
+\frac\kappa N\sum_k w_k[1+T_k^2-2T_k C_k].
\tag{10}
$$

To prove (10), use $\mathbb E|\widehat\phi_k|^2=(1-1/N)|\phi_k|^2+1/N$, with $\phi_k=C_k+iS_k$. The additional term depends on the encoder. Under differentiation–expectation interchange, its derivative contributes

$$
\mathbb E\nabla\widehat R_N
=\left(1-\frac1N\right)\nabla R_{p_R}
+\frac{2\kappa}{N}\sum_k w_kT_k\omega_k
\int\sin(\omega_kh(s))J(s)p_R(ds).
\tag{11}
$$

The iid formulas do not apply directly to correlated endpoints in temporal pairs, sampling without replacement, or random self-normalized weights. If all endpoints share marginal $p_R$, the general equal-weight relation is

$$
\mathbb E\widehat R_N=R_{p_R}
+\kappa\sum_k w_k\,\mathbb E|\widehat\phi_k-\phi_k|^2,
\tag{12}
$$

where the last term includes every endpoint covariance and depends on parameters. For unequal marginals, use their actual mean characteristic function as the center instead. For a realized update, avoid this expectation issue by using the actual batch moments in (4) and its actual alignment pairs in (7).

Even an exact expected gradient does not determine an expected Adam update: its denominator and first moment depend jointly on the sampled gradients. Correlated data and unknown future batches require a specified sampling law for probabilistic forecasts.

## The slope force for Adam

Let $g_t$ be the actual complete gradient at parameters $\theta_t$. For the baseline Adam implementation without weight decay,

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^{t+1}},\quad
D_t=\operatorname{diag}\left(\sqrt{\frac{v_t}{1-\beta_2^{t+1}}}+\epsilon\right),\quad
\Delta\theta_t=-\eta_tD_t^{-1}\widehat m_t.
\tag{13}
$$

Index $t=0,1,\ldots$ labels the update from $\theta_t$ to $\theta_{t+1}$; moments with index $t-1$ are its before-state. A fresh optimizer has $m_{-1}=v_{-1}=0$ and uses bias-correction counter $t+1$. For a resumed optimizer, use its actual saved counter instead. Define

$$
a_t(q)=\nabla_\theta u_{\theta_t}(q),\qquad
C_t(q)=a_t(q)^\top D_t^{-1}\widehat m_t.
\tag{14}
$$

Then $-\eta_t C_t(q)$ is the linearized slope change of the actual Adam update. This is the $C_t$ in (1). It already includes alignment and regularizer coefficients and history; multiplying by $\lambda_R$ again would be incorrect.

For the current regularizer gradient, an equivalent contraction is

$$
C_{t,\mathrm{current}\ R}(q)=
\frac{(1-\beta_1)\lambda_R}{1-\beta_1^{t+1}}
\int G_{\widehat p_{R,t}}(s)
\,a_t(q)^\top D_t^{-1}J_{\theta_t}(s)\,\widehat p_{R,t}(ds).
\tag{15}
$$

Keep the denominator from the actual combined gradient. Equation (15) is an exact algebraic decomposition of the first-order response conditional on this realized step. It does not give the counterfactual update obtained by removing alignment or changing occupancy, which would also change $v_t$.

For constant $\beta_1$ and $m_{-1}=0$, expand the whole numerator:

$$
C_t(q)=\frac{1-\beta_1}{1-\beta_1^{t+1}}
\sum_{j=0}^t\beta_1^{t-j}a_t(q)^\top D_t^{-1}g_j.
\tag{16}
$$

The $g_j$ are gradients at past parameters and past batches; $a_t,D_t$ are current quantities. This is how earlier occupancy-dependent forces can drive a fold even when the present force resists it. For a nonzero initial moment, add $\beta_1^{t+1}m_{-1}/(1-\beta_1^{t+1})$ to the expanded $\widehat m_t$ before contracting with $a_t(q)^\top D_t^{-1}$. With decoupled weight decay, clipping, or projection, include their actual effect in $\Delta\theta_t$ and use the general contraction $a_t^\top\Delta\theta_t$.

## A sufficient certificate for the next update

Taylor's theorem applied to the slope as a function of parameters gives

$$
u_{t+1}(q)=u_t(q)+a_t(q)^\top\Delta\theta_t+R_t(q)
=u_t(q)-\eta_t C_t(q)+R_t(q).
\tag{17}
$$

If

$$
\sup_{z\in[0,1]}\left\|\nabla_\theta^2u_{\theta_t+z\Delta\theta_t}(q)\right\|_{\mathrm{op}}\le H_t(q),
$$

then

$$
|R_t(q)|\le\rho_t(q):=\tfrac12H_t(q)\|\Delta\theta_t\|_2^2.
\tag{18}
$$

A tighter alternative directly bounds the directional quantity
$|\Delta\theta_t^\top\nabla_\theta^2u_{\theta_t+z\Delta\theta_t}(q)\Delta\theta_t|$ along the segment and divides by two. A few Hessian samples or a finite-difference agreement do not establish its supremum.

**Proposition.** Choose two query points $q_-,q_+\in Q$. If

$$
u_t(q_-)-\eta_tC_t(q_-)+\rho_t(q_-)<0,
\qquad
u_t(q_+)-\eta_tC_t(q_+)-\rho_t(q_+)>0,
\tag{19}
$$

the next encoder is folded on $Q$.

**Proof.** The first bound makes the next oriented slope strictly negative at $q_-$. The second makes it strictly positive at $q_+$. The geometric argument above proves a slope reversal and noninjectivity. If the current encoder is certified strictly ordered throughout $Q$, this step creates a fold that was absent before. Otherwise it certifies a fold after the step, without proving that it is the first one.

For numerical certificates, add slope-evaluation, contraction, parameter-rounding, and bound-computation errors to $\rho_t$ conservatively, or use outward-rounded interval arithmetic. Without a verified remainder/error bound, $u_t-\eta_t C_t<0$ is a first-order warning. Directly evaluating the proposed encoder at two points with opposite, rigorously bounded slope signs is another sufficient certificate and avoids a parameter Hessian bound. Ordinary float64 evaluation supplies strong empirical evidence but does not by itself give rigorous error intervals.

A complementary sufficient no-fold certificate uses spatial coverage. If $u_{t+1}$ is $L$-Lipschitz on $Q$ and every point is within $\delta$ of a probe $q_i$, then

$$
\min_i u_{t+1}(q_i)>L\delta+\varepsilon
\tag{20}
$$

guarantees positivity throughout $Q$ when probe error is at most $\varepsilon$. For fold existence, two sign witnesses suffice; for absence, gaps between probes must be controlled. The [existing no-fold note](../no-fold-proof/no-fold.tex) gives whole-interval guarantees for constrained encoders.

## Predicting development over several updates

One-step certification conditions on a known proposed update. A forecast from initialization must control the subsequent path, including its batches and optimizer state.

At a fixed query $q_-$, suppose every update along the realized or enclosed path obeys

$$
\eta_t C_t(q_-)\ge\gamma_t,\qquad |R_t(q_-)|\le\rho_t(q_-).
$$

Summing (17) yields

$$
u_T(q_-)\le u_0(q_-)-\sum_{t<T}\gamma_t+\sum_{t<T}\rho_t(q_-).
\tag{21}
$$

If the sum of the guaranteed decrements exceeds the initial slope plus the summed remainders, the slope must be negative by $T$. Simultaneously bound the anchor, for example with $\eta_tC_t(q_+)\le b_t$ and

$$
u_0(q_+)-\sum_{t<T}b_t-\sum_{t<T}\rho_t(q_+)>0.
\tag{22}
$$

Together these conditions prove a fold by $T$. With constants $\gamma_t\ge\gamma>\rho\ge\rho_t(q_-)$, any integer $T>u_0(q_-)/(\gamma-\rho)$ suffices for the negative witness, provided the anchor bound holds through $T$.

The bounds must hold for all intermediate states in the forecast, for example over validated enclosures propagated under the update rule. Freezing the initial kernel and moments is only a local approximation unless their drift is bounded. For stochastic training, bounds holding on an event of probability at least $1-\delta$ give a certificate with that probability. No probability follows from a marginal occupancy alone.

For continuous-time intuition, write $c(q,t)=\sigma[\lambda_AA+\lambda_RI]$ in gradient flow. If $c(q_-,t)\ge\gamma>0$ until time $T$, then $u(q_-,T)\le u(q_-,0)-\gamma T$. An independently positive anchor converts crossing of this bound into a fold. This flow argument is not an Adam or finite-step theorem.

## Where a smooth fold first appears

At a first interior loss of positivity of a smooth slope field, an interior minimizer satisfies

$$
u(q_*,t_*)=0,\qquad \partial_qu(q_*,t_*)=0.
\tag{23}
$$

In the generic case $\partial_{qq}u(q_*,t_*)>0$ and $\partial_tu(q_*,t_*)<0$, the local expansion is

$$
u(q,t)\approx-\alpha(t-t_*)+\tfrac12\beta(q-q_*)^2,
\qquad \alpha,\beta>0.
\tag{24}
$$

Two slope roots then emerge around a negative region. This describes a smooth interior onset under the stated nondegeneracy conditions. A root can instead enter through the edge of $Q$, with $\partial_qu\ne0$ there. Discrete training can step across the onset without ever sampling an exact zero. The choice of $Q$ therefore matters: an initial reversal outside observed support and its later movement into support are different events.

Once a simple root $q_*(t)$ exists with $\partial_qu\ne0$, the implicit-function formula is

$$
\dot q_*(t)=-\frac{\partial_tu(q_*,t)}{\partial_qu(q_*,t)}.
\tag{25}
$$

For gradient flow, insert (8). For an interpolated Adam path, insert its actual parameter velocity. Formula (25) describes motion of an existing simple root; it is singular at the interior birth in (23), and does not locate a fold from occupancy peaks alone.

## What the inspected birth event establishes

The [occupancy integral diagnostic](../occupancy-integral/README.md) and its [saved numerical results](../occupancy-integral/output/results.json) inspect seed 131, gain 4, update 204 to 205, at $d=16$, in the increasing orientation. The empirical data reach only $d\approx13.73643$, so this query lies beyond observed support.

| Quantity | Slope or slope change at distance 16 |
| --- | ---: |
| Before slope | $+0.004150231$ |
| Actual after slope | $-0.000832708$ |
| Actual slope change | $-0.004982939$ |
| Linearized actual Adam update | $-0.004968168$ |
| Current SIGReg contribution with the actual Adam denominator | $+0.001426908$ |
| Historical SIGReg momentum contribution | $-0.006383163$ |
| Alignment contribution including history | $-0.000011922$ |

The displayed components sum to the linearized update, within rounding. The actual Adam proposal was replayed with zero parameter mismatch in the existing audit. Its Taylor error at this point was about $1.48\times10^{-5}$. The empirical reversal agrees with the optimizer contraction; the measured Taylor error is not an advance bound on future updates. Establishing a fold also requires a positive anchor; establishing its first creation on all of $Q$ additionally requires certifying the earlier encoder there. The existing training histories report sampled geometry, rather than that global before-state certificate.

At the same weights, the full-data native regularizer-only SGD prediction is $-0.00387795$, while the actual minibatch regularizer-only SGD prediction is $+0.0139440$. The current regularizer force and its history cannot be exchanged. This inspected event supports occupancy-dependent forces transported through encoder geometry and optimizer memory; it does not establish a universal concentration threshold.

## Verification protocol for the original encoder

1. **Fix the domain and orientation.** Report $Q$, observed support, initial slope margin, and whether the earlier encoder is empirically or rigorously ordered.
2. **Reconstruct the actual update.** Reload parameters, optimizer moments and counter, batches, loss coefficients, targets, and numerical dtype. Reproduce the proposed parameter vector before attributing its effect.
3. **Compute forces.** Use actual moments in (4), the contraction in (6), and the actual pair gradient in (7). Check the integral against $\nabla_\theta\partial_qh\cdot\nabla_\theta R$. Keep full-data and minibatch estimates separate.
4. **Include optimizer history.** Compute $a_t^\top\Delta\theta_t$, then separate current and historical terms under the shared Adam denominator if attribution is desired. Historical bin attribution requires gradients at their original weights and batches, or source-separated moment accumulators.
5. **Test the two witnesses.** Use (19) with a justified remainder/error bound, or bound the two slopes of the proposed encoder directly. Record each margin. A failed sufficient test is inconclusive.
6. **For a forecast, propagate bounds.** Verify (21)–(22) over a stated horizon and sampling event. A single negative directional derivative does not prove that the slope keeps decreasing.

This framework supplies sufficient mathematical conditions and an implementation protocol. No new training run, validated neural-network Hessian enclosure, or future-fold certificate for a new checkpoint is reported here. Those are the next computations needed to turn a local warning into a certified forecast. Existing empirical evidence, analytic identities, and the additional assumptions needed for certification remain separate.

## Reproducing the algebra checks

Run `python3 icm-folding/fold-development/verify.py` from the workspace or published repository root. The [verifier](verify.py) checks the output-force, kernel, and alignment identities in float64; enumerates all ordered two-sample batches of a three-atom measure to check (10) and its gradient; checks the Adam momentum expansion; and exercises (19) on a polynomial encoder whose global parameter Hessian bound is analytic. The [saved checks](checks.json) record errors and example margins. These checks validate the formulas and example implementation; they do not supply a rigorous remainder bound for the trained neural encoder.

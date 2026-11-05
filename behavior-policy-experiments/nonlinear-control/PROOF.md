# An action-anchored nonlinear identification lemma

**Status:** proved below under explicit assumptions. This is a coordinate-identification lemma and an approximate bound, not a theorem that unconstrained JEPA training finds the right encoder. No publication novelty is claimed.

The quoted future-work passage is from [Zhang et al., *On the Identifiability of Controlled World Models*](https://arxiv.org/html/2607.22430v1). The current project originally reproduced passive LeJEPA alignment; controlled prediction is an additional problem.

## Assumptions

On an interior interval $I=[\ell,r]$ of length $L$, physical distance obeys

$$
dD_t=[b(D_t)+cA_t]dt+\sigma(D_t)dW_t,\qquad X_t=V/D_t,
$$

with $c\ne0$. The drift $b$ can be nonlinear and the diffusion coefficient can depend on state. Coefficients have the regularity needed for the diffusion and Itô's formula. We work before hitting the boundary; reflection introduces local-time terms and must be handled separately.

Assume the learned coordinate is $Y=T(D)$ for a $C^2$ diffeomorphism $T$. This is an information-preservation assumption, not a consequence of SIGReg. The fitted conditional drift has the structural form

$$
\widehat m(y,a)=\widehat b(y)+ka,
$$

where the action coefficient $k$ is constant. Let $\mu$ be the sampling measure on $I$; it need not be stationary or Gaussian. Its density satisfies $p(d)\ge p_{\min}>0$. At each state,

$$
\operatorname{Var}(A\mid D=d)\ge v_0>0.
$$

The errors below refer to conditional *drifts*, not raw noisy increments, finite-time conditional means, or ordinary JEPA prediction loss.

## Exact result

Itô's formula gives the true transformed drift

$$
m_T(d,a)=T'(d)b(d)+\tfrac12T''(d)\sigma^2(d)+cT'(d)a.
$$

At a fixed state, subtract conditional means over actions. The arbitrary state-only drift disappears:

$$
m_T(d,A)-E[m_T(d,A)\mid d]=cT'(d)(A-E[A\mid d]).
$$

If the fitted drift agrees with the true drift, positive conditional action variance implies $cT'(d)=k$ almost everywhere. Continuity and positive state density extend this throughout $I$. Therefore

$$
T(d)=\alpha d+\beta,\qquad \alpha=k/c\ne0.
$$

Thus nonlinear autonomous dynamics do not prevent affine coordinate identification when constant physical actuation is known, informative encoders are assumed, and actions vary at each state. Gaussianity is unnecessary for this lemma. A bounded physical coordinate should not be forced to have an exact Gaussian output law; scale and translation can instead be anchored by known action units and centering.

## Approximate result

Suppose the population conditional-drift fitting error is

$$
E_{D,A}[(m_T(D,A)-\widehat b(T(D))-kA)^2]\le\varepsilon.
$$

Conditional variance is at most the conditional second moment. Hence

$$
E_D[\operatorname{Var}(A\mid D)(cT'(D)-k)^2]\le\varepsilon.
$$

Set $\alpha=k/c$. Conditional excitation and the lower density bound yield

$$
\int_I(T'(d)-\alpha)^2\,dd
\le\frac{\varepsilon}{c^2v_0p_{\min}}.
$$

Choose $\beta=T(\ell)-\alpha\ell$. The fundamental theorem of calculus and Cauchy–Schwarz give

$$
\boxed{\sup_{d\in I}|T(d)-\alpha d-\beta|^2
\le\frac{L\varepsilon}{c^2v_0p_{\min}}.}
$$

Weak conditional action variation and poor state coverage worsen the guarantee through separate factors. The bound may be loose. A scale/informativeness condition is still needed to exclude nearly constant coordinates when interpreting approximate recovery.

For three actions $A\in\{-1,0,1\}$ with an $\eta$-uniform mixture,

$$
\pi(a\mid d)\ge\eta/3,\qquad v_0\ge\frac{2\eta}{3}.
$$

This follows from the variance decomposition of a mixture. The uniform component alone contributes $2\eta/3$; the remaining terms are nonnegative.

## What remains to extend the full JEPA theorem

The lemma assumes an informative, invertible scalar coordinate and a conditional-drift risk bound. It does not establish either for a neural encoder trained jointly with a predictor. A stronger theorem would need a compatible objective, a way to prevent information loss, finite-sample drift estimation with temporal dependence, discretization-bias bounds, and boundary control. Noise variance in increments grows relative to drift as the observation interval shrinks; ordinary finite-step MSE cannot simply be substituted for $\varepsilon$.

The [numerical prototype](README.md) uses paired action interventions and common simulator noise to estimate action effects. It adds information beyond passive trajectories. Its success is an assumption check and an identifiable-model example, not a proof for arbitrary behavior policies or unconstrained LeJEPA.

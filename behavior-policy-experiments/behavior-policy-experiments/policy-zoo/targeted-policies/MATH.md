# Why these controllers were plausible candidates

The visible top-vertex height is $h=1/d$. A controller can therefore compute $\widehat d=1/h$ from its observation. This uses a known rendering relation; it does not add a world coordinate or give labels to the encoder.

If the desired log-distance distribution is Gaussian with center $\mu=\log4$ and standard deviation $s$, its physical-distance density is

$$
\pi(d)\propto\frac1d
\exp\!\left[-\frac{(\log d-\mu)^2}{2s^2}\right],
\qquad 1\le d\le16.
$$

The factor $1/d$ is the Jacobian. Bounds truncate the Gaussian, and a finite-step physical walk visits a displacement lattice rather than a continuous density.

## Random walks with the intended target

Let $m(d)$ be the probability of proposing a movement, split equally between $d\pm0.25$. Reject proposals outside the world. For a valid neighboring proposal, Metropolis acceptance is

$$
\alpha(d,d')=
\min\!\left(1,
\frac{\pi(d')m(d')}{\pi(d)m(d)}\right).
$$

The proposal-activity ratio matters: a controller that changes its movement rate but omits this ratio changes its stationary distribution. The implementation tests detailed balance for both constant and distance-dependent activity. This is a stationary-law statement on a reachable lattice, not a claim that 128 burn-in actions guarantee mixing.

For the log-scaled walk,

$$
m(d)=\min\!\left(1,(d/8)^2\right).
$$

For small steps, $\Delta\log d\approx\pm0.25/d$, so the proposed log-distance variance is approximately $m(d)0.25^2/d^2$. Below the activity cap, this is constant. Acceptance and finite steps introduce corrections; clipping at $d>8$ changes it too. Thus this is an approximation to Gaussian log-coordinate diffusion, not exact OU dynamics.

The initial 64 target-approach actions were intended to avoid waiting for a slow local walk to mix from the ordinary uniform physical reset. They consume existing burn-in actions. The separate no-escort follow-up retains the same walk and removes that action phase.

## Other controllers

Random-target tracking makes an internal Gaussian target evolve slowly and uses feasible physical steps to follow it. The physical distance does not exactly inherit the target's Gaussian law or correlation. Gaussian-weighted coverage chooses actions using desired bin proportions and current counts; it has no stationary-Gaussian guarantee.

These were design hypotheses. Recorded source distributions and fresh-seed encoder scores determine whether they worked. A Gaussian-looking visitation histogram alone does not specify pair dynamics, effective sample diversity or which encoder function finite optimization reaches.

## Why matching a Gaussian target need not beat the OU reference

The physical world is bounded. Even an ideal stationary log-coordinate distribution is **truncated** Gaussian. Writing $q=\log d$, $A=\Phi(-\mu/s)$ and $B=\Phi((\log16-\mu)/s)$, a transform that Gaussianizes that truncated distribution is

$$
h(q)=\Phi^{-1}\!\left(
\frac{\Phi((q-\mu)/s)-A}{B-A}
\right).
$$

This is ordered but nonlinear in $q$. It illustrates pressure from a Gaussian-output target to distort an affine distance code; it is not asserted to minimize the combined alignment/SIGReg objective. Exact tails also cannot be implemented by the bounded finite tanh encoder.

There is a second finite-data difference: 64 strongly correlated views from one physical trajectory are not 64 independent draws from the target. The legacy fixed-OU source uses independent Gaussian pairs, while these physical controllers have only 256 independently initialized trajectories. Measured finite-pool shape, support and transition statistics therefore matter, even when the controller's intended stationary law is appropriate. These mechanisms are plausible contributors to the observed gap; this study does not isolate their separate effects.

# Learning OU parameters

Two separate tests keep the same distance observation $x=V/d$. **Neither learns from distance-recovery scores.** Seeds 13–17 are new to the earlier studies.

## Exact OU pair generator

Learn correlation $\rho$, observation spread $s$, or both before fitting the encoder:

$$
z'=\rho z+\sqrt{1-\rho^2}\,\epsilon,\qquad d=4e^{sz},\qquad x=V/d.
$$

The parameter optimizer sees rendered log apparent heights. It maximizes

$$
J=\widehat\rho(1-\widehat\rho)+0.1H(\widehat p)-0.5\,\widehat E_{\rm outside},
$$

where $\widehat p$ is a soft histogram of observed log heights and the last term penalizes views outside the shared apparent-size range. There are 600 parameter updates of 512 pairs. Parameters then freeze for 4,000 ordinary representation updates. Bounds are $\rho\in[0.1,0.98]$ and $s\in[0.3,1.2]$.

For ideal OU alignment, the separation between linear and nonlinear modes is

$$
\gamma(\rho)=\rho-\rho^2,
$$

maximized at $\rho=1/2$. The fixed-$0.5$ control tests whether learning helps beyond that analytic choice. A larger ideal gap need not improve the implemented finite-weight SIGReg loss.

This generator has differentiable simulator access and uses known apparent-size geometry. It is not a physical three-action policy.

## Physical controller

A stochastic controller learns an OU-inspired target in observed log height $u=\log(x_2)$:

$$
\Delta u_{\rm target}\sim\mathcal N((1-\rho)(m-u),\tau^2).
$$

Gaussian mass above $+0.02$, between $\pm0.02$, and below $-0.02$ supplies approach/stay/retreat probabilities. A 10% uniform action floor remains. The real environment still moves by 0.25 and reflects at distances 1 and 16. **The realized transitions are not exactly OU.**

Actor–critic learns $\rho\in[0.1,0.995]$, $\tau\in[0.002,0.3]$ and center $m\in[-\log16,0]$ using the count-coverage reward. Initial values are $(0.9,0.02,-\log4)$. Controller-parameter Adam uses learning rate 0.01; critic uses 0.0001. Interaction budget remains 192,000. The frozen controller is tested alone and with the original OU handover.

Seven conditions × five seeds = 35 final encoders. All parameter choices are fixed before this study's scores are inspected. This is a geometry-aware controller, not a general observation-agnostic OU learner.

```sh
python behavior-policy-experiments/learned-ou/run_learned_ou.py
python behavior-policy-experiments/learned-ou/run_learned_ou.py --evaluate-only
```

Use `--quick --output /tmp/learned-ou-smoke` for an execution check, or `--resume` for completed conditions. [Results](output/RESULTS.md) include the learned parameters. No best-seed selection or recovery-labelled tuning is used.

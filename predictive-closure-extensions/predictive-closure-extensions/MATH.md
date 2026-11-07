# What the diagnostic establishes

Let $Z,E$ be independent standard Gaussians, $S$ an independent uniform sign, and $Y=SrZ+\sqrt{1-r^2}E$. Each action-conditioned transition preserves the Gaussian marginal. The coordinate $d=4e^{0.7Z}$ is positive and scalar; triangle vertices are $V/d$. No additional world coordinate is introduced.

Without action information,

$$
E[Y\mid Z]=0,\quad E[Y^2\mid Z]=1-r^2+r^2Z^2,\quad
\min_g E[(Y-g(Z))^2]=1.
$$

The conditional fourth cumulant is $-2r^4Z^4$. Except at $Z=0$, this is not Gaussian. The state-dependent conditional variance also violates Appendix A.2's constant scalar variance requirement. Gaussian marginals alone do not satisfy that theorem.

The squared-magnitude correlation is $\operatorname{Corr}(Z^2,Y^2)=r^2$, even though $\operatorname{Corr}(Z,Y)=0$. The dependence missed by the mean is therefore directly visible in second moments.

Using independence of $Z$ and $S$ and the information chain rule,

$$
I(Z;(Y,S))=I(Z;Y\mid S)=-\tfrac12\log(1-r^2),
$$

$$
I(Z;Y)=-\tfrac12\log(1-r^2)-I(Z;S\mid Y)
\ge-\tfrac12\log(1-r^2)-\log2.
$$

At $r=.95$ the lower bound is about .4708 nats. A separate Gaussian AR law with correlation .5 has MI about .1438 and optimized MSE .75. Thus the two **transition laws** are oppositely ranked by MI and mean MSE. This does not compare two encoders of one fixed world. It does not refute the qualified theorem or establish any set-level selection result.

Supplying the current action gives $E[Y\mid Z,S]=SrZ$ and MSE $1-r^2=.0975$. The conditional distribution is now Gaussian with constant variance. This is a comparison of predictor information, not a proof that jointly learned encoders satisfy all finite-precision/marginal/reader assumptions.

## Independent-noise regression

For short interval $\delta$, increment divided by $\delta$ has leading conditional variance $\sigma(d)^2/\delta$. Hence a least-squares row should be multiplied by $\sqrt{\delta}/\sigma(d)$ for leading inverse-variance weighting. The follow-up uses the known diffusion shape and estimates its amplitude separately. It is not a general learned noise model.

The interval correction in the experiment is an approximation: its finite feature basis need not contain every generator term. Higher sampling variance, finite-time correction misspecification and Euler error are distinct possible contributors. Observed RMSE alone does not identify their shares. The short-time conditional-drift lemma still requires a sufficiently informative smooth representation; exact inverse-renderer calibration supplies that information in this control, rather than learning or proving it.

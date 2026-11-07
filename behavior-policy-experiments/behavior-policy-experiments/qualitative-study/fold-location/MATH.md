# Why an input scale can become a distance scale

The renderer has no special event at distance 2, 3 or 4. Its output is exactly $x(d)=V/d$. The reference distance 4 in the simulator's latent-coordinate conversion cancels when rendering; it is not a physical boundary. Reflection happens at distances 1 and 16.

The encoder does contain another, consequential 4:

$$
h_g(d)=F_\theta\!\left(\tanh\!\left(\frac{gV}{d}\right)\right),\qquad g=4.
$$

$F_\theta$ is the trainable network. The nonzero magnitudes of $V$ are $0.5$, $\sqrt{3}/2$ and $1$. Thus the tanh arguments pass through magnitude 1 at $d=2$, $d\approx3.46$ and $d=4$. These mark smooth changes in sensitivity, not exact switch points. They are distinct from both a fold root and the maximum sensitivity on a logarithmic plot.

For one preprocessed coordinate $u_i=\tanh(gV_i/d)$,

$$
\frac{d u_i}{d\log d}
=-\frac{gV_i}{d}\,\operatorname{sech}^2\!\left(\frac{gV_i}{d}\right).
$$

The learned coordinate therefore has derivative

$$
\frac{d h_g}{d\log d}
=\sum_i\frac{\partial F_\theta}{\partial u_i}\frac{d u_i}{d\log d}.
$$

The fixed tanh preserves each coordinate's order and the observation remains injective. A fold requires the **learned sum** to change sign. Different contributions can cancel near a turning point even when individual units remain sensitive. Our saved derivatives decompose this sum through the 48 units of the last hidden layer and check that the contributions reconstruct the total.

## Why tapering is a different issue

As $d$ grows, $u\to0$. Smoothness of the finite network gives

$$
h_g(d)=F_\theta(0)+\frac{g\,\nabla F_\theta(0)^\mathsf{T}V}{d}+O(d^{-2}).
$$

Consequently $d h_g/d\log d=O(d^{-1})$. If the leading coefficient cancels, decay can be faster. This gives the model a natural constant-output limit at large distance. It does **not** prove that flattening must begin between 2 and 4: that onset depends on learned weights and training examples. It also does not require tanh to be saturated at far distances; the small inputs themselves shrink with distance.

For comparison, an affine log-distance coordinate has constant nonzero derivative with respect to log-distance. OU training approximates that behavior over the finite evaluation range. Neither this bounded architecture nor the finite experiment represents an unbounded logarithm exactly at every positive distance.

## What the input-gain experiment identifies

With weights frozen there is an exact identity,

$$
h_{g,\theta}(d)=h_{4,\theta}\!\left(\frac{4d}{g}\right).
$$

Thus frozen extrema move in direct proportion to gain. This is algebra, not independent evidence for the training mechanism. The useful experiment **retrains** the weights at gains 2 and 8 on the identical stored pairs. The learned network can compensate for the changed gain, so its extrema need not obey that identity across retrained models.

The actor and ICM module also originally used gain 4. This follow-up holds their trajectories fixed and changes only JEPA preprocessing. It tests encoder-scale dependence conditional on those data. It does not identify how changing the actor's own input scale would change its exploration.

The cancellation plots describe how the saved functions behave. They do not establish why optimization chose those weights. Input-gain changes also alter the initial function and optimization conditioning, so a shift is evidence for a scale effect rather than proof that tanh saturation alone causes folding.

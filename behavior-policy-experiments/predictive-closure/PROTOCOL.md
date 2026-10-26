# Fixed protocol

Fixed before full runs. Exactly the existing triangle vertices divided by distance; one physical coordinate. $z=\log(d/4)/0.7$ changes coordinates, not state dimension. No observation noise, crop or quantization.

## Assumption diagnostic

Compare TWO laws: ordinary Gaussian AR correlation 0.5, and signed OU $z'=Srz+\sqrt{1-r^2}\epsilon$, $r=0.95$, independent uniform sign/action. Both stationary marginals are Gaussian. Signed OU has Bayes mean zero, MSE 1, state-dependent conditional variance $1-r^2+r^2z^2$, and non-Gaussian mixture conditionals. Its MI is at least $-\frac12\log(1-r^2)-\log2$, above AR's $-\frac12\log(0.75)$.

This is not a within-one-world encoder-selection counterexample or a refutation of Appendix A.2. That appendix requires a shared finite-precision Gaussian marginal, fixed dimension/precision, representable conditional mean, and isotropic Gaussian conditionals with state-independent scalar variance per encoder. Without Gaussian conditionals it states a lower bound. Our continuous computations do not certify its common precision correction.

Seeds 101–105; 200,000 independent pairs per law/seed. Stable exact mixture log densities, marginal moments, MI Monte Carlo SE, and squared-magnitude correlation. No histogram MI fitting.

## Common-coordinate predictor control

Explicit calibrated inverse rendering gives a frozen scalar coordinate: an oracle control, not learned identification. Four two-scalar models: linear mean/constant Gaussian variance; zero-mean heteroscedastic Gaussian; equal-weight signed mixture; sign-conditioned linear mean/constant Gaussian variance. Same 16,384 training pairs, 65,536 independent test pairs, one initialization and 150-iteration L-BFGS-B cap. Report termination, test NLL/MSE in a common coordinate. Different likelihood families are different inductive biases.

## Learned encoders

Signed world, five seeds 101–105. Same 6–48–48–1 tanh scalar encoder and 2–32–32–3 predictor in every arm; 2,000 Adam updates, batch 256, LR .001, SIGReg weight .1. Arms: mean MSE; Gaussian NLL; equal-weight two-component Gaussian mixture NLL; action-conditioned Gaussian NLL. Both encoder branches get gradients. Only conditioned arm sees current action; other side input is zero. Same total parameters; all output channels enter the objective (mean sums three outputs; Gaussian sums two for mean; mixture uses two means). Equal counts/updates do not imply equal effective function classes or computation.

Fresh pairs each update, independent evaluation seeds. Save checkpoints/settings, normalized latent MSE/NLL, Gaussian KS/moments, and degree-7 polynomial probes fitted on 8,192 fresh observations and scored on 16,384 disjoint observations. Probe distance, log-distance, magnitude and sign. Sign probe is least-squares score thresholded at .5. No physical labels in encoder/predictor training. Mean density variance is calibrated separately. NLL is in each encoder's own standardized coordinate: it cannot rank physical density across encoders. Grid on distance [1,16] reports reversals and normalized sensitivity, not an injectivity certificate. Approximate SIGReg does not establish exact Gaussian marginals. No tuning on test outcomes.

## Independent-noise drift extension

Existing interior diffusion: drift $0.3\sin d+0.25a$, diffusion $0.05(1+0.1\cos d)$. Reset queries and inverse-renderer calibration are privileges; this is not encoder identification. Compare one interval .2 with intervals .02,.05,.1,.2, each using 8,192 independent endpoints total, 64 Euler substeps. Seeds 201–205. Regress increment/interval on 1,sin(d),cos(d),d/16 and constant action. Multi-interval adds interval times those state features and action-state interactions to extrapolate instantaneous drift. The true drift belongs to the fitted basis: favorable specification is explicit.

Report held-out instantaneous drift and deterministic one-second transition error. Test noisy local goal reaching: 16 opposing-drift goals/seed, 8 decisions, horizon 4, fixed bank of all 81 action sequences. Compare mean terminal squared cost against mean plus standard deviation using 16 simulated noise paths, fitted diffusion, identical candidate bank and common execution noise. Save coefficients and scores. This is a calibrated planning pilot, not a robust-control guarantee. Keep failures and do not declare universal rankings.

## Prospective follow-up after the first drift results

The unweighted multi-interval fit had larger drift RMSE in all five seeds (201–205). Before further runs, add weighted least squares using the known small-interval variance shape: multiply design and target by $\sqrt{\delta}/(1+0.1\cos d)$. This uses the simulator's diffusion shape, not an estimated general noise model. Compare single, unweighted multi and weighted multi on fresh seeds 301–305, with the same endpoints per fit and same planning protocol. Save separately in drift-weighted.json; do not pool these as preregistered original results. This tests a statistical correction motivated by observed variance, not a new encoder theorem.

## Prospective explicit same-world encoder check

Added after reading the base branch's SIGNED_OU_EXTENSION.md, before running this new check. No finished neural models are changed. Fresh seeds 401–405; 32,768 train, 16,384 calibration and 65,536 test pairs. Compare the calibrated full coordinate with Gaussianized magnitude, in the SAME signed world, without action input. Both apply Gaussian quantile clipping at epsilon 1e-4, 1e-6, 1e-8, then either no rounding or rounding to steps .5, .25, .125. Same scalar cubic spline predictor with 19 fixed interior knots on [-3,3], boundaries [-6,6] and ridge 1e-6; fitted on training only. Calibration error is recorded, not used for model selection. Report test MSE, target variance and MSE divided by variance, plus clipping fractions. This is an explicit-encoder control using known sensor calibration, not learned selection.

The continuous full-versus-magnitude information identity is checked pointwise with exact mixture/folded-normal densities, before clipping. For rounded representations compute discrete joint probabilities from known transition laws using Gaussian-Legendre quadrature within source bins; compare 64 versus 128 nodes, and report total mass and marginal errors. This avoids sample-count MI estimators. Quantization bins differ between encoders, so finite-resolution MI need not tie. Numerical results do not establish the paper's finite-precision shared-marginal or conditional-Gaussian assumptions. Verify the exact ambiguity d and 16/d, and report the irreducible distance MSE from dropping sign. Preserve all outcomes; no hyperparameter selection on held-out seeds.

# What the policy changes, and how to extend the identifiability result

These notes separate measured findings, mathematical deductions, and research proposals. The deductions below are not claims of publication novelty. The spectral viewpoint is closely related to Slow Feature Analysis and is already discussed in Appendix F of the Klindt paper.

## 1. Was ICM→OU the best?

**Follow-up:** the [curriculum controls](../curriculum-controls/INTERPRETATION.md) include five new seeds and identify two serious folded-representation failures. The three-run ranking below describes the original experiment; it does not establish that the handover is reliably best.

In these three runs, yes: the handover and the handover with entropy-bonus decay are essentially tied. Their shared-range R² values are 0.9991255 and 0.9991273 on average. The difference is too small to support a preference between them. Fixed OU averages 0.9945710. The plain handover beats fixed OU on each tested seed by about 0.00476, 0.00307 and 0.00583.

The same pattern holds on the broader fixed Gaussian OU test: about 0.9932–0.9933 for the handovers, versus 0.9802 for fixed OU. Their output variances are also closer to one. See the [complete schedule results](../schedules/output/RESULTS.md).

This establishes the ranking of these trained models on these tests. Three seeds and one simple world do not establish that the handover is the best general exploration policy. The combined schedule is marginally higher numerically, but “tied” is the useful interpretation.

## 2. First distinguish the meanings of “nonlinear”

The paper already allows an unknown **nonlinear observation map**, x=g(z). Making the renderer more complicated is not, by itself, a theoretical extension.

Three different extensions matter:

- **Non-Gaussian state coordinates:** the physical variables have a different distribution. Our distance d is positive and log-normal; z=log(d/4)/0.7 is the Gaussian coordinate to which we apply the theorem.
- **Nonlinear state dynamics:** the next state depends nonlinearly on the current state, rather than following additive Gaussian OU dynamics.
- **Behavior-dependent dynamics:** actions drawn from a learned policy change the transition rule and usually also change which states are visited.

“Linear recovery” always refers to a chosen coordinate system. A representation can be linear in log-distance and nonlinear in physical distance at the same time. An extension needs to specify which physical coordinates or equivalence class it intends to identify.

## 3. The general objective depends on two things, not one

Let q_pi(z,z') be the distribution of pairs collected under policy pi, including its starting-state and episode protocol. Let mu_0 and mu_1 be its two endpoint distributions. Define

$$
 \nu=(\mu_0+\mu_1)/2,\qquad
 q_s(z,z')=\tfrac12[q_\pi(z,z')+q_\pi(z',z)].
$$

The alignment loss is unchanged by swapping endpoints, so using q_s gives exactly the same loss. Both marginals of q_s are nu. This also matches the marginal tested when SIGReg is applied to the concatenated pair endpoints in our code.

Let S_pi be the conditional-expectation operator associated with q_s:

$$
 (S_\pi f)(z)=\mathbb E_{q_s}[f(z')\mid z].
$$

It is self-adjoint on L²(nu). For a stationary policy with stationary measure mu, nu=mu and S_pi=(P_pi+P_pi^*)/2, where P_pi is the ordinary transition operator. Thus the construction covers nonreversible policies too. It also gives a well-defined pair objective when our burn-in has not established stationarity.

For a mean-zero representation h with identity covariance under nu, the identity is

$$
 \boxed{L_\pi(h)=2n-2\sum_{j=1}^n\langle h_j,S_\pi h_j\rangle_\nu.}
$$

**Deduction:** a policy changes both the state measure nu and the operator S_pi. Keeping coverage high, or matching the state histogram, does not by itself fix which features alignment prefers.

Under the weaker mean/covariance constraints, an isolated top-n nonconstant eigenspace of S_pi determines the minimizing feature subspace, by the Rayleigh/Ky Fan variational principle. This statement assumes the required eigenfunctions and a spectral gap exist. It does not hold as an unconditional claim about every Markov process.

**Important constraint distinction:** full Gaussian output is stronger than identity covariance. The leading whitened eigenfunctions need not jointly be Gaussian, and may therefore be infeasible for the exact SIGReg-constrained problem. Spectral arguments for whitening must not silently be described as solutions of the full Gaussian problem.

## 4. Why OU gives the linear coordinates

For isotropic Gaussian OU pairs, nu=N(0,I) and the Hermite functions of total degree k have eigenvalue rho^k. Since 0<rho<1, the n linear coordinate functions have eigenvalue rho, while every nonlinear component has eigenvalue at most rho². The gap is

$$
 \gamma=\rho-\rho^2>0.
$$

The alignment objective favors the linear subspace. Those linear coordinates already have the required joint Gaussian distribution. This makes the whitening optimum feasible for the stronger constraint and recovers the paper’s orthogonal identifiability result.

That is stronger than saying “OU explores evenly” or “OU has more entropy.” It specifies the joint pair distribution and why its nonlinear features are less temporally correlated.

## 5. Why ICM is different

### A. The state distribution can require a nonlinear readout

ICM maximizes the prediction error of its evolving forward model, not the Gaussianity or spectral gap required by the identifiability theorem. In two saved runs it favored approach, concentrated near the object, and left substantial parts of the range poorly sampled.

There is a simple obstruction even before optimization: if z is non-Gaussian and A is invertible, then Az+b cannot be exactly Gaussian. In our bounded behavior world, the obstruction is stronger: every affine image of bounded log-distance is bounded, whereas a non-degenerate Gaussian is unbounded.

For a scalar continuous distribution with strictly increasing CDF F_nu, its monotone Gaussianizing map is

$$
 T(z)=\Phi^{-1}(F_\nu(z)).
$$

It is usually nonlinear. Restricting the encoder to monotone maps would force the exact Gaussianizing output to T(z) or −T(z), not an affine physical z. Without monotonicity, many other Gaussian-preserving rearrangements exist, and alignment still matters. This is a distribution argument, not a claim that our neural network has learned exactly T.

### B. The transition preferences can change independently of the state distribution

ICM’s actions induce P_pi(z'|z)=integral P(z'|z,a) pi(a|g(z)) da. Altering pi changes the pair correlations, even if the stationary state distribution is held fixed. Section 6 gives a controlled construction where Gaussian state occupancy is identical but the optimal representation is nonlinear.

### C. The finite training objective was not fully satisfied

Original ICM output variances on its own test trajectories were about 0.76, 18.20 and 13.84, versus the target 1. These are observable regularization/generalization failures. A lower distance-recovery score in those runs cannot be attributed solely to a changed population optimum.

The 17-frequency SIGReg implementation is a finite approximation. It is not an exact Gaussian constraint. Weakly observed tails, batch noise, finite model capacity and optimization can all matter. We have not isolated their separate contributions with the present experiments.

### D. Entropy is not a sufficient description of a policy

For our three actions, away from boundaries,

$$
 \mathbb E[\Delta d\mid d]=0.25\,[\pi(\text{away}\mid d)-\pi(\text{approach}\mid d)].
$$

The action probabilities (0.6,0.2,0.2) and (0.2,0.2,0.6), ordered approach/stay/away, have exactly the same entropy, but opposite drift. An entropy number alone therefore cannot determine occupancy or temporal features. Lowering a small entropy bonus need not make the process OU-like; our measured action entropies changed little under the tested bonus decay.

## 6. A controlled mathematical counterexample: same Gaussian states, nonlinear optimum

**This is a deduction and a separate 2D simulation, not an ICM result or a newly claimed discovery.** It is a useful control for separating occupancy from transitions. It lies outside the paper’s independent additive-OU transition assumptions.

Let z~N(0,I_2), and define a radius-dependent rotation

$$
 F(z)=R_{c\|z\|^2}z,\qquad c\ne0.
$$

F preserves the radius and rotates the uniform angle on every circle. It therefore preserves the entire standard Gaussian measure. Its inverse is the rotation with strength −c; it is an invertible smooth map.

Define the physical transition by

$$
 z'=F^{-1}\!\left(\rho F(z)+\sqrt{1-\rho^2}\eta\right),
 \qquad \eta\sim N(0,I_2).
$$

Both endpoints are exactly N(0,I_2). In coordinates u=F(z), however, the process is standard OU. For any representation h, put k=h composed with F^{-1}. Then the Gaussian-output constraint and alignment are exactly the OU constraint and alignment for k. The original theorem therefore implies

$$
 \boxed{h_*(z)=QF(z),\quad L_*=4(1-\rho),\quad Q\in O(2).}
$$

Every Gaussian-compatible linear map h(z)=Qz is feasible, but strictly suboptimal when F is genuinely nonlinear. Otherwise k=QF^{-1} would tie a linear OU optimum while being nonlinear, contradicting the equality case of the OU theorem.

This is an explicit example where matching Gaussian marginals does not ensure linear recovery of the specified physical coordinates. It also shows the coordinate ambiguity: without structural or action information, the same observed process can be described as nonlinear physical dynamics or OU in warped coordinates.

The transition can be realized by a fully actuated theoretical controller with action a=z'−z. We have not trained ICM to implement it or added it to the triangle-agent benchmark.

Run the numerical check:

```sh
python behavior-policy-experiments/analysis/check_policy_effects.py
```

The [saved numbers](output/policy_counterexample.json) and [figure](output/policy_counterexample.png) compare the two losses. Gaussianity follows from the construction; sample mean and covariance checks alone would not prove it.

## 7. A limited robustness lemma for behavior-policy mixtures

**Deduction under explicit matched-marginal assumptions.** Let q_0 be Gaussian OU pairs and q_K be any other pair distribution with the *same* Gaussian endpoint marginal mu. Consider q_alpha=(1−alpha)q_0+alpha q_K, with 0<=alpha<1. Let h have zero mean and identity covariance under mu, and suppose

$$
 L_\alpha(h)\le L_\alpha(z)+\varepsilon.
$$

For any such normalized representation, 0<=L_K(h)<=4n. Rearranging the inequality gives the OU excess loss bound

$$
 \delta_0=L_0(h)-2n(1-\rho)
 \le \frac{4n\alpha+\varepsilon}{1-\alpha}.
$$

The OU Hermite argument then bounds the total nonlinear variance E_nl:

$$
 E_{\rm nl}\le\frac{\delta_0}{2\rho(1-\rho)}
 \le\frac{4n\alpha+\varepsilon}{2(1-\alpha)\rho(1-\rho)}.
$$

With exact whitening, writing h=Mz+r gives MM^T+Cov(r)=I. The nearest orthogonal linear map has error at most 2 E_nl, since the squared deviations of M’s singular values from one are at most the corresponding variance deficits.

This gives an elementary route to approximate recovery as the non-OU fraction and optimization error go to zero. The bound can be loose or vacuous for large alpha. It is not a theorem about gradient descent, curriculum tracking, or our actual ICM mixture: **the actual ICM and OU state marginals differ**, so its assumptions do not hold there. Correcting that mismatch is part of the proposed research program.

## 8. Why the handover could beat fixed OU in finite training

Both runs end with the same OU pair generator, loss weight, and model class. Their ideal late-stage population optimum is the same. A population identifiability argument therefore cannot explain a different best achievable optimum.

The observed advantage must be investigated as a finite-data/training-path effect: early pairs, warm-started features, exposure to different states, the timing of the learning-rate decay, or optimizer history. Our experiment keeps update count and final settings fixed, but does not isolate those mechanisms. “Curiosity provides a better initialization” is a hypothesis, not a conclusion already established by the ranking.

The next controls should be random→OU, OU→OU with the same phase boundaries, matched OU exposure, and controlled optimizer/learning-rate resets. Freeze the chosen schedules, then evaluate on additional independent seeds and worlds. Decreasing the entropy bonus added essentially nothing at the original starting weight; other entropy weights remain untested.

## 9. A research program, ordered by usefulness

| Direction | Precise target | Small next experiment | What would count as progress? |
| --- | --- | --- | --- |
| Separate occupancy from transition effects | Characterize when S_pi has the physical linear coordinates as its isolated leading subspace | Compare ordinary and conjugated OU with identical Gaussian states, then train the same encoder | Recover z for OU and F(z) for conjugated OU; relate failure to the operator rather than the histogram |
| Approximate policy robustness | Bound recovery by occupancy mismatch, transition-operator mismatch, spectral gap and optimization error | Start with the matched-Gaussian mixture in Section 7, then introduce controlled marginal changes | A bound with explicit assumptions that predicts which perturbations are harmless |
| Action-conditioned identification | Use known actions or interventions to distinguish physical coordinates from arbitrary warps | A two-dimensional agent with independently controlled sideways position and log-distance | Derive and test conditions under which action information eliminates nonlinear coordinate ambiguity |
| Non-Gaussian factors | Identify meaningful coordinates up to specified componentwise nonlinearities | Same-distribution independent factors with reversible scalar dynamics; compare whitening-only and full SIGReg | A proof that states the right equivalence class and a compatible output constraint |
| Finite-training curriculum theory | Explain improved convergence without inventing a different final optimum | Random→OU, ICM→OU, OU→OU, matched exposure and learning-rate/optimizer controls | Reproducible gains across new seeds plus a tractable optimization model |
| Predictive subspaces | Recover a task-relevant part of a larger state | Add a rapidly changing nuisance variable and reduce representation dimension | State which subspace is selected and why it retains the controllable variables |

### 9.1 Occupancy correction requires correcting pairs too

If a target pair distribution q_* is absolutely continuous with respect to q_pi, then

$$
 \mathbb E_{q_*}[\ell]=\mathbb E_{q_\pi}[(dq_*/dq_\pi)\ell].
$$

This suggests off-policy alignment and weighted output-distribution constraints. State-only importance weights are generally insufficient: they change occupancy but not the conditional transition rule. The Gaussian regularizer must also target the corrected endpoint distribution.

Our bounded three-action transitions and full Gaussian OU transitions do not have matching support; in fact the former are concentrated on action-defined curves in pair space. One cannot obtain the latter by finite importance weights. A valid test needs shared continuous transition noise and adequate support, or an explicitly restricted target. Estimating the required density ratios without true latent labels is an additional problem, not a solved implementation detail.

### 9.2 A useful anisotropic-Gaussian building block

For independent Gaussian OU coordinates with correlations rho_i in (0,1), all linear modes have eigenvalues rho_i and every nonlinear Hermite mode has eigenvalue at most rho_max². Therefore

$$
 \rho_{\min}>\rho_{\max}^2
$$

is a sufficient condition for the top-n nonconstant subspace to remain linear. Since that linear solution is jointly Gaussian, it is feasible for full Gaussian regularization as well. Exact isotropy is sufficient, but is not necessary under this gap condition. Appendix F already discusses the corresponding continuous-time rate condition, so this should be treated as a building block rather than a claimed new contribution.

With exact whitening and excess alignment delta above the linear optimum, spectral separation bounds nonlinear variance by delta/[2(rho_min−rho_max²)]. Below that sufficient-gap threshold, the whitening optimum can favor higher-order slow modes; this does **not** by itself prove that those modes solve the more restrictive full-Gaussian problem.

### 9.3 Non-Gaussian extensions need a compatible objective

Appendix F, page 30, explicitly suggests a shared monotonic transformation of independent, identically distributed non-Gaussian factors. Under appropriate reversible diffusion, boundary, and spectral-gap assumptions, the first scalar eigenfunctions can be monotonic, and their span can be identified by whitening-based Slow Feature Analysis.

But those eigenfunctions need not have Gaussian distributions. A direct extension to unchanged full SIGReg therefore has a feasibility problem. Options include identifying the nonlinear spectral coordinates under whitening only, choosing a compatible target distribution, or explicitly modeling a transport to Gaussian coordinates and defining the desired recovery class accordingly. Gaussianizing a variable does not automatically turn arbitrary dynamics into OU.

This direction builds on [Sprekeler, Zito and Wiskott (2014)](https://www.jmlr.org/papers/v15/sprekeler14a.html), rather than starting from an entirely new idea. Regularity and boundary assumptions must accompany any monotonicity claim.

### 9.4 Actions and multiple policies as additional information

The paper’s discussion and Appendix D identify action-conditioned dynamics as unfinished work. Policy identity, action, or intervention labels could provide information that plain paired-view alignment discards. Existing [auxiliary-variable nonlinear ICA](https://proceedings.mlr.press/v89/hyvarinen19a.html) and [identifiable latent-variable models](https://proceedings.mlr.press/v108/khemakhem20a.html) offer related routes, under their own conditional-independence and variation assumptions. Their guarantees do not automatically apply just because we record an action label.

A useful target is a theorem about a controlled family P_a, requiring enough independent variation across actions or interventions. The target coordinate system and allowed residual transformations must be specified. Known physical displacements might anchor axes and units; unknown actions generally leave more ambiguity.

### 9.5 The next agent world should have two continuous coordinates

Use sideways position u and log-distance l, with the same front-facing triangle. The apparent center is proportional to u/d and the apparent size to 1/d, so both coordinates remain recoverable from ideal observations. Give the agent separate sideways and approach/retreat controls.

This permits genuine coordinate mixing, anisotropic dynamics, different action coverage, and the Gaussian-preserving warp test. Adding only discrete shape labels would not test the paper’s continuous Gaussian-factor result. Add nuisance factors later to study which predictive subspace is learned when the output has fewer dimensions than the full world.

## 10. Recommended next steps

First use the controlled Gaussian-marginal example to isolate transition effects, and run the matched-exposure curriculum controls on additional seeds. Then move to the two-coordinate agent and specify an action-conditioned identifiability question. Those steps make a stronger research claim possible than adding more policies to the current scalar benchmark without isolating what changes.

The source paper is *When Does LeJEPA Learn a World Model?*, supplied in the workspace, especially Sections 3–5, 7–8, Appendix D.2 and Appendix F. The mathematical constructions and limited lemmas above are deductions presented for this project, not externally verified novelty claims.

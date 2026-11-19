# Evaluate occupancy's contribution to the slope at fold birth

Completed November 15, 2026. This diagnostic evaluates seed 131, gain 4, immediately before update 204 → 205, at physical distance 16. It imports the original study through its canonical source path; it creates no competing training implementation or checkpoints. [Code](run.py), [numerical results](output/results.json), [figure](output/integral.png).

## What is evaluated

Write q = log d, h(q) = fθ(x(q)), and K(q,s) = ∇θh(q)ᵀ∇θh(s). At the saved before-state, evaluate

    I_p(q) = ∫ p(s) G_p(s) ∂q K(q,s) ds.

G_p is the derivative of the characteristic-function discrepancy with respect to an output, before multiplying by that output's probability weight. The regularizer uses 17 original frequency knots and the original buffers. The objective's multiplier is held at 1024, the training endpoint batch size, for every comparison; it is not increased to the full dataset size. The regularizer coefficient 0.1 and comparison SGD learning rate 0.001 are applied separately: Δh′ ≈ −0.0001 I_p.

The integral is evaluated exactly up to numerical arithmetic over the **stored empirical measure**, using all 32,768 source endpoints. This is not integration against a known continuous population density. The endpoints span distances 1.00062–13.73643, so the query at d=16 is beyond the stored observations' coverage. Of 32 equal log-distance bins, 31 are occupied. The balanced contrast gives those 31 bins equal probability while retaining their within-bin observations; no missing observations are synthesized. Distance is used for this diagnostic weighting, not supplied to the encoder.

For each observation i, forward-mode differentiation computes ∂qK(q,qᵢ) as Jθh(qᵢ) · ∇θh′(q). Its weighted sum is independently checked against ∇θh′(q) · ∇θL_R using reverse-mode differentiation of the complete objective. Profiles at 65 query distances and binwise signed contributions are retained in the JSON.

## Results

| Measure at identical saved weights | Integral I_p at d=16 | Predicted regularizer-only SGD slope change |
| --- | ---: | ---: |
| Native empirical occupancy | +38.7795253 | −0.00387795 |
| Equal occupied log-bin probability | −5057.0907158 | +0.50570907 |

Thus changing occupancy reverses the local regularizer direction at these weights. The finite-difference and finite-step checks are stored separately: a linearized change is not a finite-step guarantee. The native population-plug-in regularizer contribution alone does not exceed the initial positive slope of 0.00415023 at the comparison learning rate.

Occupancy affects the outer integral and the global sine/cosine moments. Keeping native moments while balancing only the outer measure gives I = +844.498788; changing only the moments gives I = −6429.101331. These hybrid contractions are algebraic diagnostics, not separately trained objectives. Native probability below d=2 is 84.63745%, but its signed contribution to I is −4.98483, while the remaining observations contribute +43.76435. Mass alone does not determine a bin's signed effect; the moments and response kernel matter.

## The real Adam update needs history

The exact historical minibatches are reconstructed and one float32 Adam update is replayed from the saved optimizer state. It matches the saved after-state with maximum parameter error **zero**.

| Slope measurement or first-order contribution at d=16 | Value |
| --- | ---: |
| Before slope | +0.004150231 |
| After slope | −0.000832708 |
| Measured full slope change | −0.004982939 |
| First-order actual-update prediction | −0.004968168 |
| Alignment Adam component, including history | −0.000011922 |
| Current SIGReg gradient within Adam | +0.001426908 |
| Historical SIGReg momentum within Adam | −0.006383163 |

The current minibatch's unpreconditioned integral is −139.440067, which differs even in sign from the full-data plug-in integral. That batch's regularizer-only SGD slope change would be approximately +0.0139440. Finite-batch global moments, sample variation, and Adam history therefore cannot be replaced by the full-data integral when explaining the actual step.

To compute the current Adam contribution, fix the actual combined second-moment denominator D after this step. If a(q)=∇θh′(q), β₁=0.9 and t=205, its samplewise contraction is

    −η·0.1·(1−β₁)/(1−β₁ᵗ) Σᵢ αᵢ G_batch(qᵢ) Jθh(qᵢ)ᵀ D⁻¹ a(q).

Subtract this from the saved SIGReg Adam-component contraction to obtain the historical contribution. It aggregates gradients evaluated at past weights and past minibatches, transported through the current denominator and slope sensitivity. This is a decomposition with the actual shared denominator, not independent optimizers. A float64 analytic contraction is checked against the replayed float32 current gradient. A central finite difference along the actual update independently checks the mixed derivative. The total linearization misses the measured slope change by about 1.48e−5 (0.30% of the change).

**Interpretation:** occupancy changes the regularizer force at fixed weights. At the actual fold-creation step, accumulated SIGReg momentum drives the reversal while the current regularizer gradient opposes it. This strengthens the explanation to occupancy-dependent gradients plus optimizer memory; it contradicts an explanation based solely on the current population gradient crossing zero. Attribution of historical momentum to particular past occupancy bins would require extending this calculation backward through training and has not been performed here.

## Reproduction and limits

From the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-integral/run.py
```

Requires the original source and birth-event checkpoints, NumPy, PyTorch and Matplotlib. Input/checkpoint and script SHA-256 hashes are recorded in the JSON. The script asserts the integral/autograd identities, exact Adam replay, a current-gradient contraction check and a central finite-difference check. It also evaluates finite regularizer-only SGD steps without using them to claim whole-domain order.

This is one previously inspected seed and update, not an independent cohort, continuous-domain certificate, full training-history occupancy attribution, or a universal concentration threshold. Full-data plug-in moments do not equal the expectation of a finite-batch squared-moment objective. No new policy or encoder training sweep was run. Next work is to accumulate per-bin Adam contributions over the preceding updates and repeat paired occupancy contrasts across seeds.


## November 19: Historical attribution and actual optimizer-state intervention

The [history tracer](momentum_history.py) replays all 205 updates exactly and
contracts each past sample's SIGReg gradient through the actual birth-step
second-moment denominator and before-step slope sensitivity. It records
signed contributions by update and log-distance bin; these are linear
attributions along the original path, not independent training histories.
Their historical sum is −0.0063831652, agreeing with the saved numerator
contraction within 3.7e−10. Per-sample sums agree with parameter-gradient
contractions within 4.4e−11. [Measurements](output/momentum_history.json),
[figure](output/momentum_history.png).

More decisively, four actual one-step Adam forks start with identical saved
before-weights, current gradients, second moments, step count and learning
rate. Only the historical first-moment numerator is changed:

| First-moment history | Slope at distance 16 after update | Detected turn |
| --- | ---: | --- |
| Original | −0.000832708 | 15.8112 |
| Remove SIGReg history | +0.005567166 | None |
| Remove all history | +0.005577816 | None |
| Remove alignment history | −0.000822330 | 15.8136 |

The full fork reproduces saved parameters bit-for-bit. Removing SIGReg
momentum prevents this specific fold-creation step, while removing alignment
momentum does not. Every fork is checked on a 20,001-point float64 derivative
grid; fork checkpoints (`output/momentum_history_forks.pt`) remain local.
This identifies a causal role for accumulated SIGReg momentum at this event.
It does not show permanent prevention, that every fold needs momentum, or that
occupancy is the only upstream cause. The first turn is outside the stored
source support and later moves inside it. Run with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/occupancy-integral/momentum_history.py
```


## Replication across all five historical failures

The [fixed-protocol replication](BIRTH_PROTOCOL.md) scans every update through
500 in all five original final-folded runs, then requires their full
4,000-update replays to match historical final weights exactly. It finds eight
ordered-to-folded events and retains all of them. All 32 intervention models
pass independent dense checkpoint-reload checks. Removing SIGReg history
prevents six event steps; removing alignment history prevents none.

The exceptions matter: source 139 has a first-update transient without any
history, and source 132's update 368 still folds after removing all first
moment history. Therefore momentum is a demonstrated trigger for several
births, not a universal prerequisite. One-step prevention also does not imply
permanent repair. [Complete event table](output/BIRTH_RESULTS.md),
[measurements](output/birth_replication.json),
[figure](output/birth_replication.png), [audit](output/birth_verification.json).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/occupancy-integral/birth_replication.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/occupancy-integral/report_births.py
```

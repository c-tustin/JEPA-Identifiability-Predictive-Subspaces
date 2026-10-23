# Nonlinear dynamics without adding state coordinates

The triangle still changes as $x=V/d$. We allow nonlinear drift and state-dependent noise in the same distance variable:

$$
dD_t=[0.3\sin D_t+0.25A_t]dt+0.05(1+0.1\cos D_t)dW_t.
$$

The [proof](PROOF.md) shows that constant physical action effects can identify distance up to an affine map among smooth invertible coordinates. It also bounds coordinate error using drift-fit error, state coverage and conditional action variance. **It does not prove that unconstrained JEPA finds such coordinates.**

The numerical prototype trains a smooth ordered encoder (positive combinations of softplus hinges) from paired approach/retreat interventions, using common simulator noise. It uses 4,000 updates of 512 paired queries per seed, Euler interval 0.02, and interior distances 1.2–15.8. Its target is the known action coefficient, not distance labels. These are simulator intervention queries, not ordinary policy trajectories. It avoids boundaries and uses physical-distance recovery only for evaluation. Output Gaussianity is not imposed on bounded distance. These additional structural assumptions are explicit.

```sh
python behavior-policy-experiments/nonlinear-control/check_actuation.py
```

Use `--quick --output /tmp/nonlinear-smoke` for an execution check. [Results](output/RESULTS.md) include three fixed seeds and a separate analytic check of the approximate bound. A future finite-sample JEPA theorem must still connect its objective to the conditional-drift risk, establish information preservation, and control sampling and discretization errors.

All three smooth encoders recover physical distance with $R^2>0.9999999$ in this restricted prototype. This illustrates the anchored model class; it does not validate a general nonlinear-JEPA guarantee.

# OU alignment with broad observations for SIGReg

**Completed and verified. [Concise results](output/SUMMARY.md) · [All tests](output/RESULTS.md) · [Graphs](output/FIGURES.md).

The broad pool improves affine recovery modestly, but compresses the raw output scale and worsens recovery up to sign. It produces no detected planning improvement or fold reduction.

We tested the same observation-pool change as the ICM control, using the original Gaussian-latent OU pair source. Keep OU alignment pairs fixed and compare:

1. Original OU training: both losses use the same pair batch.
2. Independent OU views for SIGReg.
3. Independent broad views for SIGReg: log-uniform distance 1–16.

[Fixed protocol](PROTOCOL.md). The broad-versus-independent-OU comparison isolates the pool change. The original condition checks whether it improves the actual regular-OU recipe as well.

We evaluate both linear recovery on distances 1–16 and on held-out Gaussian states. Raw recovery error up to sign is also checked: fitting a new scale/offset for R² can conceal a failure of orthogonal recovery. Planning uses shared physical-action data, because Gaussian OU pairs do not have the three physical action labels.

Sixty paired seeds, 180 encoders and 180 planning evaluations. These reuse the earlier ICM study's initializations and common observation pools, allowing explicit recipe comparisons with its ICM/broad-pool controls. All 180 fits, Gaussian recovery diagnostics and physical planning rollouts are verified, including historical and resumed-training gates.


From the repository root with Python 3.12:

```sh
python3 -m pip install -r behavior-policy-experiments/policy-zoo/ou-broad-observations/requirements.txt
python3 behavior-policy-experiments/policy-zoo/ou-broad-observations/run_ou.py --run --workers 6
python3 behavior-policy-experiments/policy-zoo/ou-broad-observations/finish_ou.py
python3 behavior-policy-experiments/policy-zoo/ou-broad-observations/write_summary.py
```

The published code lock is already present. Repeating `--run` after a stopped process resumes the same fixed cohort. Encoder/Adam checkpoints save every 500 updates. Completed predictors are reused. The finisher waits for every run, then performs the locked analysis, complete-source/geometry/planning verification and an exact resumed-training replay. No partial-cohort tests are substituted.

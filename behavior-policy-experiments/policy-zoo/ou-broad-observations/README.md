# OU alignment with broad observations for SIGReg

Test the same observation-pool change as the ICM control, using the original Gaussian-latent OU pair source. Keep OU alignment pairs fixed and compare:

1. Original OU training: both losses use the same pair batch.
2. Independent OU views for SIGReg.
3. Independent broad views for SIGReg: log-uniform distance 1–16.

[Fixed protocol](PROTOCOL.md). The broad-versus-independent-OU comparison isolates the pool change. The original condition checks whether it improves the actual regular-OU recipe as well.

We evaluate both linear recovery on distances 1–16 and on held-out Gaussian states. Raw recovery error up to sign is also checked: fitting a new scale/offset for R² can conceal a failure of orthogonal recovery. Planning uses shared physical-action data, because Gaussian OU pairs do not have the three physical action labels.

Sixty paired seeds, 180 encoders and 180 planning evaluations. These reuse the earlier ICM study's initializations and common observation pools, allowing explicit recipe comparisons with its ICM/broad-pool controls. No result is claimed until the complete cohort and verification finish.

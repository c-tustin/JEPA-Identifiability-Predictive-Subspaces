# Protocol 2: how much of the fold's downstream cost do trained repairs recover?

Fixed November 25, 2026, before any fit in this stage. The SHA-256 is recorded in `output/repairs/protocol.sha256` before
the first run. The downstream harness, predictor seeds and goals are exactly those of [Protocol 1](PROTOCOL.md).

## Encoders (all existing checkpoints; no encoder is retrained)

Every encoder is reduced to its code curve on the 20,001-point uniform log-d grid using its own model class, then
evaluated through the same interpolating wrapper. This is exact up to interpolation, because the world is deterministic
and every code is a function of d. MLP checkpoints load into the original `Encoder` (identical to `ScaledEncoder(4)`).
Splines are evaluated from their stored knot values, piecewise-linear in apparent height 1/d.

| Family | Repair | Baseline | Sources |
| --- | --- | --- | --- |
| Scratch, occupancy | `native-weighted_moments` (weighted moments) | `native-ordinary` (bit-identical to the crossing diagonal) | 130–139 |
| Scratch, occupancy + order | `scratch-g4-weighted_order` | same | 130–139 |
| Hard order | `projected-native`, `positive-native` splines | same | 130–139 |
| Mining | `hard-mining/<s>-c100` (global successor mining) | `runs/<s>-sigreg-c0` from the same study | 130–144 |
| Late repair | `repair-g4-weighted_strong`; `repair-g4-weighted` | `repair-g4-ordinary` (ordinary continuation) | 131, 132, 133, 136, 139 |

Every baseline also gets its monotone rearrangement (the oracle of Protocol 1). Sources 140–144 have no stored
random-policy set; it is generated with the original recipe `collect(c, 'random', c.train_episodes, 510000 + seed)`.

Information use: mining uses only observed transitions. Weighted moments use privileged distance bins, the order
penalties use simulator-ordered probes, and the splines use the known ordering coordinate (and a different model class).

## Measures

Same as Protocol 1: native held-out next-observation MSE, log-d readout MSE, and shared- and native-data MPC planning on
the standard and near goal sets. Three replicates per encoder.

**Gap recovered** for a source whose baseline is materially folded: G = (baseline − repair) / (baseline − oracle). G = 1
means the repair matches the oracle; G > 1 means it beats it.

## Pre-registered predictions

Unit: source. "Folded sources" means sources whose own baseline is materially folded.

- **R1.** Weighted moments: G ≥ 0.8 for native next-obs MSE and for near-goal shared planning error in at least 4 of 5
  folded sources.
- **R2.** Weighted + order: the same criterion.
- **R3.** Each hard-order spline: the same criterion.
- **R4.** Global mining: G ≥ 0.8 on both measures in every folded source where the mining encoder is not itself
  materially folded.
- **R5.** Late strong repair: G ≥ 0.8 on both measures in at least 4 of 5 sources. Late weighted-only repair, which stays
  folded: G < 0.5 on both in at least 4 of 5.
- **R6.** On sources whose baseline is not folded, no repair raises the source-mean near-goal shared planning error by
  more than 0.05.

## Limits fixed in advance

Existing checkpoints were trained by different studies with different hyperparameter-selection histories (the mining and
weighted coefficients were partly tuned on these development sources). This stage measures downstream value only; it
does not re-establish fold rates.

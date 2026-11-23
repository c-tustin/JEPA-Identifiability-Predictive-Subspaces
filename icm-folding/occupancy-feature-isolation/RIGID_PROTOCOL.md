# Causal test: does the differential pull along the tail cause the fold?

Fixed November 22, 2026, after the [stretch test](STRETCH_PROTOCOL.md) and before any fit below was trained.

## Claim under test

The fold is the tail's far end overtaking its middle. The force landscape pulls the far end back harder than the middle,
and the encoder lets the far end move on its own. The stretch test ruled out "the spike can't stretch" as the cause at
m ≤ 0.75.

## Design (24 fits)

Baseline `tanh(4x)` encoder, identical initialization, data, plan and loss value. Only the gradient that SIGReg sends to
one group of codes is changed. Within each batch, for the codes y_g of group g (far: d ≥ 2; near: d < 2), SIGReg receives

  y_g' = sg(y_g) + (μ − sg(μ)) + (sg(y_g) − sg(μ))·(σ/sg(σ) − 1),

where μ and σ are the group's batch mean and standard deviation, and sg is stop-gradient. The forward value is unchanged
(y_g' = y_g), but SIGReg's force on that group is projected onto shifting and rescaling the group as a block. No
differential force reaches individual codes within the group. Alignment is untouched.

| Condition | Group made rigid for SIGReg |
| --- | --- |
| tail-rigid | far codes (d ≥ 2) |
| spike-rigid | near codes (d < 2) |

| Setting | Values |
| --- | --- |
| Laws | m0.625, m0.75, m0.875 (as in the stretch test) |
| Initializations | 131, 133, 135, 137 |
| Training | as in the geometry test |

Baseline fits on these laws fold at every initialization tested.

## Predictions (junction fold means backward fraction > 0.1)

1. tail-rigid: no fold in at least 10 of 12 cells.
2. spike-rigid: fold in at least 10 of 12 cells.

If (1) fails, differential pull along the tail is not needed for the fold. All cells are reported, and failures are kept.
One architecture; an empirical check, not an analytic guarantee.

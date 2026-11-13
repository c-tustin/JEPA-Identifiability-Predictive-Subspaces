# Does broad SIGReg exposure help ordinary OU?

Sixty paired seeds, three OU conditions. The alignment pairs are identical; the SIGReg formula/weight and training budget stay fixed. [Fixed plan](../PROTOCOL.md).

| OU training condition | Bounded-grid R² | Held-out Gaussian R² | Raw orthogonal MSE ↓ | Folds / 60 | Physical goal error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original coupled OU | 0.994170 | 0.979435 | 0.026249 | 0 | 0.074870 |
| Independent OU observations | 0.996951 | 0.985627 | 0.018254 | 0 | 0.076432 |
| Independent broad observations | 0.997607 | 0.988763 | 0.075353 | 0 | 0.075065 |

R² permits a fitted scale and offset. Raw orthogonal MSE permits only a sign change, matching a one-dimensional rotation. Thus high R² alone is not sufficient evidence of orthogonal recovery. The held-out Gaussian states include distances outside 1–16; OU alignment is not clipped. Broad SIGReg observations cover only 1–16.

| Comparison | Outcome | Mean benefit [descriptive 95% interval] | Better / worse / tie | Holm p |
| --- | --- | ---: | ---: | ---: |
| broad_vs_ou_observations | affine_log_r2 | +0.00065628 [+0.000281325, +0.00099595] | 48/12/0 | 3.184e-05 |
| broad_vs_ou_observations | gaussian_r2 | +0.00313601 [+0.00258161, +0.00367351] | 54/6/0 | 1.07e-09 |
| broad_vs_ou_observations | orthogonal_mse | -0.0570994 [-0.0578578, -0.0563647] | 0/60/0 | 2.776e-17 |
| broad_vs_ou_observations | goal_error | +0.00136719 [-0.00904948, +0.0115251] | 27/26/7 | 1 |
| broad_vs_ou_observations | folded | +0 [+0, +0] | 0/0/60 | 1 |
| broad_vs_original | affine_log_r2 | +0.00343746 [+0.00298143, +0.00385388] | 56/4/0 | 1.09e-11 |
| broad_vs_original | gaussian_r2 | +0.00932769 [+0.0086164, +0.0100384] | 60/0/0 | 2.776e-17 |
| broad_vs_original | orthogonal_mse | -0.0491048 [-0.0499777, -0.0482455] | 0/60/0 | 2.776e-17 |
| broad_vs_original | goal_error | -0.000195313 [-0.0119141, +0.0115234] | 29/27/4 | 1 |
| broad_vs_original | folded | +0 [+0, +0] | 0/0/60 | 1 |
| icm_broad_vs_ou_original | affine_log_r2 | +0.00337623 [+0.00298924, +0.00374086] | 57/3/0 | 8.13e-13 |
| icm_broad_vs_ou_original | goal_error | -0.00110677 [-0.0115885, +0.00963542] | 25/28/7 | 1 |
| icm_broad_vs_ou_original | folded | +0 [+0, +0] | 0/0/60 | 1 |
| icm_broad_vs_ou_broad_observations | affine_log_r2 | -6.12379e-05 [-0.000144475, +3.04628e-05] | 18/42/0 | 0.02403 |
| icm_broad_vs_ou_broad_observations | goal_error | -0.000911458 [-0.00572917, +0.00397135] | 23/22/15 | 1 |
| icm_broad_vs_ou_broad_observations | folded | +0 [+0, +0] | 0/0/60 | 1 |

Positive benefit means the first condition is better. All 16 tests use one Holm correction. Tests are two-sided sign/McNemar tests of direction; the unadjusted bootstrap intervals describe mean effects. Nonsignificance is not equivalence.

The broad-versus-independent-OU contrast isolates the observation pool. Original-versus-independent-OU also changes sampler/gradient implementation; its descriptive comparison is in the JSON.

ICM comparisons reuse the preceding native-ICM-alignment/broad-SIGReg runs on the same 60 seeds. These compare fixed recipes with different transition laws and step scales, not identical stationary joint distributions. OU supplies no physical action labels; every planning evaluation trains its predictor on the same shared physical-action data.

[Exact results](INFERENCE.json) · [Verification](VERIFICATION.json) · [Graphs](FIGURES.md) · [Code/time lock](LOCK.json)

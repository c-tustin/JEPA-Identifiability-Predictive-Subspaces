# Nonlinear dynamics: the short version

**Follow-up completed:** [matched coverage, actual nonlinear goal reaching, and independent-noise drift estimation](EXTENSION_SUMMARY.md). The results below describe the initial trajectory study.

**Completed:** 36 encoders, three new seeds, the same triangle and one physical distance. The original flexible encoder is trained from ordinary trajectories, without the ordered architecture or shared-noise action pairs used in the earlier prototype.

The world has nonlinear drift $0.3\sin d$, constant physical action effect $0.25a$, and state-dependent noise. A matched zero-drift world is the control. Actions are held for one unit of simulated time. [Exact settings](TRAJECTORY_PROTOCOL.md).

Mean physical-distance $R^2$ on the shared interior grid:

| Behavior policy | Nonlinear: raw SIGReg | Nonlinear: balanced SIGReg |
| --- | ---: | ---: |
| Random actions | 0.9917 | 0.4670 |
| Persistent actions | **0.9975** | 0.6536 |
| Action-balanced exploration | 0.9916 | 0.4036 |

**Persistent actions with raw SIGReg look strongest in this comparison**, with no detected grid reversals in any of the three seeds. These are representation scores; the [subsequent planning comparison](EXTENSION_SUMMARY.md) evaluates physical goal reaching separately.

Balancing is not a universal repair. In the zero-drift control it favors log-distance recovery (about 0.991–0.992), while raw sampling favors physical-distance recovery (about 0.978). Under nonlinear drift, balancing strongly compresses broad distance intervals. The largest detected reversal is 2.09% of output range; reversal counts alone exaggerate the size of the shallow wiggles.

Coverage is a major unresolved factor. Zero-drift trajectories visit all 32 sensor bins. Nonlinear trajectories visit 11–12 with random actions, 16 with persistent actions, and 10–11 with action balancing. Rebalancing cannot populate missing bins. Good scores across those gaps are generalization by the encoder, not a guarantee that the trajectory data identify every distance.

The [follow-up](EXTENSION_SUMMARY.md) now matches source-state/action coverage across worlds and evaluates action-conditioned prediction and actual goal reaching. It also explains the remaining gap between finite-time prediction and the constant-action-effect mechanism in the [proof](PROOF.md). Three seeds and this passive loss do not establish a nonlinear identification theorem.

[Full table and figure](output-trajectories/RESULTS.md) · [Why coordinate warping differs from folding](TRAJECTORY_INTERPRETATION.md) · [Run or reevaluate](README.md)

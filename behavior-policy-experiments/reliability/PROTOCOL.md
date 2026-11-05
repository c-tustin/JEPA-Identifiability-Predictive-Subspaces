# Fixed protocol

Recorded before full-size results from this study were inspected. All scheduled settings are fixed, all final checkpoints are evaluated, and no distance-labelled validation is used to change them. New seeds 8–12 are distinguished from the selected diagnostic seeds 3 and 7. Five validation seeds cannot precisely establish failure probabilities or a universal policy ranking.

## Representation repairs

The ordinary schedule holds its first source through 25% of epochs, increases OU over the next 50%, then holds all OU. The early schedule instead transitions from 10% to 50%. The OU-floor schedule uses 20% OU initially and increases that fraction to one on the original timing. Each change is tested separately. Long controls double the entire original schedule and optimizer budget, not just the last phase.

Index streams are replayed from the original study and are independent of encoder initialization. Original transition-mask draws are preserved; additional endpoint draws for changed schedules use an independent stream. Ordinary matched-seed ICM/OU controls reproduce the old mathematical training protocol. No previous final encoder is used to initialize a repair.

Balanced replay resamples complete ICM pairs with probability inversely proportional to the count of their source apparent-size bin, retaining 16,384 stored pairs. There are 32 uniform log-apparent-size bins. Empty bins cannot be repaired by replay. The monotone encoder is an increasing 65-knot linear spline of observed height, with positive learned increments and linear extrapolation; its initialization is affine in apparent height, not log-distance. It changes capacity and encodes an ordering prior.

## Exploration policies

All learned policies use the same source-derived actor–critic equations, actor architecture, actor optimizer, entropy bonus and interaction budget as the original ICM policy. The intrinsic rewards differ and have not been independently tuned to optimal settings.

| Policy | Intrinsic reward / action rule |
| --- | --- |
| ICM | Original clipped forward-feature prediction error |
| ICM + random actions | Same reward; sample and train log probabilities from a 20% uniform mixture |
| ICM + coverage | Original ICM reward plus $0.1/\sqrt{1+N_b}$, clipped at one |
| Coverage | Only $0.1/\sqrt{1+N_b}$ |
| RND | Error predicting a fixed random 32-feature target from the successor view; divide by a running RMS and scale by 0.1, clip at one |
| Disagreement | Variance of five independent predictions of the successor's bounded observation coordinates; same RMS normalization; each model trains on an independent bootstrap sample |
| Action-balanced | Adaptive counts of three actions in each observed-size bin; softmax of negative counts plus 10% uniform sampling |

Counts use the observed triangle height and known sensor range, not simulator distances. Learned actors still receive the six corner coordinates. The adaptive action-balanced collector resets its count memory for each collection; it is not a frozen Markov actor and needs no actor-training phase. This distinction and its lower computational cost are reported rather than hidden.

ICM policies freeze after learning. The RND target never receives gradients. Intrinsic rewards are detached during actor optimization. Source data and policy checkpoints are saved so collection and planning can be audited. Additional model capacity, reward scales and geometry differ; a policy ranking is conditional on this implementation.

Related methods: [RND](https://arxiv.org/abs/1810.12894), [ensemble disagreement](https://pathak22.github.io/exploration-by-disagreement/resources/icml19.pdf). These modules implement their ideas for this toy world; they are not upstream software ports. The ICM/actor code retains the original project's attribution in [third-party notices](../../THIRD_PARTY_NOTICES.md).

## Crossed seeds

All eight saved ICM policies are frozen, each collected dataset is held fixed, and initialization IDs 100, 101 and 102 are independently crossed with every dataset. Pair-index and OU-mixture streams depend on dataset seed, not initialization ID. Matching OU-only encoders distinguish generic initialization sensitivity from effects of the collected ICM data. No newly favorable encoder seed replaces a failed run in the main comparison.

## Outcomes

Primary outcome: shared-range affine log-distance $R^2$, evaluated at fixed final update count on an independent calibration set and test grid. Secondary outcomes: broader fixed-OU $R^2$, output variance, Gaussian characteristic-function discrepancy and reversed-direction grid steps. The latter is a descriptive folding diagnostic, not a proof of global injectivity. All run-level scores remain in the reports; long-budget and architecture-prior results are labelled separately.

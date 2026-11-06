# Is the policy-equivalence result significant or useful?

**Statistically significant?** The previous exploratory TOST rejects differences larger than the stated tolerances for recovery and shared-data planning, under its paired-mean assumptions. The new cohort tests this with tolerances fixed in advance. A small p-value is evidence for that specific claim; it is not the probability that the policies are identical or a measure of scientific importance.

**Useful?** As a control, yes. On native data, behavior policies differed greatly. Under broad common exposure, their recovery and shared-data reaching were almost equal. This narrows the explanation: a poor native-policy result need not mean that its action rule intrinsically prevents a useful representation. But matching exposure changes the data regime and can obscure differences that matter in actual exploration. Equal average scores do not establish equal representations, collection efficiency or rare-failure rates.

The paper gives a population/global-optimum result and explicitly leaves finite-sample and training dynamics unresolved; action-conditioned transitions still need to be learned. Our experiment is outside its exact Gaussian-OU setting. [Paper, §§7–8](https://arxiv.org/html/2605.26379v1#S7).

## The substantive hypothesis tested now

**ICM can damage geometry through the observations its behavior collects, even when its transition rule is not intrinsically harmful.**

The new ICM control holds actor, native alignment pairs, initialization and optimizer budget fixed. Both arms independently sample SIGReg inputs. Only the observation pool changes from native exposure to broad common exposure. If this improves recovery, reduces folds and improves physical reaching on fresh actors/seeds, it supports a finite-training exposure mechanism. The additional broad observations are a real intervention with a collection cost; this is not “ICM is as good for free.”

This is stronger evidence than a policy leaderboard or a coverage/error correlation. It remains an environment-specific mechanism result, not a new identifiability theorem. Neither replication nor a low p-value alone establishes novelty relative to the wider literature.

## The next extension worth testing

Separate **representation quality** from **learning the action-conditioned transition**:

1. Freeze exactly the same encoder for all policies.
2. Train predictors on each policy's data, matching data volume and state coverage.
3. Measure coverage of actions *conditional on the starting state*, not just distance-bin totals.
4. Test real goal-reaching on common cases, then intervene on missing state/action transitions.

If planning differences survive a common encoder but disappear after filling specific action-coverage gaps, behavior affects planning through transition learning even when recovery is fixed. If they survive both controls, investigate multistep model error and planner sensitivity. This is not claimed to have been answered by the current cohort.

Linear identifiability is also stronger than necessary for control: a one-to-one nonlinear representation can support planning with the correctly transformed dynamics and goal cost. Conversely, two distant goals sharing an encoding can force goal ambiguity even with excellent one-step prediction. Those are distinct questions from asking which policy attains the highest R².

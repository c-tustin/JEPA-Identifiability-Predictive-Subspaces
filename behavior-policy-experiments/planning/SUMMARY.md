# What we learned about planning

The agent sees a triangle that shrinks with distance. We tested whether what it learns lets it move toward a goal, rather than just score well on a prediction test.

**The main findings:**

- **ICM curiosity was not reliably better than OU.** More runs exposed failures. In the original task, average final goal error was **0.083 for OU versus 0.231 for ICM→OU**, when both predictors used the same training data. Smaller is better.
- **The data matter twice:** first when learning the representation, then when learning how actions change it. Training the ICM predictor on its own collected data increased error to **1.521**. Uneven coverage was part of the problem.
- **A representation can confuse distances or squeeze them too close together.** Both can hurt control even when prediction error looks small. Rebalancing the data fixed the tested ICM folds, but was harmful in our later nonlinear world.

## The newer nonlinear test

We kept one distance coordinate and added nonlinear drift and noise. Persistent actions with ordinary, unbalanced SIGReg performed best on the tested local goals:

| Predictor training data | Closest distance from goal | Reached within 0.1 |
| --- | ---: | ---: |
| Shared data | 0.0346 | 97.4% |
| The policy’s own data | **0.0204** | **100%** |

These are closest-approach scores, not final errors, so do not compare them directly with the original task above. The nonlinear test has three representation seeds; success on these local goals does not establish long-range planning.

**Completed:** policy comparisons, shared/own-data controls, nonlinear planning, and checks of coverage and distance confusion. Saved evaluations reproduce.

**Cloud follow-up:** [Predictive Closure experiments are now available in PR #4](https://github.com/c-tustin/JEPA-Identifiability-Predictive-Subspaces/pull/4). Supplying the action improved prediction, but all 20 learned models still retained distance; predicting uncertainty did not produce a clear feature-selection advantage.

**Next:** give the representation a stricter information budget, and separate what an action reveals about position from what it physically changes.

[Nonlinear results and plots](../nonlinear-control/EXTENSION_SUMMARY.md) · [Full original results](README.md) · [Why prediction error can mislead](../nonlinear-control/CONTROL_INTERPRETATION.md)

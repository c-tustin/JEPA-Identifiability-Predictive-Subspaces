# Beyond reproduction: coverage and nonlinear planning

This follow-up is fixed before inspecting its results. Keep one distance, the same triangle, the same noisy nonlinear/zero-drift simulators, and the original flexible encoder. No distance labels enter encoder or predictor training.

## 1. Match source states and actions

The trajectory experiment confounded dynamics with coverage. Add a simulator-reset control: 16,384 independent source distances sampled uniformly on [1,16], each with an independently uniform approach/stay/retreat action. Hold the action for 50 Euler steps of size 0.02. Each query receives its own noise path. The source states, actions and noise draws are identical across zero/nonlinear drift; successor states differ.

This is a **reset-query intervention**, not a realizable exploration policy. It changes both state and action coverage relative to trajectories; it cannot identify the separate contribution of either change. Its controlled drift comparison holds both fixed. All encoders may use the same shared reference data for the later planning comparison.

Reuse seeds 23–25 and encoder initializations from the trajectory study for paired controls; these are not new validation seeds. Train raw/balanced SIGReg in each world with the same 4,000-update objective and batch streams: **12 additional encoders**. Retain all results, even if balancing hurts. Report distance/log-distance recovery, reversal size and observed-bin coverage. Sampling the source is a simulator privilege; source distance is discarded before fitting.

## 2. Measure physical goal reaching

Evaluate all **24 nonlinear-world encoders**: 18 from trajectories and 6 reset-query controls. The encoder is frozen. Fit the existing observation/action residual predictor on either a common reset-query dataset or the trajectory policy's own dataset. The reset controls use the common dataset only. Predictor training uses 1,000 updates of batch 256, with **two initialization/batch seeds** per encoder. This gives **84 predictors**.

Shared predictor data use fresh random streams, independent of representation training. Normalize every encoder using the same shared observations. Shared held-out reset queries measure prediction error; native data are regenerated exactly from the saved trajectory protocol. Policy names describe the representation's collection process, not the planner's action rule.

Each seed has 32 common starts drawn uniformly from the part of [2,15] where $|0.3\sin d|\le0.1$. Generate each goal by two noiseless simulator decisions using the action opposing the initial autonomous drift. Save that action path as a reachability witness. This tests active control in a locally controllable region and avoids assuming that arbitrary goals are reachable against nonlinear drift. The witness is evaluated, not given to the learned planner.

**Task revision before full evaluation:** an execution-only four-case pilot used unrestricted starts and random goal directions; always-stay reached three of four goals through passive drift. Its code/results are retained outside the research outputs in `/tmp/kempner-nonlinear-control-smoke`. The restricted starts and opposing-drift goals above were fixed before training the 84 full predictors. Pilot scores are not included in the research tables.

Use 12 decisions of receding-horizon search, horizon 8 and 128 fixed candidate action sequences. Include the existing direct-path candidates, including both two-decision witnesses. Compare the learned predictor with a **known-dynamics nominal MPC** using the same candidates and cost, and with always-stay. The known model ignores future noise, so its stochastic score is a reference, not an optimal lower bound. Candidate-bank search is not a global optimal controller.

Score both deterministic execution and execution with fresh state-dependent noise, using identical Brownian draws across controllers. Primary goal-reaching metrics are minimum physical error over the 12 decisions and fraction coming within 0.1. Also report terminal error; some goals cannot be held against autonomous drift. There is no true-distance stopping rule: all rollouts last 12 decisions and the hit time is scored afterward. The learned planner uses only current/goal embeddings and learned transition predictions. Simulation, reachability checks and scoring use physical distance.

Prediction MSE and recovery $R^2$ do not certify control. Compare common versus native data within each encoder, and raw versus balanced SIGReg within each representation seed. The two predictor initializations and 32 goals are nested within only three representation seeds; they do not create 192 independent experimental replicates. No algorithm ranking or nonlinear identification theorem follows from this study alone.

## 3. Mathematical target

The [action-effect lemma](PROOF.md) identifies affine distance under an informative smooth encoder and constant action coefficient in a conditional-drift model. Neither passive alignment nor an unrestricted finite-time residual predictor imposes that structure. This follow-up tests coverage and practical control first. A separate next step must estimate infinitesimal drift from independently noisy interventions, handle discretization bias and justify information preservation before claiming a theorem for learned nonlinear world models.

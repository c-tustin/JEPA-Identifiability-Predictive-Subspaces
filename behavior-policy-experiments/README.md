# Behavior policies: what the agent sees, and what it learns

A behavior policy chooses the next action. The scene remains a triangle with

$$
x=V/d,\qquad z=\log(d/4)/0.7.
$$

The representation sees the triangle, not distance labels. Its fitting loss is

$$
\mathcal L=0.9\,\mathcal L_{\rm alignment}+0.1\,\mathcal L_{\rm SIGReg}.
$$

The policy changes both which distances are sampled and which views are paired. Some ICM datasets lead to folded encoders: different distances produce the same output. The original three-run handover gain did not hold reliably on new seeds.

## Latest work

- [Reliability findings](reliability/INTERPRETATION.md): balanced replay, alternative policies and independent encoder seeds.
- [Rigorous explanation](reliability/MECHANISM.md): what follows mathematically, what the controls establish, and what remains uncertain.
- [Learned OU](learned-ou/README.md): train correlation/spread or an OU-inspired physical controller.
- [Planning summary](planning/SUMMARY.md): actual goal error with shared versus policy-native predictor data.
- [Latest nonlinear extension](nonlinear-control/EXTENSION_SUMMARY.md): matched-coverage controls, physical goal reaching, and the finite-time bias/independent-noise sampling cost behind drift identification. Still one physical coordinate.

## Earlier studies

[Original reproduction](identifiability/README.md) · [ICM comparison](curiosity/README.md) · [Schedules](schedules/README.md) · [Eight-seed curriculum controls](curriculum-controls/README.md) · [Research notes](analysis/RESEARCH_NOTES.md)

Follow the [installation instructions](../README.md#try-the-code), then use the run and reevaluation commands in each guide. Results distinguish historical diagnostics from new validation seeds and retain unsuccessful runs.

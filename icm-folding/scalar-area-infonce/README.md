# Scalar projected-area observations with InfoNCE

Replace the six projected vertex coordinates by **one projected-area observation**. The observation adapter uses the determinant/shoelace formula; neither encoder nor critic receives coordinates. For the fixed triangle, `A(d)=A0/d²`, strictly decreasing on positive distances. Area would also change with triangle size/shape; independently varying shape is not tested and could introduce distance ambiguity.

The actor and ICM remain frozen from the existing coordinate-based sources. This tests a new representation observation model on identical trajectories, not end-to-end ICM trained on scalar observations. All 15 source datasets are reused; they are not a newly collected validation cohort.

## Architecture

- Observation: projected area `A`, shape `[batch, 1]`.
- Encoder: `tanh(4A) → Linear(1,48) → Tanh → Linear(48,48) → Tanh → Linear(48,1)`.
- Critic: `z → Linear(1,48) → Tanh → Linear(48,48) → Tanh → Linear(48,1)`.
- Score: negative squared distance between critic output and training-standardized candidate area, divided by 0.2.
- Loss: own-area/code InfoNCE with 256 objects and 255 negatives; no temporal alignment, EMA, SIGReg or extra reconstruction MSE.

The final linear code is in arbitrary units. The scalar critic output is in standardized area units. This removes the previous six-dimensional scorer's perpendicular-to-data-line directions, but does not certify calibrated reconstruction or prevent encoder folds by construction.

[Fixed protocol](PROTOCOL.md) · [Trainer and checks](run.py) · [Independent report](report.py)

From the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/scalar-area-infonce/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/scalar-area-infonce/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/scalar-area-infonce/report.py
```

**Completed November 19:** all 15 fits and independent reload checks passed, with **0/15 detected final folds** and minimum output standard deviation **3.4685**. Mean held-out next-area MSE is **0.162994**, versus action-only errors of 0.498–1.273 across sources; mean uniform-distance coverage MSE is **0.012023**. Curves compress distance resolution strongly near the closest boundary and at the far end; no detected fold does not imply uniform precision. The encoder/critic shapes differ from the coordinate experiment, so weights are initialized deterministically rather than claimed to be identical. Recorded transition plans are paired. Downstream prediction targets **next scalar area**, not next six-coordinate observation, so MSE values are not directly comparable with earlier results.

The scalar observation is analytically ordered for this fixed geometry; the learned MLP is unconstrained. Grid-based fold checks remain empirical and provide no all-input or training-trajectory guarantee.


[Full per-seed results](output/RESULTS.md) · [Verification](output/verification.json) · [Scalar observation and its representation](output/scalar_observation_and_representation.png) · [All distance representations](output/distance_representations.png) · [Training and separate next-area prediction](output/training_and_prediction.png). Each figure also has a vector SVG.

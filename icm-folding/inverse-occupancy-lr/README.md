# Inverse occupancy with learning-rate controls

November 15, 2026. Paired diagnostic follow-up on historical ICM sources 130, 131, 136, each trained from the original initialization for 4,000 scalar-encoder updates. The frozen actors are unchanged. Reuses the existing study through `../occupancy-study`; no checkpoints are copied.

Weights are inverse endpoint occupancy in 32 equal log-distance bins, normalized to batch mean one. This removes their arbitrary global scale. Compare output-gradient scaling and weighted characteristic-function moments at 0.25 and 4 times the original scheduled Adam rate; existing 1-times runs are the paired controls. A further gradient-only arm multiplies the scheduled rate at each update by `||g_ordinary|| / ||g_weighted||`, computed at the same parameters and batch for the full encoder objective. Only SIGReg receives occupancy weights; alignment remains unchanged.

Norm compensation equalizes the nominal SGD gradient-step norm before Adam preconditioning. It does **not** make Adam updates identical: Adam uses coordinatewise accumulated moments. Inverse learning-rate scaling is not a general compensation for uniform gradient scaling under Adam. No clipping or coefficient selection based on results is used. Rare bins can produce high-variance weights. Training uses known sensor distance to estimate occupancy; this is not an observation-only general method.

Run from the workspace root: `OPENBLAS_NUM_THREADS=1 python3 icm-folding/inverse-occupancy-lr/run.py`, then `python3 icm-folding/inverse-occupancy-lr/report.py`. Final curves, slopes and fold roots are independently recomputed after checkpoint reload; folds are finite-grid sign-reversal checks, not injectivity guarantees. The comparison tests prevention from initialization, not repair of a pre-existing fold. Historical 10-seed results are reported separately from this 3-seed rate follow-up.

## Completed results

All 15 new fits completed and passed checkpoint reload verification. Inverse-gradient scaling folds in 2/3 seeds at 0.25-times LR, 1/3 at the reused original rate, 2/3 at 4-times, and 3/3 with norm compensation. Weighted moments fold in 0/3 at each tested rate; ordinary original-rate controls fold in 2/3. These three diagnostic seeds are a subset of the historical ten-seed study, where inverse-gradient scaling folds in 8/10 and weighted moments in 0/10.

See [results](output/RESULTS.md), [summary](output/summary.json), [gradient and exact-reproduction audit](output/audit.json), [rate-control curves](output/learning_rate_comparison.png), [native histograms and encoders](output/native_occupancy_encoders.png), and [bimodal/trimodal histograms and encoders](output/bimodal_trimodal_encoders.png). The original-rate seed-130 inverse-gradient fit was rerun separately and reproduced its original checkpoint bit-for-bit.

# Predictive Closure: paper reproduction

This branch reproduces **Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection**, by Mosh Levy, Aly Lidayan and Yoav Goldberg.

The experiments use CIFAR-10 image collages and a small JEPA representation. They test which features the representation keeps, and how reconstruction or predictor side information changes that choice.

- [Setup and commands](predictive-closure-reproduction/README.md)
- [Current status](predictive-closure-reproduction/RUN_STATUS.md)
- [Checkpoint and restart instructions](predictive-closure-reproduction/RESTART.md)

**Status:** data and execution checks passed; local training is paused. No full model or published accuracy result has been reproduced yet. The planned main suite contains 77 encoders. The CPU trial used a documented runtime variant; the authors' CUDA profile remains available.

This branch focuses on the paper reproduction. The practice folder is intentionally excluded; the repository license and original research proposal are retained.

Sources: [paper](https://arxiv.org/abs/2610.27364v1), [authors' code](https://github.com/Mosh0110/predictive-closure).

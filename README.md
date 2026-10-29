# Predictive Closure: extensions

This branch studies limits and extensions of **Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection**. These are standalone experiments, rather than a reproduction of the paper's CIFAR results.

The central question is whether good prediction of a representation's **mean** captures all useful information. In a one-distance triangle world, we compare mean and distributional prediction, examine a folded representation that confuses two distances, and test a nonlinear drift/planning extension.

- [Overview and commands](predictive-closure-extensions/README.md)
- [Completed results](predictive-closure-extensions/RESULTS.md)
- [Assumptions and mathematics](predictive-closure-extensions/MATH.md)
- [What was run](predictive-closure-extensions/RUN_STATUS.md)

**Main findings:** an explicit folded coordinate predicts its own future better while losing distance information; all 20 learned neural encoders still retain distance. Multiple sampling intervals did not improve drift estimation in these pilots. These results do not refute the paper's conditional-Gaussian theorem.

All required code and saved outputs are inside predictive-closure-extensions/. No other experimental branch is required. Files already present on main are retained unchanged, apart from this branch overview.

Sources: [paper](https://arxiv.org/abs/2610.27364v1), [authors' code](https://github.com/Mosh0110/predictive-closure).

# Behavior policy experiments

This folder contains the inverse-distance reproduction and a new test of what happens when a learned curiosity policy supplies the training transitions.

- [Original Klindt-paper reproduction](identifiability/README.md): Gaussian log-distance and stationary OU positive pairs; original figure, metrics and checkpoints are retained.
- [Curiosity-policy experiment](curiosity/README.md): compares OU, random-action, and learned ICM trajectories with the same triangle observation and SIGReg encoder.
- [Curiosity results](curiosity/output/RESULTS.md): generated from the multi-seed run.

Run commands from the repository root after installing `requirements.txt` with Python 3.12:

```sh
python behavior-policy-experiments/identifiability/reproduce.py
python behavior-policy-experiments/curiosity/experiment.py
```

The second command is the behavior-policy comparison. Its OU condition uses a fixed offline pair budget matching the behavior conditions. It is a new control run, separate from the historical online-OU reproduction.

The experiment assesses empirical linear recovery. A bounded, policy-driven distribution violates the exact Gaussian OU assumptions, so neither success nor failure proves or disproves the original theorem outside its stated setting.

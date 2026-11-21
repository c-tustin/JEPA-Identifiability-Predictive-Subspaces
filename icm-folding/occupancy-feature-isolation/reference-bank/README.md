# Frozen prediction-bank aliases

These twelve relative file links expose the already trained dose-study checkpoints selected before generation of sources 9100–9109. They are not competing checkpoint copies. BANK.json records their original file hashes and folded/ordered status. Portable publication includes these twelve binary reference artifacts at these paths while omitting the competing derived-cache paths under output/dose/runs. This lets the frozen predictions be checked without retraining the entire dose sweep.

The source generator used before source validation is preserved exactly in ../provenance/fresh_sources_preregistered.py. The current generator also supports resolving this bank outside the original filesystem and verifying archived predictions without replacing them. The historical prediction files and PREDICTIONS_LOCK.json remain unchanged.

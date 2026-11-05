# Does reflection alignment actually cause the learned fold?

Post hoc experiment, specified after the crossed results and mirror-code diagnostic, before these reweighting fits. The mirror-code comparison alone does not establish the cause of the network's fold.

Use all five fixed ICM datasets and all five initializations from the crossed experiment, with its same batch indices, optimizer, 4,000 updates and common evaluation seed. Add two arms per combination (50 fits):

1. Set the alignment contribution of near-wall reflected approach pairs to zero. Identify them from the approach action and source apparent height > 1/1.25. Do not renormalize the remaining alignment weights.
2. Set the alignment contribution of an equally large fixed random subset of stored pairs to zero (seed 2,400,000 + data seed). This controls for the amount of alignment removed.

**SIGReg receives the original endpoints of every sampled pair unchanged in both arms.** Only the alignment pair weights differ. No policy training, added observations or changes to the physical environment. A loss-equivalent all-ones mask must first reproduce the existing ICM data-8/init-8 weights exactly.

Compare all three 5×5 matrices (ordinary, reflection removal, random removal), with R² and fold counts. Descriptive policy effect is reflection removal minus random removal. Average across initialization within dataset before reporting any sign test: there are five dataset levels, not 25 independent trials. Report a two-sided exact sign test over five data means and retain the resulting low power. No preselection of successful initializations or final winners.

If selective removal improves recovery, it supports a contribution of reflection alignment at this training budget. If it fails to improve recovery, retain that negative result: a lower-loss mirror comparator is not a causal explanation of the trained network. This does not test every possible pairing effect or remove SIGReg's sampling pressure.

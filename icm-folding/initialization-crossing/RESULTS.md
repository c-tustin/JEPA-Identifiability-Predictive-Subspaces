# Dataset × initialization crossing: completed results

All 100 planned 4,000-update fits completed. All ten historical diagonal checkpoints reproduce with zero maximum parameter difference. Final geometry was evaluated after checkpoint reload on 20,001 float64 log-distance grid points; initial-weight and sampling-plan pairing checks passed.

55/100 fits have a detected final slope reversal; 51/100 meet the prespecified material-fold threshold (backward variation fraction > 0.001). These are crossed cells from ten reused datasets, not 100 independent policy sources.

| Dataset seed | Any final fold / 10 initializations | Material final fold / 10 |
| --- | ---: | ---: |
| 130 | 0/10 | 0/10 |
| 131 | 10/10 | 10/10 |
| 132 | 10/10 | 10/10 |
| 133 | 10/10 | 10/10 |
| 134 | 2/10 | 1/10 |
| 135 | 2/10 | 0/10 |
| 136 | 10/10 | 10/10 |
| 137 | 1/10 | 0/10 |
| 138 | 0/10 | 0/10 |
| 139 | 10/10 | 10/10 |

All five originally folded datasets (131,132,133,136,139) fold under every tested initialization. Datasets 130 and 138 remain ordered on the evaluation grid under every initialization. Datasets 134 and 135 fold for initializations 130 and 136; dataset 137 folds only for initialization 130. Thus dataset/sampling condition strongly predicts the final outcome in this crossing, while initialization affects borderline conditions. The design bundles each dataset with its historical minibatch plans and does not distinguish the learned policy, rollout randomness, occupancy shape, and minibatch noise. It does not establish occupancy as the unique cause.

Final folds do not alone distinguish folds retained from initialization from those created during training; the saved histories permit that separate analysis. No new actor training, recollection, downstream planning, fresh-source validation, or analytic injectivity certificate was performed.

[Machine-readable cell results](output/summary-n4000.json) · [Fixed protocol](PROTOCOL.md)

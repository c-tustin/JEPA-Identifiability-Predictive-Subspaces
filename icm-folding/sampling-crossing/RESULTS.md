# Independent data and minibatch sampling: completed results

All 500 fits completed and passed independent checkpoint reload, source/initialization/plan checks, and 20,001-point float64 derivative evaluation. Two original full-run replays and the short model/Adam interruption-resume check passed exactly before launch. Final detected folds: **270/500**, including **256/500 material folds**.

Every dataset/init pair was trained with five fresh, independently seeded alignment/regularizer plan pairs. At each plan label, all datasets and initializations share exact index tensors. Thus dataset identity no longer determines a unique training plan.

| Dataset | Any final fold / 50 | Material final fold / 50 |
| --- | ---: | ---: |
| 130 | 0/50 | 0/50 |
| 131 | 50/50 | 50/50 |
| 132 | 50/50 | 50/50 |
| 133 | 50/50 | 50/50 |
| 134 | 10/50 | 6/50 |
| 135 | 10/50 | 0/50 |
| 136 | 50/50 | 50/50 |
| 137 | 0/50 | 0/50 |
| 138 | 0/50 | 0/50 |
| 139 | 50/50 | 50/50 |

Secondary range diagnostic: 250/500 fits have a final turn inside their own saved dataset's observed distance range. All are from datasets 131/132/133/136/139; the 20 extra turns from 134/135 are outside their respective observed ranges. The primary 270/500 count above still uses the prespecified full [1,16] evaluation interval.

| Batch-plan label | Any final fold / 100 | Material final fold / 100 |
| --- | ---: | ---: |
| 0 | 54/100 | 52/100 |
| 1 | 54/100 | 51/100 |
| 2 | 54/100 | 51/100 |
| 3 | 54/100 | 51/100 |
| 4 | 54/100 | 51/100 |

Across the five plans, 0/100 fixed dataset/init pairs change binary fold status; 1/100 change material-fold status. Across all plan pairs at fixed data/init, disagreement occurs in 0/1000 binary-fold comparisons and 4/1000 material-fold comparisons. Across dataset pairs at fixed init/plan, disagreement occurs in 1210/2250 and 1244/2250 comparisons, respectively. These are descriptive pairwise disagreement rates, not independent statistical tests.

## Descriptive balanced-factor decomposition

Fraction of centered sum of squares for binary cell outcomes, including interactions. No residual is treated as independent noise; the three-way term is retained. These fractions describe this particular finite crossed matrix, not population causal shares.

| Term | Any fold | Material fold |
| --- | ---: | ---: |
| dataset | 87.118% | 95.774% |
| initialization | 2.576% | 0.359% |
| sampling | 0.000% | 0.006% |
| dataset × initialization | 10.306% | 3.227% |
| dataset × sampling | 0.000% | 0.058% |
| initialization × sampling | 0.000% | 0.058% |
| dataset × initialization × sampling | 0.000% | 0.519% |

The separate historical matrix had 55/100 final turns. Its differences from the fresh matrix are: dataset 137, initialization 130: historical folded=True, material=False; fresh folded=[False, False, False, False, False]

The old extra tiny turn demonstrates that sampling can affect marginal cases. The invariance across five fresh streams does not establish invariance across every possible stream. Historical results are not pooled into the new 500-cell matrix.

These fits isolate encoder-training minibatch randomness from the fixed saved datasets conditional on the tested initialization bank and optimizer. They do not distinguish actor-training randomness from collection randomness, establish that occupancy is the unique dataset mechanism, or assess fresh-policy generalization. Shared episode/time indices do not match the physical observations across datasets. Five plans are a limited sampling replication. Fold detection remains a finite-grid diagnostic; no continuous-domain injectivity guarantee or downstream planning benefit is claimed.

[Cell results and decompositions](output/summary.json) · [Independent audit](output/verification.json) · [Preflight](output/preflight.json) · [Fold matrices](output/sampling_crossing.png) · [Protocol](PROTOCOL.md)

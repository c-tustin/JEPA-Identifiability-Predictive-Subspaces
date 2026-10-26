# Which loss term is sensitive to replay?

Exploratory mechanism test after the repair study; all four cells fixed before inspecting new seeds 18–22. Seeds 3 and 7 are historical diagnostics. Every update uses separate 512-pair alignment and regularizer batches in all four conditions. This doubles observation evaluations relative to the shared-batch experiment, equally in every cell.

| Seed group | Alignment data | SIGReg data | Runs | Shared-range $R^2$ | Broader OU $R^2$ | Reversed grid steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| new | raw | raw | 5 | 0.910027 ± 0.198216 | 0.898935 ± 0.215192 | 0.063734 ± 0.142514 |
| new | raw | balanced | 5 | 0.994026 ± 0.000763 | 0.978745 ± 0.002669 | 0.000000 ± 0.000000 |
| new | balanced | raw | 5 | 0.911878 ± 0.193375 | 0.900589 ± 0.211219 | 0.063148 ± 0.141202 |
| new | balanced | balanced | 5 | 0.994081 ± 0.000766 | 0.978859 ± 0.002678 | 0.000000 ± 0.000000 |
| diagnostic | raw | raw | 2 | 0.332563 ± 0.466909 | 0.310355 ± 0.403620 | 0.341642 ± 0.075342 |
| diagnostic | raw | balanced | 2 | 0.993981 ± 0.000912 | 0.979099 ± 0.004878 | 0.000000 ± 0.000000 |
| diagnostic | balanced | raw | 2 | 0.336848 ± 0.474274 | 0.313233 ± 0.410683 | 0.340176 ± 0.077415 |
| diagnostic | balanced | balanced | 2 | 0.994035 ± 0.000925 | 0.979214 ± 0.004906 | 0.000000 ± 0.000000 |

Both losses finish with the same OU sources. Changing a sampling measure changes the finite objective and its optimization path; this does not prove a particular local minimum or establish Gaussianity. Complete per-seed metrics and checkpoints are retained.

# Gradient-strength control

Ten development sources; original batch-local collision gradient raised to coefficient 100 without global mining. Same encoder initialization and base batches. This isolates auxiliary strength partly; global mining still has more observation exposure and compute.

| Method | Any folds | Material folds | Held-out MSE | Coverage-stress MSE |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 5/10 | 5/10 | 0.210549 | 1.316871 |
| Local 1 | 5/10 | 5/10 | 0.207936 | 1.287567 |
| Local 100 | 4/10 | 3/10 | 0.094987 | 0.917025 |
| Global 100 | 1/10 | 1/10 | 0.019100 | 0.103342 |

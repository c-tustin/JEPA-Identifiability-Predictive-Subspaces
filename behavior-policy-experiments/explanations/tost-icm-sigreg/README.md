# TOST, ICM folds and SIGReg graphs

[Download the offline ZIP](shareable-graphs.zip), extract it and open
`shareable-graphs/index.html` in any browser. No server or network is needed.

The bundle contains four figures:

- Learned representation curves at different encoder-training times.
- First detected fold and inward migration, showing only the migration panel.
- TOST comparisons, with a zoom revealing the recovery confidence intervals.
- SIGReg exposure and outcomes, with a zoom revealing variation across seeds.

The training-location heatmaps are deliberately omitted. The broad-exposure
R² values are distinct across all 60 seeds (range 0.992733–0.999467); they
overlap on the full-scale plot. The TOST recovery intervals are also present
but require the added zoom to be visible.

Rebuild from saved experiment results with Python, NumPy and Matplotlib:

```sh
python explanations/tost-icm-sigreg/make_graphs.py
```

Run this from the enclosing experiment folder, or invoke the script by its
full path. It reads saved JSON results without training models or changing
scientific source files, checkpoints or locks. `graph_checks.json` records
basic checks of the plotted cohort.

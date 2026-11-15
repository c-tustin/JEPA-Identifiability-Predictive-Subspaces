# No-fold proof and empirical audit

[LaTeX source](no-fold.tex) · [Compiled note](output/pdf/no-fold.pdf) · [Fresh verification](output/verification.json) · [Training stress figure](output/training-stress.png)

The note proves that projecting spline knot values onto a minimum-slope constraint after every optimizer proposal preserves strict ordering over the whole interval. It also proves the positive-slope construction, derives the characteristic-function gradient under occupancy weighting, gives explicit soft-penalty and finite-probe counterexamples, and states a sufficient certificate for an unchanged smooth MLP. The latter is a mathematical option, not an implemented method here.

The implemented guarantee assumes the simulator's known observed coordinate `s=1/d` and changes the encoder class to a piecewise-linear spline. It does not guarantee linear log-distance codes, ICM coverage, predictor accuracy, or ordering of arbitrary images.

`verify.py` reuses the existing [hard-order study](../sources/explanations/occupancy-folding/hypothesis-guarantee/README.md) through its canonical source location, without duplicating checkpoints. It independently loads the study's report module by path to avoid a `report.py` name collision in historical imports. It recomputes final slopes, full curves, derivatives and objectives for 100 constrained and 50 free models using the existing audit; checks 100,000 additional ordered pairs; verifies 1,000 random PAVA KKT cases; and runs a new 500-update adversarial Adam experiment. The historical training-time checks are verified from retained minima and histories, not by replaying all historical training updates. Counts are descriptive, paired by ten historical sources.

Run from the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/icm-fold-mpl python3 icm-folding/no-fold-proof/verify.py
cd icm-folding/no-fold-proof
pdflatex -halt-on-error -output-directory=output/pdf no-fold.tex
pdflatex -halt-on-error -output-directory=output/pdf no-fold.tex
```

The JSON includes checkpoint hashes and fresh audit results. The table and stress figure are generated from the verifier. Numerical slope tolerances are explicit; the mathematical theorem uses real arithmetic. The PDF should be rendered and visually checked after changes.

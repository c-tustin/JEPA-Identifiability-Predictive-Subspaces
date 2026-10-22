# How the SIGReg code works

The implementation is now in [`sigreg.py`](sigreg.py). `reproduce.py` still exports `SIGReg1D`, so the existing experiments can import it as before.

The goal is to give the network’s output numbers a standard bell-shaped distribution, with mean zero and variance one. Without a spread requirement, the network could give every view the same number and make the pair-matching loss zero.

## The five steps in the code

1. Choose 17 frequencies between 0 and 3.
2. At each frequency, compute the cosine and sine averages we expect from a standard Gaussian. The cosine average is `exp(-t**2 / 2)`; the sine average is zero.
3. Compute those same averages from the network’s output numbers.
4. Square the differences and add them together at each frequency.
5. Add the errors using the authors’ integration weights, then multiply by the number of output samples.

The sine/cosine averages together are called a **characteristic function**. They describe a distribution in a different way from a histogram.

The integration weights are larger at low frequencies. The endpoint frequencies get half the usual trapezoidal weight. We multiply the positive-frequency integral by two to account for negative frequencies, which have the same squared error. These choices match the [authors’ implementation](https://github.com/klindtlab/lejepa-identifiability/blob/main/experiments/lejepa_id/losses.py).

The final sample-count multiplier also matches the authors’ code. Removing it would change the relative strength of SIGReg and the pair-matching loss. Each view counts separately: a batch with 512 pairs has 1,024 scalar outputs.

## Why there are no random projections

Full SIGReg projects vectors onto random directions and tests the resulting scalar values. Our output has only one dimension. Its normalized directions are +1 and −1, and both give the same squared error. Repeating random projections would therefore repeat the same calculation.

This class expects `[samples]`, `[samples, 1]`, or `[views, samples, 1]`. Several-feature vectors need the full projection step; this scalar class rejects them rather than silently flattening different features into one distribution.

## What changed in the refactor?

The calculation is split into named steps, with comments explaining the weighting and scaling. The buffer names `frequency`, `target`, and `weight` are preserved for compatibility. Existing experiment imports and constructor arguments remain valid.

Tests compare both the loss values and gradients with the exact previous expression in float32 and float64, for all supported input layouts. The results are bit-for-bit equal in those tests. The independent comparison with the upstream 256-projection formula is also retained.

No trained model or recorded result was changed. Empty input and multi-feature input now give clear errors.

## Limits

Matching a finite set of frequencies on a finite batch does not certify an exactly Gaussian distribution. A small SIGReg loss can coexist with errors in the tails or in the output variance. That is why our reports also show variance, recovery on new data, and behavior outside frequently visited distances.

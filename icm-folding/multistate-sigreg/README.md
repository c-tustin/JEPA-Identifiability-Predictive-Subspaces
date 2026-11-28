# SIGReg with a 2-d state (running; results pending)

**Status (Nov 27): training and evaluation in progress. No results are claimed yet.** Per-cell records in
`output/cells/` are partial until this README is updated.

[Protocol (hash-locked before any fit)](PROTOCOL.md) · [World, trainer and evaluation](run.py)

## Question

In the 1-d-state world, SIGReg never aliased states once the embedding had two or more dimensions
([multidim-sigreg](../multidim-sigreg/README.md)). Real JEPA states have several dimensions. This study asks whether
skewed occupancy makes SIGReg alias distant states when the state is 2-d and the embedding dimension is below, equal to,
or above it.

## Design

- **World.** The original triangle pinhole renderer with depth d ∈ [1, 16] and lateral offset b ∈ [−3, 3]. The
  observation has 6 coordinates, and there are 5 actions of 0.25.
- **Occupancy.** Scripted random-walk policies. Realized mass below d = 2: balanced about 6%, skew-d about 87% (like
  ICM), and skew-db about 85% plus about 81% at b < −2.
- **Fits.** D ∈ {1, 2, 3, 4, 8}, ordinary and occupancy-weighted SIGReg (64 random projections), 5 seeds: 150 fits.
- **Measures.** Aliasing (a state more than four moves away embeds closer than the nearest one-move neighbour), and the
  downstream harness generalized to a 2-d state: next-observation prediction, state readout, and MPC planning.

**Smoke test (one fit, not a result):** skew-d, ordinary, D = 2, seed 0 had 28% of native mass aliased, with minimum far
ratio 0.013.

## Reproduce

From the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/multistate-sigreg/run.py --workers 10
```

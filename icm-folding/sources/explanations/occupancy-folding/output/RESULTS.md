# Measured occupancy interventions

All runs: gain 4, 4,000 updates, pure ICM alignment. R² uses a uniform-log-distance evaluation grid. Controlled distributions change only SIGReg sampling; seeds 130, 131, 136 are diagnostic.

| Occupancy | Method | Final folded runs | Mean log-distance R² |
| --- | --- | ---: | ---: |
| native | Ordinary SIGReg | 5/10 | 0.6115 |
| native | Inverse gradient only | 8/10 | 0.6091 |
| native | Inverse-weighted moments | 0/10 | 0.9820 |
| near2 | Ordinary SIGReg | 0/3 | 0.9486 |
| near2 | Inverse gradient only | 1/3 | 0.7235 |
| near2 | Inverse-weighted moments | 0/3 | 0.9909 |
| bimodal | Ordinary SIGReg | 0/3 | 0.9897 |
| bimodal | Inverse gradient only | 3/3 | 0.6042 |
| bimodal | Inverse-weighted moments | 0/3 | 0.9915 |
| trimodal | Ordinary SIGReg | 0/3 | 0.9879 |
| trimodal | Inverse gradient only | 1/3 | 0.9420 |
| trimodal | Inverse-weighted moments | 0/3 | 0.9925 |
| uniform | Ordinary SIGReg | 0/3 | 0.9991 |

## Ordinary-SIGReg trajectory details

All turning distances are reported; a blank means no turn in [1,16]. Transient turns can disappear. Snapshots are 25 updates apart.

| Occupancy | Seed | At 250 | At 500 | At 1,000 | At 4,000 |
| --- | ---: | --- | --- | --- | --- |
| native | 130 | — | — | 8.803 | — |
| native | 131 | 6.063 | 3.444 | 3.073 | 2.368 |
| native | 136 | — | 5.978 | 3.805 | 3.137 |
| near2 | 130 | — | — | — | — |
| near2 | 131 | — | — | — | — |
| near2 | 136 | — | — | — | — |
| bimodal | 130 | — | — | — | — |
| bimodal | 131 | — | — | — | — |
| bimodal | 136 | — | — | — | — |
| trimodal | 130 | — | — | — | — |
| trimodal | 131 | — | — | — | — |
| trimodal | 136 | — | — | — | — |
| uniform | 130 | — | — | — | — |
| uniform | 131 | — | — | — | — |
| uniform | 136 | — | — | — | — |

[Explanation and limitations](../README.md) · [Graph gallery](index.html) · [Machine-readable metrics](summary.json)

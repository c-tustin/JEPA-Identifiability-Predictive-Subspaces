# Stage 1: completed component swaps

All 40 full hybrid fits passed independent dense checkpoint reload, component normalization, source/initialization/plan identity, and fixed-component checks. There are 21 detected final turns and 20 material folds.

| Pair: original nonfolded / folded | Near mass from first / second | Material folds with first mass | Material folds with second mass |
| --- | --- | ---: | ---: |
| 130 / 131 | 92.981% / 84.637% | 0/4 | 4/4 |
| 134 / 132 | 95.273% / 86.874% | 0/4 | 4/4 |
| 135 / 133 | 95.386% / 86.710% | 0/4 | 4/4 |
| 137 / 136 | 93.500% / 68.582% | 0/4 | 4/4 |
| 138 / 139 | 93.488% / 87.326% | 0/4 | 4/4 |

Every material-fold status follows which source supplies the near/far mixture mass. All 20 paired mass-only changes introduce a material fold; all 20 near-shape-only and all 20 far-shape-only changes leave material status unchanged in this bank. Shapes still affect turning location and severity.

The any-turn exception is retained: source-137 near shape and 93.500% near mass, with source-136 far shape, develops a tiny turn at distance 12.4203 (backward variation 0.0007723), within its hybrid stored range. Thus near mass alone does not explain every possible small reversal.

These interventions isolate a mass effect for the original substantial-fold split across the tested conditional shapes. They do not yet identify a universal threshold or explain the nonmonotone dose response already seen in older concentration studies. Distance 2 was a prior analysis split. Stage 2 varies the mass widely and tests raw/own-code InfoNCE and a variance control under the same distributions; those outcomes and mechanism tests remain pending.

[Matrix](hybrid_components.png) · [All measurements and contrasts](summary.json) · [Audit](verification.json) · [Protocol](../PROTOCOL.md)

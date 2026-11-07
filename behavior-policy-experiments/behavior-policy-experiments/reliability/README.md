# Making the handover more reliable

Same triangle, same single distance coordinate. This study tests the proposed repairs and exploration alternatives. **Seeds 3 and 7 are diagnostic failures; seeds 8–12 are new validation seeds.** No method is selected using distance recovery on those new seeds.

The renderer and representation loss are

$$
x(d)=\frac{V}{d}, \qquad z=\frac{\log(d/4)}{0.7},
$$

$$
\mathcal L=0.9\,\mathbb E[(h(x)-h(x'))^2]+0.1\,\operatorname{SIGReg}(h).
$$

## Fixed comparisons

| Suite | Changes tested | Runs |
| --- | --- | ---: |
| Repairs | OU, random→OU, ICM→OU; 20% OU from the start; earlier handover (10%–50%); ICM with 20% random actions; ICM + coverage reward; balanced replay; monotone encoder; 8,000-update OU/random/ICM controls | 12 conditions × 7 seeds |
| Other policies | RND, five-model disagreement, count-based coverage, action-balanced collection; each followed by the original OU handover | 4 × 5 new seeds |
| Crossed seeds | Freeze each of the eight earlier ICM policies/datasets; cross with encoder initializations 100–102 and matched OU controls | 2 × 8 × 3 |

Ordinary training has 4,000 updates, batch size 512 and the original learning-rate schedule. Long runs have 8,000 updates and matched controls. Learned policies retain 600 updates and 192,000 interactions. Defaults and reward details are in [PROTOCOL.md](PROTOCOL.md).

For the action-floor policy,

$$
\pi_\epsilon(a\mid x)=(1-\epsilon)\pi_{\mathrm{ICM}}(a\mid x)+\frac{\epsilon}{3},
\qquad \epsilon=0.2.
$$

Coverage and the monotone spline deliberately use apparent-size ordering. The spline operates on raw size, not its logarithm. These are geometry-aware interventions, not evidence of unconstrained identifiability. Policy learning never uses true distance labels. Fitting receives only observation pairs.

## Run

```sh
python -m unittest discover -s behavior-policy-experiments/reliability -p 'test_*.py' -v
python behavior-policy-experiments/reliability/run_study.py --suite repairs
python behavior-policy-experiments/reliability/run_study.py --suite policies
python behavior-policy-experiments/reliability/run_study.py --suite crossed
```

Use `--resume` for completed runs, `--evaluate-only` to check saved scores, or `--quick --output /tmp/reliability-smoke` for an execution check. Quick scores are not trained results. Each suite has its own output directory and frozen source datasets.

[Repairs](output-repairs/RESULTS.md) · [Other policies](output-policies/RESULTS.md) · [Crossed seeds](output-crossed/RESULTS.md) · [Concise interpretation](INTERPRETATION.md)

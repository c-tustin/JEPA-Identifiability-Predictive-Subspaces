# Is alignment or SIGReg responsible for the replay effect?

**Post-repair mechanism test.** The replay result motivated this comparison; it is not an untouched primary hypothesis from the earlier protocol. All four cells are fixed before looking at new seeds 18–22. Seeds 3 and 7 remain diagnostic.

| | SIGReg: ordinary data | SIGReg: balanced replay |
| --- | --- | --- |
| Alignment: ordinary data | Control | Change only the regularizer's sampling |
| Alignment: balanced replay | Change only alignment sampling | Change both |

Every update uses separate batches of 512 pairs for the two terms, with matched seeds, model initialization, learning rates, OU masks and final OU sources. This equally increases compute across all four cells; the control is retrained rather than substituted from the shared-batch study. The stored observations and transitions remain the same; replay only changes repetitions. Empty bins are not filled.

For pair weights $w_i$ and encoded endpoint measure $\nu_{w,h}$, the population analogue is

$$
\mathcal L_{u,v}(h)
=(1-\lambda)\sum_i u_i(h(x_i)-h(x_i'))^2
+\lambda R(\nu_{v,h}),
$$

$$
\nu_{v,h}=\frac12\sum_i v_i\big(\delta_{h(x_i)}+\delta_{h(x_i')}\big).
$$

Thus changing replay affects the Gaussian target's sampled distribution as well as temporal alignment. Finite-batch SIGReg adds sampling error; this expression does not assert exact Gaussian constraints. Reweighting by source state also changes the symmetrized pair operator. This is not automatically an occupancy-only intervention.

```sh
python behavior-policy-experiments/reliability/ablate_losses.py
python behavior-policy-experiments/reliability/ablate_losses.py --evaluate-only
```

[All results](output-loss-ablation/RESULTS.md). Effects on one term would narrow the explanation; they would not establish a universal failure mechanism or a local-minimum theorem.

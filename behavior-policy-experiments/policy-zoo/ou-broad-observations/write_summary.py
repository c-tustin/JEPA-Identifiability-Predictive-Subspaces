"""Concise presentation of the verified fixed OU comparison."""
import json
import run_ou as run


def main():
    run.check_lock()
    verified=json.loads((run.OUT/'VERIFICATION.json').read_text())
    assert verified['complete'] and verified['errors']==0
    result=json.loads((run.OUT/'INFERENCE.json').read_text())
    means=result['means']
    lines=['# Broad observations help OU’s affine recovery, with an important scale trade-off','',
           '**Completed:** 60 paired seeds, 180 OU encoders and 180 shared-data planning evaluations. The OU alignment pairs are identical across conditions. [Fixed protocol](../PROTOCOL.md).','',
           '| SIGReg data | R² over distances 1–16 ↑ | R² on held-out Gaussian states ↑ | Raw sign-only error ↓ | Physical goal error ↓ |',
           '| --- | ---: | ---: | ---: | ---: |']
    labels={'original':'Original OU: same pair batch','ou_observations':'Independent OU observations','broad_observations':'Independent broad observations'}
    for arm in run.ARMS:
        m=means[arm]
        lines.append(f"| {labels[arm]} | {m['affine_log_r2']:.5f} | {m['gaussian_r2']:.5f} | {m['orthogonal_mse']:.5f} | {m['goal_error']:.5f} |")
    lines+=['','All three conditions have **0/60 grid-detected folds**. Broad observations use the same log-uniform 1–16 pool as the ICM experiment. OU alignment is not clipped; the Gaussian evaluation includes its tails outside that interval.','',
            '## What changes when only the observation pool changes?','',
            'Compare the last two rows: both sample SIGReg observations independently and use the same gradient implementation.','',
            '| Outcome | Mean benefit of broad pool | Holm-adjusted p |',
            '| --- | ---: | ---: |']
    for r in result['comparisons']:
        if r['contrast']=='broad_vs_ou_observations' and r['outcome']!='folded':
            labels2={'affine_log_r2':'Bounded-grid R² gain','gaussian_r2':'Gaussian R² gain','orthogonal_mse':'Raw sign-only error reduction','goal_error':'Goal-error reduction'}
            lines.append(f"| {labels2[r['outcome']]} | {r['mean_benefit']:+.6f} | {r['holm_p']:.3g} |")
    lines+=['','The isolated bounded-grid gain is about **0.00066 R²**, smaller than the 0.001 practical tolerance used in the earlier policy-equivalence study. Much of the gain over original OU already appears with independent sampling of OU observations. No significant reaching improvement is detected; nonsignificance is not proof of equivalence. All 16 planned comparisons share one Holm correction.','',
            '## Why does R² improve while sign-only recovery worsens?','',
            'R² permits fitting a scale and offset. The stricter error compares the raw network output with the true standardized latent, allowing only a global sign flip—the one-dimensional rotation. Broad-pool outputs become more compressed: on Gaussian test states, their mean standard deviation is '+f"**{means['broad_observations']['raw_output_std']:.3f}**, versus **{means['ou_observations']['raw_output_std']:.3f}**"+' with independent OU observations; the target is 1. This worsens raw recovery despite better linearity. It is a scale error, not newly observed folding or evidence of lost information.','',
            'The planner centers and scales embeddings before predictor training, so a global scale error need not hurt this planning evaluation. The broad pool also changes the marginal on which SIGReg enforces its output distribution. Its formula and weight are unchanged, but the exact original population-theorem setup is not preserved.','',
            '## Does ICM plus broad observations beat regular OU?','',
            'On the same 60 seeds, the saved ICM/native-alignment+broad-SIGReg condition has bounded-grid recovery about **0.99755**, higher than original OU’s **0.99417**. But OU supplied the broad pool also reaches **0.99761**. The ICM condition therefore shows no special recovery advantage over broad-pool OU.','',
            'ICM+broad goal error is about **0.07598**, versus original OU **0.07487** and OU+broad **0.07507**. Neither planned ICM planning comparison is significant after correction. These compare fixed recipes with different pair laws and step scales; they do not establish a general ranking of autonomous policies.','',
            '![Paired OU results](ou_comparison.png)','',
            '[All tests/effect intervals](RESULTS.md) · [Representation curves](FIGURES.md) · [Full verification](VERIFICATION.json) · [Published code lock](LOCK.json)','']
    (run.OUT/'SUMMARY.md').write_text('\n'.join(lines))
    print('Verified OU summary written.')


if __name__=='__main__':main()

"""Fixed paired OU exposure comparisons; refuses incomplete cohorts."""
import json
import numpy as np
from scipy import stats
import run_ou as run
import matched_equivalence as helpers

OUTCOMES = ('affine_log_r2','gaussian_r2','orthogonal_mse','goal_error','folded')


def value(row, key):
    if key in ('affine_log_r2','folded'):
        return float(row['metrics'][key])
    if key == 'goal_error':
        planning = row['planning']
        return planning['shared']['goal_error'] if 'shared' in planning else planning['goal_error']
    return row['gaussian'][key]


def compare(treatment, reference, key):
    d = np.array([value(a,key)-value(b,key) for a,b in zip(treatment,reference)])
    if key not in ('affine_log_r2','gaussian_r2'):
        d *= -1
    wins, losses = int(np.sum(d>1e-10)), int(np.sum(d < -1e-10))
    p = float(stats.binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
    rng = np.random.default_rng(20261107)
    boot = d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
    return {'outcome':key,'mean_benefit':float(d.mean()),'ci95_unadjusted':np.quantile(boot,[.025,.975]).tolist(),
            'wins':wins,'losses':losses,'ties':len(d)-wins-losses,'p':p,'seed_benefits':d.tolist()}


def collect():
    run.check_lock()
    groups = {arm:[] for arm in run.ARMS}
    missing = []
    for arm in run.ARMS:
        for seed in run.SEEDS:
            p=run.OUT/f'{arm}_seed{seed}.json'
            if not p.exists():
                missing.append(p.name)
                continue
            row=json.loads(p.read_text())
            assert row['signature']==run.signature() and row['seed']==seed and row['arm']==arm
            groups[arm].append(row)
    return groups,missing


def main():
    groups,missing=collect()
    if missing:
        raise SystemExit(f'Incomplete OU cohort: {len(missing)} missing; no partial inference.')
    comparisons=[]
    for reference in ('ou_observations','original'):
        for key in OUTCOMES:
            comparisons.append({'contrast':'broad_vs_'+reference,**compare(groups['broad_observations'],groups[reference],key)})
    icm=[]
    for seed in run.SEEDS:
        row=json.loads((run.previous.OUT/f'icm_common_exposure_seed{seed}.json').read_text())
        assert row['signature']==run.previous.check_lock()['signature']
        icm.append(row)
    for reference in ('original','broad_observations'):
        for key in ('affine_log_r2','goal_error','folded'):
            comparisons.append({'contrast':'icm_broad_vs_ou_'+reference,**compare(icm,groups[reference],key)})
    assert len(comparisons)==16
    for row,p in zip(comparisons,helpers.holm([x['p'] for x in comparisons])):
        row['holm_p']=p
    means={}
    for arm,rows in groups.items():
        means[arm]={key:float(np.mean([value(r,key) for r in rows])) for key in OUTCOMES}
        means[arm]['fold_count']=sum(r['metrics']['folded'] for r in rows)
        means[arm]['raw_output_mean']=float(np.mean([r['gaussian']['raw_mean'] for r in rows]))
        means[arm]['raw_output_std']=float(np.mean([r['gaussian']['raw_std'] for r in rows]))
    sampler=[compare(groups['ou_observations'],groups['original'],key) for key in OUTCOMES]
    result={'signature':run.signature(),'complete':True,'seeds':list(run.SEEDS),'means':means,
            'holm_family_size':16,'comparisons':comparisons,'sampler_comparison_descriptive':sampler,
            'icm_reference_previously_observed':True,
            'interpretation':'Positive benefit means better recovery/lower error. Sign tests concern direction, not mean-null or equivalence.'}
    run.save_json(run.OUT/'INFERENCE.json',result)
    lines=['# Does broad SIGReg exposure help ordinary OU?','',
           'Sixty paired seeds, three OU conditions. The alignment pairs are identical; the SIGReg formula/weight and training budget stay fixed. [Fixed plan](../PROTOCOL.md).','',
           '| OU training condition | Bounded-grid R² | Held-out Gaussian R² | Raw orthogonal MSE ↓ | Folds / 60 | Physical goal error ↓ |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    names={'original':'Original coupled OU','ou_observations':'Independent OU observations','broad_observations':'Independent broad observations'}
    for arm,m in means.items():
        lines.append(f"| {names[arm]} | {m['affine_log_r2']:.6f} | {m['gaussian_r2']:.6f} | {m['orthogonal_mse']:.6f} | {m['fold_count']} | {m['goal_error']:.6f} |")
    lines+=['','R² permits a fitted scale and offset. Raw orthogonal MSE permits only a sign change, matching a one-dimensional rotation. Thus high R² alone is not sufficient evidence of orthogonal recovery. The held-out Gaussian states include distances outside 1–16; OU alignment is not clipped. Broad SIGReg observations cover only 1–16.','',
            '| Comparison | Outcome | Mean benefit [descriptive 95% interval] | Better / worse / tie | Holm p |',
            '| --- | --- | ---: | ---: | ---: |']
    for r in comparisons:
        lo,hi=r['ci95_unadjusted']
        lines.append(f"| {r['contrast']} | {r['outcome']} | {r['mean_benefit']:+.6g} [{lo:+.6g}, {hi:+.6g}] | {r['wins']}/{r['losses']}/{r['ties']} | {r['holm_p']:.4g} |")
    lines+=['','Positive benefit means the first condition is better. All 16 tests use one Holm correction. Tests are two-sided sign/McNemar tests of direction; the unadjusted bootstrap intervals describe mean effects. Nonsignificance is not equivalence.','',
            'The broad-versus-independent-OU contrast isolates the observation pool. Original-versus-independent-OU also changes sampler/gradient implementation; its descriptive comparison is in the JSON.','',
            'ICM comparisons reuse the preceding native-ICM-alignment/broad-SIGReg runs on the same 60 seeds. These compare fixed recipes with different transition laws and step scales, not identical stationary joint distributions. OU supplies no physical action labels; every planning evaluation trains its predictor on the same shared physical-action data.','',
            '[Exact results](INFERENCE.json) · [Verification](VERIFICATION.json) · [Graphs](FIGURES.md) · [Code/time lock](LOCK.json)','']
    (run.OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'means':means,'pool_change_tests':[r for r in comparisons[:5] if r['outcome']!='folded']},default=str))


if __name__=='__main__':
    main()

"""Margin-free blocked tests of policy performance in both matched studies.

The primary statistic targets policy means (permutation repeated-measures ANOVA).
Friedman ranks are a sensitivity analysis. Seeds, not goals, are the blocks.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matched_equivalence as data


def blocked_f(matrix,permutations=100000):
    n,k=matrix.shape
    centered=matrix-matrix.mean(1,keepdims=True)
    within=float(np.sum(centered**2))
    if within==0:
        return {'statistic':0.,'p':1.,'df_policy':k-1,'df_residual':(n-1)*(k-1),'permutations':0,'all_within_seed_equal':True}
    def statistic(totals):
        policy=np.sum(totals**2,axis=-1)/n
        residual=np.maximum(within-policy,0)
        with np.errstate(divide='ignore',invalid='ignore'):
            return (n-1)*policy/residual
    observed=float(statistic(centered.sum(0)))
    rng=np.random.default_rng(20261108);exceed=0
    for start in range(0,permutations,2000):
        count=min(2000,permutations-start)
        order=np.argsort(rng.random((count,n,k)),axis=2)
        shuffled=np.take_along_axis(centered[None,:,:],order,axis=2)
        values=statistic(shuffled.sum(1))
        tolerance=1e-10*max(1,abs(observed)) if np.isfinite(observed) else 0.
        exceed+=int(np.sum(values>=observed-tolerance))
    return {'statistic':observed,'p':(exceed+1)/(permutations+1),
            'df_policy':k-1,'df_residual':(n-1)*(k-1),'permutations':permutations,
            'exceedances':exceed,'all_within_seed_equal':False,
            'assumption':'Exchangeability of policy labels within seed under the null; no parametric F reference distribution used.'}


def main():
    studies=data.load_studies();assert all(s['complete'] for s in studies)
    rows=[]
    for study in studies:
        for endpoint in data.ENDPOINTS:
            matrix=np.array([[b[name][endpoint] for name in study['names']] for b in study['blocks']])
            mean_test=blocked_f(matrix)
            rank_test=data.friedman_permutation(matrix)
            rows.append({'study':study['name'],'endpoint':endpoint,'seeds':len(study['seeds']),
                         'policies':len(study['names']),'mean_test':mean_test,'rank_test':rank_test})
    all_tests=[row[key] for row in rows for key in ('mean_test','rank_test')]
    for test,adjusted in zip(all_tests,data.holm([t['p'] for t in all_tests])):test['holm_p']=adjusted
    result={'post_hoc':True,'primary':'Seed-blocked permutation repeated-measures ANOVA (policy means)',
            'sensitivity':'Seed-blocked Friedman permutation test (policy ranks)',
            'permutations_per_nondegenerate_test':100000,'multiplicity_family':12,
            'margins_used':False,'rows':rows,
            'interpretation':'Failure to reject a policy effect does not prove exact equality or practical equivalence. Proposed margins were not endorsed; equivalence remains a separate illustrative sensitivity analysis.'}
    (data.OUT/'MATCHED_OMNIBUS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Actual hypothesis tests for matched-policy performance','',
           '**Primary test:** permutation repeated-measures ANOVA, with seed as the block. It tests policy mean differences without choosing an equivalence margin. **Sensitivity check:** Friedman permutation test of policy ranks. Both use 100,000 within-seed label permutations. No goals or grid points are treated as independent replicates.','',
           'Null: no systematic policy effect within matched seed blocks. Alternative: some policy differs. Permutation inference assumes within-seed policy-label exchangeability under the null. One Holm correction covers all 12 mean/rank tests across both studies and three endpoints.','',
           '| Study | Outcome | Seeds × policies | F statistic | Mean-test permutation p | Mean-test Holm p | Friedman statistic | Rank-test permutation p |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        a,b=r['mean_test'],r['rank_test']
        lines += [f"| {r['study']} | {r['endpoint']} | {r['seeds']} × {r['policies']} | {a['statistic']:.4f} | {a['p']:.4f} | {a['holm_p']:.4f} | {b['statistic']:.4f} | {b['p']:.4f} |"]
    lines += ['', '**Conclusion:** no systematic policy-performance difference is detected in these matched conditions at alpha 0.05. This is not proof that their true performances are exactly equal. It also does not imply equal performance in the original native-policy zoo.','',
              '- The earlier study used four modified stationary policies with both endpoint marginals matched.',
              '- The new study uses seven original policies with starting observations and SIGReg exposure matched; successors and stationarity are not forced to match.',
              '- “Native” prediction here means each matched cell’s own policy/source data; for common-reset cells those are reset transitions, not the old native trajectories.',
              '- [Optional equivalence sensitivity](MATCHED_EQUIVALENCE.md) uses illustrative margins and extra assumptions. These margins were chosen after observing results; the primary tests above use no margins.', '',
              '[Exact statistics, counts and permutation p-values](MATCHED_OMNIBUS.json).','']
    (data.OUT/'MATCHED_OMNIBUS.md').write_text('\n'.join(lines))
    for r in rows:print(r['study'],r['endpoint'],r['mean_test']['statistic'],r['mean_test']['p'],r['rank_test']['p'])


if __name__=='__main__':main()

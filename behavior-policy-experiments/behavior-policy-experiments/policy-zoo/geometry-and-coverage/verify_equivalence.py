"""Independently check paired mean/median TOST and omnibus corrections."""
import json
import numpy as np
from scipy import stats
import matched_equivalence as analysis


def main():
    output=analysis.OUT
    result=json.loads((output/'MATCHED_EQUIVALENCE.json').read_text())
    studies=analysis.load_studies();assert all(s['complete'] for s in studies)
    pair_count=0;omnibus=[]
    for study,reported in zip(studies,result['studies']):
        assert study['name']==reported['name'] and study['seeds']==reported['seeds']
        for endpoint in analysis.ENDPOINTS:
            matrix=np.array([[block[name][endpoint] for name in study['names']] for block in study['blocks']])
            expected=reported['endpoints'][endpoint]
            margin=expected['equivalence']['margin']
            mean_ps=[];median_ps=[]
            for pair in expected['equivalence']['pairs']:
                a,b=study['names'].index(pair['policy_a']),study['names'].index(pair['policy_b'])
                delta=matrix[:,a]-matrix[:,b]
                if np.std(delta,ddof=1)>0:
                    lower=stats.ttest_1samp(delta,-margin,alternative='greater').pvalue
                    upper=stats.ttest_1samp(delta,margin,alternative='less').pvalue
                    ci=stats.ttest_1samp(delta,0).confidence_interval(confidence_level=.90)
                    np.testing.assert_allclose(pair['mean_ci90'],[ci.low,ci.high],atol=1e-14,rtol=1e-12)
                    p=max(lower,upper)
                else:
                    p=0. if -margin<delta[0]<margin else 1.
                    np.testing.assert_array_equal(pair['mean_ci90'],[delta[0],delta[0]])
                assert np.isclose(p,pair['mean_tost_p'],atol=1e-14,rtol=1e-12)
                p_median=max(stats.binomtest(int(np.sum(delta>-margin)),len(delta),.5,alternative='greater').pvalue,
                             stats.binomtest(int(np.sum(delta<margin)),len(delta),.5,alternative='greater').pvalue)
                assert abs(p_median-pair['median_sign_tost_p'])<1e-14
                mean_ps.append(p);median_ps.append(p_median);pair_count+=1
            eq=expected['equivalence']
            assert np.isclose(max(mean_ps),eq['mean_global_p'],atol=1e-14,rtol=1e-12)
            assert abs(max(median_ps)-eq['median_global_p'])<1e-14
            permutation=analysis.friedman_permutation(matrix,expected['omnibus']['permutations'] or 100000)
            assert abs(permutation['p']-expected['omnibus']['p'])<1e-14
            omnibus.append(expected['omnibus'])
        primary=max(reported['endpoints'][e]['equivalence']['mean_global_p'] for e in ('recovery','shared_goal_error'))
        median=max(reported['endpoints'][e]['equivalence']['median_global_p'] for e in ('recovery','shared_goal_error'))
        all_three=max(reported['endpoints'][e]['equivalence']['mean_global_p'] for e in analysis.ENDPOINTS)
        assert primary==reported['primary_joint_mean_p']
        assert median==reported['primary_joint_median_p']
        assert all_three==reported['all_three_joint_mean_p']
    assert pair_count==81
    sorted_p=sorted(r['p'] for r in omnibus)
    for row in omnibus:
        index=max(i for i,p in enumerate(sorted_p) if p<=row['p'])
        adjusted=min(1.,max((len(sorted_p)-i)*p for i,p in enumerate(sorted_p[:index+1])))
        assert abs(adjusted-row['holm_p'])<1e-14
    # Tiny exact enumeration validates the blocked-permutation implementation.
    import itertools
    matrix=np.array([[1.,2.,3.],[1.,2.,3.]])
    choices=list(itertools.permutations([1.,2.,3.]))
    observed=stats.friedmanchisquare(*matrix.T).statistic
    exact=np.mean([stats.friedmanchisquare(*np.array([a,b]).T).statistic>=observed-1e-12 for a in choices for b in choices])
    simulated=analysis.friedman_permutation(matrix)['p']
    assert abs(simulated-exact)<5*np.sqrt(exact*(1-exact)/100000)+1/100001
    record={'pairwise_equivalence_checks':pair_count,'omnibus_tests_recomputed':len(omnibus),
            'global_iut_checks':12,'exact_small_permutation_p':float(exact),'simulated_small_permutation_p':simulated,
            'errors':0,'scope':'Mean and paired-median equivalence have different estimands/assumptions; no implication of exact equality.'}
    (output/'EQUIVALENCE_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()

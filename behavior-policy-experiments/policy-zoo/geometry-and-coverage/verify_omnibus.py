"""Validate blocked mean tests with OLS and exact small-case enumeration."""
import itertools
import json
import numpy as np
from scipy import stats
import matched_equivalence as data
import omnibus_test as analysis


def ols_f(matrix):
    n,k=matrix.shape
    y=matrix.ravel()
    blocks=np.repeat(np.eye(n),k,axis=0)
    policies=np.tile(np.eye(k),(n,1))[:,:-1]
    full=np.column_stack([blocks,policies])
    residual0=y-blocks@np.linalg.lstsq(blocks,y,rcond=None)[0]
    residual1=y-full@np.linalg.lstsq(full,y,rcond=None)[0]
    ss0=float(residual0@residual0);ss1=float(residual1@residual1)
    return ((ss0-ss1)/(k-1))/(ss1/((n-1)*(k-1)))


def main():
    result=json.loads((data.OUT/'MATCHED_OMNIBUS.json').read_text())
    studies={s['name']:s for s in data.load_studies()}
    checked=[]
    for row in result['rows']:
        study=studies[row['study']]
        matrix=np.array([[block[n][row['endpoint']] for n in study['names']] for block in study['blocks']])
        independent=ols_f(matrix)
        assert np.isclose(independent,row['mean_test']['statistic'],atol=1e-7,rtol=1e-7)
        repeated=analysis.blocked_f(matrix)
        assert repeated['p']==row['mean_test']['p']
        checked.append({'study':row['study'],'endpoint':row['endpoint'],'ols_f':float(independent)})
    all_tests=[r[key] for r in result['rows'] for key in ('mean_test','rank_test')]
    ps=sorted(r['p'] for r in all_tests)
    for row in all_tests:
        index=max(i for i,p in enumerate(ps) if p<=row['p'])
        adjusted=min(1.,max((len(ps)-i)*p for i,p in enumerate(ps[:index+1])))
        assert abs(adjusted-row['holm_p'])<1e-14
    toy=np.array([[1.,2.,3.],[1.,2.,3.]])
    choices=list(itertools.permutations([-1.,0.,1.]))
    observed=4.
    exact=np.mean([np.sum(np.array([a,b]).sum(0)**2)/2>=observed-1e-12 for a in choices for b in choices])
    simulated=analysis.blocked_f(toy)['p']
    assert abs(simulated-exact)<5*np.sqrt(exact*(1-exact)/100000)+1/100001
    tied=analysis.blocked_f(np.repeat(np.arange(3)[:,None],4,axis=1))
    assert tied['statistic']==0 and tied['p']==1
    record={'ols_statistic_checks':checked,'mean_tests_reproduced':6,'holm_tests_checked':12,
            'small_exact_p':float(exact),'small_permutation_p':simulated,'all_tied_case_passed':True,'errors':0}
    (data.OUT/'OMNIBUS_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()

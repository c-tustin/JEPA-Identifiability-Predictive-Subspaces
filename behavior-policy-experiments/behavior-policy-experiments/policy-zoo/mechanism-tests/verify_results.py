"""Check every crossed curve/score and independently replay four off-diagonals."""

import hashlib
import json
from math import comb

import numpy as np
from scipy import stats
import torch

from run_crossed import EVAL_SEED, HERE, LEVELS, OUT, POLICIES, ZOO_OUT, behavior, controls, geometry, load, reference, reliability, run_zoo, sha
from report_crossed import fold_flag


def main():
    torch.set_num_threads(1)
    settings=json.loads((OUT/'crossed-settings.json').read_text())
    assert settings['code_sha256']==sha(HERE/'run_crossed.py')
    assert settings['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    c=behavior.Config(**json.loads((ZOO_OUT/'config.json').read_text())['config'])
    common=reference(c,EVAL_SEED)
    rows=[]
    parameter_errors=[]
    max_curve_error,max_score_error=0.,0.
    sources={}
    for p in POLICIES:
        for d in LEVELS:
            source=load(OUT/'sources'/f'{p}-{d}.pt')
            sources[p,d]=source
            old=json.loads((ZOO_OUT/'runs'/f'pure-{p}_seed{d}.json').read_text())
            assert run_zoo.diagnostics(source,c)['sha256']==old['source_stats']['sha256']
    initial_by_seed={}
    indices_by_data={}
    diagonal=0
    matrices=json.loads((OUT/'crossed-summary.json').read_text())['matrices']
    for p in POLICIES:
        for d in LEVELS:
            for i in LEVELS:
                saved=load(OUT/'crossed'/f'{p}-data{d}-init{i}.pt')
                model,plan=reliability.make_plan(c,d,i,'icm')
                expected_initial=geometry(model,c)
                assert np.array_equal(expected_initial['curve'],saved['initial']['curve'])
                if i in initial_by_seed:assert np.array_equal(initial_by_seed[i],saved['initial']['curve'])
                initial_by_seed[i]=saved['initial']['curve']
                digest=hashlib.sha256(plan['indices'].numpy().tobytes()).hexdigest()
                assert digest==saved['pair_indices_sha256']
                if d in indices_by_data:assert digest==indices_by_data[d]
                indices_by_data[d]=digest
                model.load_state_dict(saved['encoder'])
                recreated=geometry(model,c)
                curve_error=max(float(np.max(np.abs(recreated[k]-saved['final'][k]))) for k in ('curve','slope'))
                assert curve_error==0.
                max_curve_error=max(max_curve_error,curve_error)
                metric,_=controls.evaluate(model,common,c,EVAL_SEED)
                error=abs(metric['common_grid_r2']-saved['metrics']['common_grid_r2'])
                assert error<1e-12
                max_score_error=max(max_score_error,error)
                row,col=LEVELS.index(d),LEVELS.index(i)
                assert metric['common_grid_r2']==matrices[p]['r2'][row][col]
                assert fold_flag(model,c)==matrices[p]['folded'][row][col]
                if d==i:
                    old=load(ZOO_OUT/'checkpoints'/f'encoder_pure-{p}_data{d}_init{i}.pt')
                    assert all(torch.equal(v,old['encoder'][k]) for k,v in model.state_dict().items())
                    old_metric=json.loads((ZOO_OUT/'runs'/f'pure-{p}_seed{d}.json').read_text())['metrics']
                    assert abs(saved['original_metrics']['common_grid_r2']-old_metric['common_grid_r2'])<1e-7
                    assert fold_flag(model,c)==(old_metric['reversed_grid_fraction']>0)
                    diagonal+=1
    for p in POLICIES:
        d,i=8,9  # Same preselected off-diagonal for every source.
        model,plan=reliability.make_plan(c,d,i,'icm')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        regularizer=behavior.SIGReg1D()
        source=sources[p,d]
        for step in range(c.jepa_steps):
            ix=plan['indices'][step]
            controls.update(model,optimizer,regularizer,source['x'][ix],source['xp'][ix],c,controls.learning_rate(step,c.jepa_steps))
        saved=load(OUT/'crossed'/f'{p}-data{d}-init{i}.pt')
        error=max(float((v-saved['encoder'][k]).abs().max()) for k,v in model.state_dict().items())
        assert error==0.,(p,error)
        parameter_errors.append(error)
        print(f'Exact off-diagonal replay: {p}, data 8, init 9',flush=True)

    tests=json.loads((OUT/'targeted-tests.json').read_text())['families']
    checked_signs=0
    for family in ('ranking','persistence','baseline_change','associations'):
        group=tests[family]
        for r in group:
            n=r['wins']+r['losses']
            expected=1. if n==0 else min(1.,2*sum(comb(n,k) for k in range(min(r['wins'],r['losses'])+1))/2**n)
            assert abs(expected-r['p'])<1e-12
            checked_signs+=1
        ordered=sorted(group,key=lambda r:r['p'])
        passing=True
        for rank,r in enumerate(ordered):
            passing=passing and r['p']<=.05/(len(group)-rank)
            assert (r['p_holm']<=.05)==passing
    for r in tests['equivalence_TOST']:
        name=r['name'];delta=r['margin']
        d=np.array([json.loads((ZOO_OUT/'runs'/f'pure-{name}_seed{s}.json').read_text())['metrics']['common_grid_r2']
                    -json.loads((ZOO_OUT/'runs'/f'pure-iid_{name}_seed{s}.json').read_text())['metrics']['common_grid_r2'] for s in range(8,28)])
        expected=max(stats.ttest_1samp(d,-delta,alternative='greater').pvalue,stats.ttest_1samp(d,delta,alternative='less').pvalue)
        assert np.isclose(expected,r['p'],rtol=1e-10,atol=1e-20)
    result=dict(passed=True,sources_checked=20,crossed_models_checked=100,diagonal_reproductions=diagonal,
                max_curve_or_slope_error=max_curve_error,max_metric_error=max_score_error,
                off_diagonal_full_replays=4,off_diagonal_parameter_errors=parameter_errors,
                sign_tests_checked=checked_signs,TOST_checks=16,
                scope='All sources, initial curves, final curves/scores, fold matrices and batch-index invariants; 20 original checkpoints and four full off-diagonal training replays. Not 100 full training replays or a convergence theorem.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

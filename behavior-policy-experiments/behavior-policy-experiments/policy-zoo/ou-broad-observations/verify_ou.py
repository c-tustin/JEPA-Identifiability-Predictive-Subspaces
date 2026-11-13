"""Recreate every OU source, recovery diagnostic, rollout and planned test."""
import json
import multiprocessing as mp
import numpy as np
from scipy import stats
import torch
import run_ou as run
import analyze_ou as analysis


def same(left,right):
    assert left.keys()==right.keys()
    for key in left:
        np.testing.assert_allclose(left[key],right[key],atol=1e-11,rtol=0)


def check_seed(seed):
    torch.set_num_threads(1)
    data,broad=run.source(seed)
    import hashlib
    poolhash=hashlib.sha256(broad.numpy().tobytes()).hexdigest()
    icm=json.loads((run.previous.OUT/f'icm_common_exposure_seed{seed}.json').read_text())
    assert poolhash==icm['regularizer_pool_hash']
    for arm in run.ARMS:
        row=json.loads((run.OUT/f'{arm}_seed{seed}.json').read_text())
        assert row['signature']==run.signature() and row['source_hash']==run.world.source_hash(data)
        if arm!='original':
            pool=run.world.pooled(data) if arm=='ou_observations' else broad
            assert row['regularizer_pool_hash']==hashlib.sha256(pool.numpy().tobytes()).hexdigest()
        state=torch.load(run.OUT/f'encoder_{arm}_seed{seed}.pt',map_location='cpu')
        assert state['signature']==run.signature() and state['step']==4000
        assert all(int(s['step'])==4000 for s in state['optimizer']['state'].values())
        model=run.previous.behavior.Encoder();model.load_state_dict(state['encoder']);model.eval()
        metrics,curve=run.original.geometry(model)
        same(metrics,row['metrics']);same(run.gaussian_metrics(model,seed),row['gaussian'])
        rng=torch.Generator().manual_seed(9400000+seed)
        z=torch.randn(16384,generator=rng)
        with torch.no_grad():
            y=model(run.previous.behavior.observe(z,run.original.config().sigma)).numpy().astype(float)
        target=z.numpy().astype(float)
        assert np.isclose(np.corrcoef(y,target)[0,1]**2,row['gaussian']['gaussian_r2'],atol=1e-13)
        mse=min(np.mean((y-target)**2),np.mean((y+target)**2))
        assert abs(mse-row['gaussian']['orthogonal_mse'])<1e-13
        arrays=np.load(run.OUT/f'{arm}_seed{seed}.npz')
        for key,value in curve.items():np.testing.assert_array_equal(value,arrays[key])
        saved=torch.load(run.OUT/f'predictor_{arm}_seed{seed}.pt',map_location='cpu')
        predictor=run.world.prior.matched.planning.Predictor();predictor.load_state_dict(saved['predictor'])
        planning,paths=run.world.prior.matched.planning_evaluation(model,predictor,saved['mean'],saved['scale'],seed)
        same(planning,row['planning']);assert planning['oracle_error']==0
        for key,value in paths.items():np.testing.assert_array_equal(value,arrays[key])
    return 3


def equal(left,right):
    if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,atol=0,rtol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):equal(a,b)
    else:assert left==right


def resume_gate():
    seed=6000;arm='broad_observations';output=run.OUT/'gates/resumed'
    result=output/f'{arm}_seed{seed}.json'
    if not result.exists() and not (output/'partials'/f'{arm}_seed{seed}.pt').exists():
        run.train(arm,seed,output,stop_at=500)
    if not result.exists():
        assert torch.load(output/'partials'/f'{arm}_seed{seed}.pt',map_location='cpu')['step']==500
    actual=run.train(arm,seed,output)
    expected=json.loads((run.OUT/f'{arm}_seed{seed}.json').read_text())
    assert actual==expected
    for name in [f'encoder_{arm}_seed{seed}.pt',f'predictor_{arm}_seed{seed}.pt']:
        equal(torch.load(output/name,map_location='cpu'),torch.load(run.OUT/name,map_location='cpu'))
    return {'seed':seed,'arm':arm,'resumed_at':500,'all_final_states_and_results_exact':True}


def check_inference(groups):
    inference=json.loads((run.OUT/'INFERENCE.json').read_text())
    assert inference['signature']==run.signature() and len(inference['comparisons'])==16
    icm=[json.loads((run.previous.OUT/f'icm_common_exposure_seed{s}.json').read_text()) for s in run.SEEDS]
    for record in inference['comparisons']:
        name=record['contrast'];key=record['outcome']
        if name.startswith('icm_broad_vs_ou_'):
            treatment=icm;reference=groups[name.removeprefix('icm_broad_vs_ou_')]
        else:
            treatment=groups['broad_observations'];reference=groups[name.removeprefix('broad_vs_')]
        d=np.array([analysis.value(a,key)-analysis.value(b,key) for a,b in zip(treatment,reference)])
        if key not in ('affine_log_r2','gaussian_r2'):d*=-1
        np.testing.assert_array_equal(d,record['seed_benefits'])
        positive=int(np.sum(d>1e-10));negative=int(np.sum(d < -1e-10));n=positive+negative
        p=min(1.,2*stats.binom.sf(max(positive,negative)-1,n,.5)) if n else 1.
        assert abs(p-record['p'])<1e-14
        assert abs(d.mean()-record['mean_benefit'])<1e-13
    records=inference['comparisons'];order=np.argsort([r['p'] for r in records]);largest=0.
    for rank,index in enumerate(order):
        largest=min(1.,max(largest,(16-rank)*records[index]['p']))
        assert abs(largest-records[index]['holm_p'])<1e-14
    return 16


if __name__=='__main__':
    run.check_lock();groups,missing=analysis.collect();assert not missing
    with mp.get_context('spawn').Pool(6) as pool:
        checked=list(pool.imap_unordered(check_seed,run.SEEDS))
    replay=resume_gate()
    stats_checked=check_inference(groups)
    old=json.loads((run.OUT/'GATES.json').read_text());assert old['historical_ou_seed13_bit_exact']
    result={'signature':run.signature(),'complete':True,'sources_recreated':60,
            'encoder_geometries_and_gaussian_diagnostics_replayed':sum(checked),
            'physical_planning_replays':sum(checked),'planned_tests_and_holm_checks':stats_checked,
            'same_broad_pool_as_icm_verified':True,'historical_original_ou_bit_exact':True,
            'independent_sigreg_resume_gate':replay,'errors':0}
    run.save_json(run.OUT/'VERIFICATION.json',result);print(json.dumps(result))

"""Replay sources, models, real planning, optimizer attribution and inference."""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp

import numpy as np
from scipy import stats
import torch
from torch.func import functional_call

import common_gc as study
import gc_statistics


def close(actual,expected,label):
    for key,value in actual.items():
        if isinstance(value,(list,float,int,bool)):
            assert np.allclose(value,expected[key],atol=1e-11,rtol=0),(label,key,value,expected[key])


def verify_source(task):
    policy,seed=task
    torch.set_num_threads(1)
    saved=torch.load(study.OUT/'sources'/f'{policy}_seed{seed}.pt',map_location='cpu')
    manifest=json.loads((study.OUT/'source_manifests'/f'{policy}_seed{seed}.json').read_text())
    assert saved['signature']==manifest['signature']==study.signature()
    native=study.native_data(policy,seed)
    reset=study.reset_data(policy,seed,native)
    for kind,data in [('native',native),('reset',reset)]:
        assert study.source_hash(data)==manifest[kind+'_sha256']
        for key,value in data.items():torch.testing.assert_close(value,saved[kind][key],atol=0,rtol=0)
    distance,pool=study.reset_pool(seed)
    torch.testing.assert_close(saved['pool'],pool,atol=0,rtol=0)
    torch.testing.assert_close(reset['x'],pool,atol=0,rtol=0)
    torch.testing.assert_close(study.behavior.sensor(study.behavior.move(distance,reset['action'],study.original.config()),study.original.config()),reset['xp'],atol=0,rtol=0)
    return manifest['reset_start_sha256'],seed


def inputs(row):
    if row['experiment']=='factor':
        data=study.factor_data(row['seed'],0 if row['alignment_source']=='early' else 128)
        pool=study.pooled(study.factor_data(row['seed'],0 if row['regularizer_source']=='early' else 128))
        initial=study.behavior.Encoder()
        path=study.prior.OUT/f"encoder_icm_age000_seed{row['seed']}.pt"
        parent=torch.load(path,map_location='cpu')
        initial.load_state_dict(parent['encoder'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['parent_checkpoint_sha256']
    else:
        saved=torch.load(study.OUT/'sources'/f"{row['policy']}_seed{row['seed']}.pt",map_location='cpu')
        data,pool=saved[row['alignment_source']],saved['pool']
        torch.manual_seed(40000+row['seed']);initial=study.behavior.Encoder()
    return data,pool,initial


def verify_result(path):
    torch.set_num_threads(1)
    row=json.loads(path.read_text());name,seed=row['condition'],row['seed']
    assert row['signature']==study.signature()
    data,pool,initial=inputs(row)
    assert study.source_hash(data)==row['source_hash']
    assert hashlib.sha256(pool.numpy().tobytes()).hexdigest()==row['regularizer_pool_sha256']
    model=study.behavior.Encoder()
    state=torch.load(study.OUT/f'encoder_{name}_seed{seed}.pt',map_location='cpu')
    assert state['signature']==study.signature()
    model.load_state_dict(state['encoder']);model.eval()
    metrics,curve=study.original.geometry(model);close(metrics,row['metrics'],name)
    initial_metrics,initial_curve=study.original.geometry(initial);close(initial_metrics,row['initial_metrics'],name+' initial')
    arrays=np.load(study.OUT/f'{name}_seed{seed}.npz')
    np.testing.assert_array_equal(curve['curve'],arrays['curve'])
    np.testing.assert_array_equal(initial_curve['curve'],arrays['initial_curve'])
    for kind,expected in row['planning'].items():
        state=torch.load(study.OUT/f'predictor_{name}_{kind}_seed{seed}.pt',map_location='cpu')
        predictor=study.prior.matched.planning.Predictor();predictor.load_state_dict(state['predictor']);predictor.eval()
        actual,paths=study.prior.matched.planning_evaluation(model,predictor,state['mean'],state['scale'],seed)
        close(actual,expected,name+' '+kind)
        for key,value in paths.items():np.testing.assert_array_equal(value,arrays[kind+'_'+key])
    assert len(row['planning'])==(1 if row['experiment']=='factor' else 2)
    return len(row['planning'])


def verify_snapshot(path):
    torch.set_num_threads(1)
    saved=torch.load(path,map_location='cpu')
    assert saved['signature']==study.signature()
    name,seed=saved['condition'],saved['seed']
    row=json.loads((study.OUT/f'{name}_seed{seed}.json').read_text())
    data,pool,_=inputs(row);c=study.original.config()
    model=study.behavior.Encoder();model.load_state_dict(saved['before'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(saved['optimizer_before'])
    ix,ir=saved['alignment_indices'],saved['regularizer_indices']
    y=model(torch.cat([data['x'][ix],data['xp'][ix]]))
    a=.9*(y[:c.batch]-y[c.batch:]).square().mean()
    r=.1*study.behavior.SIGReg1D()(model(pool[ir]))
    named=list(model.named_parameters());parameters=[p for _,p in named]
    ga=torch.autograd.grad(a,parameters);gr=torch.autograd.grad(r,parameters)
    attribution=saved['attribution']
    for (n,p),g1,g2 in zip(named,ga,gr):
        torch.testing.assert_close(g1,attribution['gradient_alignment'][n],atol=0,rtol=0)
        torch.testing.assert_close(g2,attribution['gradient_regularizer'][n],atol=0,rtol=0)
        p.grad=g1+g2
    optimizer.step()
    actual=study.frame(model)
    for key,value in actual.items():np.testing.assert_array_equal(value,attribution['after'][key])
    assert attribution['parameter_reconstruction_error']<5e-7
    # Independent central finite differences in double precision for mixed slope JVPs.
    double_model=study.behavior.Encoder().double()
    before={n:v.double() for n,v in saved['before'].items()}
    q=torch.linspace(0,np.log(16),257).double().requires_grad_(True)
    def evaluate(parameters):
        y=functional_call(double_model,parameters,(study.behavior.sensor(q.exp(),c),))
        slope=torch.autograd.grad(y.sum(),q)[0]
        return y.detach().numpy(),slope.detach().numpy()
    errors=[]
    for key,direction in attribution['directions'].items():
        direction={n:v.double() for n,v in direction.items()};eps=.01
        plus=evaluate({n:before[n]+eps*direction[n] for n in before})
        minus=evaluate({n:before[n]-eps*direction[n] for n in before})
        for index,quantity in enumerate(('curve','slope')):
            fd=(plus[index]-minus[index])/(2*eps)
            expected=attribution['components'][key]['delta_'+quantity+'_jvp']
            error=float(np.max(np.abs(fd-expected)));errors.append(error)
            assert error<1e-4*(1+float(np.max(np.abs(fd))))
    return {'parameter_error':attribution['parameter_reconstruction_error'],
            'moment_error':attribution['moment_reconstruction_error'],'max_jvp_fd_error':max(errors)}


def verify_expected_loss():
    from diagnose_geometry import expected_losses
    torch.set_num_threads(1)
    data=study.factor_data(200,128);pool=study.pooled(data)
    model=study.behavior.Encoder()
    model.load_state_dict(torch.load(study.prior.OUT/'encoder_icm_age000_seed200.pt',map_location='cpu')['encoder'])
    model.eval()
    expected=expected_losses(model,data,pool)['expected_sigreg_iid_1024']
    with torch.no_grad():values=model(pool).double()
    reg=study.behavior.SIGReg1D().double();rng=torch.Generator().manual_seed(20261106)
    draws=[]
    for _ in range(512):
        ix=torch.randint(len(values),(1024,),generator=rng)
        draws.append(float(reg(values[ix])))
    se=float(np.std(draws,ddof=1)/np.sqrt(len(draws)))
    assert abs(np.mean(draws)-expected)<5*se+.001
    return {'analytic':expected,'monte_carlo_mean':float(np.mean(draws)),'monte_carlo_se':se,'batches':512}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    paths=sorted(study.OUT.glob('*_seed*.json'))
    paths=[p for p in paths if 'experiment' in json.loads(p.read_text())]
    sources=[(p,s) for s in study.ZOO_SEEDS for p in study.ZOO_POLICIES if (study.OUT/'source_manifests'/f'{p}_seed{s}.json').exists()]
    snapshots=sorted((study.OUT/'snapshots').glob('*.pt'))
    if args.require_complete:
        assert len(paths)==360 and len(sources)==140 and len(snapshots)==100
        expected={(f'factor_A{a}_R{r}',s) for s in study.FACTOR_SEEDS for a in ('early','late') for r in ('early','late')}
        expected|={(f'zoo_{p}_{a}',s) for s in study.ZOO_SEEDS for p in study.ZOO_POLICIES for a in ('native','reset')}
        actual={(r['condition'],r['seed']) for r in [json.loads(p.read_text()) for p in paths]}
        assert expected==actual
    with mp.get_context('spawn').Pool(args.workers) as pool:
        source_results=list(pool.imap_unordered(verify_source,sources))
        planning_counts=list(pool.imap_unordered(verify_result,paths))
        snapshot_results=list(pool.imap_unordered(verify_snapshot,snapshots))
    for seed in study.ZOO_SEEDS:
        hashes={h for h,s in source_results if s==seed}
        if hashes:assert len(hashes)==1
    rows=[json.loads(p.read_text()) for p in paths]
    inference=gc_statistics.comparisons(rows)
    assert inference==json.loads((study.OUT/'INFERENCE.json').read_text())
    if args.require_complete:assert inference['complete'] and len(inference['tests'])==54
    ps=sorted(r['p'] for r in inference['tests'])
    for row in inference['tests']:
        n=row['wins']+row['losses']
        expected_p=stats.binomtest(row['wins'],n,.5).pvalue if n else 1.
        assert abs(expected_p-row['p'])<1e-14
        index=max(i for i,p in enumerate(ps) if p<=row['p'])
        adjusted=min(1.,max((len(ps)-i)*p for i,p in enumerate(ps[:index+1])))
        assert abs(adjusted-row['holm_p'])<1e-14
    result={'signature':study.signature(),'source_pairs_recreated':len(source_results),'geometry_replays':len(paths),
            'physical_planning_replays':sum(planning_counts),'gradient_adam_snapshots_replayed':len(snapshot_results),
            'max_parameter_update_error':max([r['parameter_error'] for r in snapshot_results] or [0.]),
            'max_jvp_finite_difference_error':max([r['max_jvp_fd_error'] for r in snapshot_results] or [0.]),
            'expected_loss_formula_check':verify_expected_loss(),'exact_tests_checked':54,
            'complete':args.require_complete,'errors':0}
    if args.require_complete:assert result['physical_planning_replays']==640
    study.save_json(study.OUT/('VERIFICATION.json' if args.require_complete else 'PARTIAL_VERIFICATION.json'),result)
    print(json.dumps(result))


if __name__=='__main__':main()

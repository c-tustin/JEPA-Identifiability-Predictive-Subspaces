"""Factor near mass, near conditional shape and far conditional shape."""
import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('feature_reference',HERE.parent/'initialization-crossing/run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
torch,np,B=R.torch,R.np,R.B
OUT=HERE/'output'
PAIRS=((130,131),(134,132),(135,133),(137,136),(138,139))


def dependency_hashes():
    paths=[Path(__file__),HERE/'PROTOCOL.md',HERE.parent/'initialization-crossing/run.py',
           R.LEGACY/'run_trials.py',R.LEGACY.parent/'identifiability/reproduce.py',
           R.LEGACY.parent/'identifiability/sigreg.py',R.LEGACY.parent/'curiosity/experiment.py',
           R.LEGACY.parent/'reliability/run_study.py',R.LEGACY.parent/'curriculum-controls/run_controls.py']
    return {str(p):R.digest(p.read_bytes()) for p in paths}


def sources():
    result={}
    for seed in range(130,140):
        path=R.LEGACY/'output/sources'/f'icm-{seed}.pt';s=R.trials.load(path)
        x=torch.cat([s['data']['x'],s['data']['xp']]);d=1/x[:,1]
        order=d.argsort(stable=True);x,d=x[order],d[order]
        result[seed]=dict(x=x,d=d,near=x[d<2],far=x[d>=2],mass=float((d<2).double().mean()),
                          data=s['data'],config=s['config'],source_hash=R.digest(path.read_bytes()))
    return result


def indices(alpha,near_count,far_count,u):
    near=(u/alpha*near_count).long().clamp(0,near_count-1)
    far=((u-alpha)/(1-alpha)*far_count).long().clamp(0,far_count-1)+near_count
    return torch.where(u<alpha,near,far)


def setup():
    ds=sources();c=B.Config(**ds[131]['config'])
    assert all(s['config']==vars(c) for s in ds.values())
    model,plan=R.trials.study.make_plan(c,131,131,'icm')
    u=torch.rand((4000,1024),dtype=torch.float64,generator=torch.Generator().manual_seed(8400131))
    return ds,c,model.state_dict(),plan['indices'],u


def validate(ds,c,initial,ix,u):
    rows=[]
    for seed,s in ds.items():
        n=len(s['x']);alpha=s['mass']
        # Midpoints cover every atom once, including multiplicities.
        mids=(torch.arange(n,dtype=torch.float64)+.5)/n
        ids=indices(alpha,len(s['near']),len(s['far']),mids)
        assert torch.equal(ids,torch.arange(n))
        assert torch.equal(torch.cat([s['near'],s['far']]),s['x'])
        d=s['d'].double().numpy();q=np.log(d);h=1/d
        rows.append(dict(seed=seed,mass_below2=alpha,
            distance_quantiles=np.quantile(d,[0,.1,.25,.5,.75,.9,.95,.99,1]).tolist(),
            near_quantiles=np.quantile(d[d<2],[.1,.25,.5,.75,.9,.99]).tolist(),
            far_quantiles=np.quantile(d[d>=2],[.1,.25,.5,.75,.9,.99]).tolist(),
            distance_mean=float(d.mean()),log_distance_mean=float(q.mean()),
            log_distance_std=float(q.std()),apparent_height_std=float(h.std()),
            mass_at_thresholds={str(t):float((d<t).mean()) for t in [1.1,1.25,1.5,2,3,4,8]},
            source_hash=s['source_hash']))
    audit=dict(passed=True,native_empirical_corners_exact=True,profiles=rows,
               initial_hash=R.state_hash(initial),alignment_plan_hash=R.digest(ix.numpy().tobytes()),
               uniform_plan_hash=R.digest(u.numpy().tobytes()),dependencies=dependency_hashes())
    R.trials.atomic_json(OUT/'preflight.json',audit)
    print(json.dumps(dict(passed=True,profiles=[{k:v for k,v in r.items() if k in ['seed','mass_below2','near_quantiles','far_quantiles']} for r in rows])),flush=True)


def fit(pair,bits,ds,c,initial,ix,u):
    ms,ns,fs=[pair[k] for k in bits]
    name=f'{pair[0]}-{pair[1]}-m{bits[0]}n{bits[1]}f{bits[2]}'
    path=OUT/'runs'/f'{name}.pt'
    x=torch.cat([ds[ns]['near'],ds[fs]['far']]);alpha=ds[ms]['mass']
    ids=indices(alpha,len(ds[ns]['near']),len(ds[fs]['far']),u)
    identity=dict(pair=pair,bits=bits,mass_source=ms,near_source=ns,far_source=fs,alpha=alpha,
        initial_hash=R.state_hash(initial),alignment_plan_hash=R.digest(ix.numpy().tobytes()),
        regularizer_index_hash=R.digest(ids.numpy().tobytes()),uniform_plan_hash=R.digest(u.numpy().tobytes()),
        source_hashes={s:ds[s]['source_hash'] for s in set([ms,ns,fs,131])},dependencies=dependency_hashes())
    if path.exists():
        assert R.trials.load(path)['identity']==identity
        return
    m=B.Encoder();m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
    data=ds[131]['data'];frames=[dict(step=0,**R.geometry(m,c,1025))];started=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        y=m(torch.cat([data['x'][ix[step]],data['xp'][ix[step]]]))
        alignment=(y[:c.batch]-y[c.batch:]).square().mean()
        penalty=reg(m(x[ids[step]]));loss=.9*alignment+.1*penalty
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,**R.geometry(m,c,1025)))
    sample_d=(1/x[:,1]);lo,hi=float(sample_d.min()),float(sample_d.max())
    saved=dict(identity=identity,model=m.state_dict(),initial=initial,optimizer=opt.state_dict(),
        config=vars(c),frames=frames,seconds=time.monotonic()-started,
        sampling=dict(realized_near_mass=float((u<alpha).double().mean()),target_near_mass=alpha,
                      near_count=len(ds[ns]['near']),far_count=len(ds[fs]['far']),support=[lo,hi]))
    R.trials.save_torch(path,saved)
    check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001)
    saved['geometry']['roots_in_source_support']=[d for d in saved['geometry']['roots'] if lo<=d<=hi]
    R.trials.save_torch(path,saved)
    row=dict(name=name,**identity,**saved['geometry'],sampling=saved['sampling'])
    R.trials.atomic_json(OUT/'cells'/f'{name}.json',row)
    print(json.dumps(dict(name=name,**saved['geometry'],seconds=saved['seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    a=parser.parse_args();torch.set_num_threads(1);ds,c,initial,ix,u=setup()
    if a.validate:return validate(ds,c,initial,ix,u)
    audit=json.loads((OUT/'preflight.json').read_text());assert audit['passed'] and audit['dependencies']==dependency_hashes()
    jobs=list(itertools.product(PAIRS,itertools.product([0,1],repeat=3)))
    for k,(pair,bits) in enumerate(jobs):
        if k%a.shards==a.shard:fit(pair,bits,ds,c,initial,ix,u)


if __name__=='__main__':main()

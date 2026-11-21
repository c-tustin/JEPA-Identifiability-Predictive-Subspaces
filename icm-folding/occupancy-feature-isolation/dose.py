"""Paired occupancy mass/shape doses across three explicitly specified losses."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import run as S

R,torch,np,B=S.R,S.torch,S.np,S.B
spec=importlib.util.spec_from_file_location('feature_raw_infonce',S.HERE.parent/'raw-embedding-infonce/run.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
OUT=S.OUT/'dose'
MASSES=(.25,.5,.625,.6875,.75,.8125,.875,.90625,.9375,.96875)
METHODS=('sigreg','raw_infonce','variance_control')


def dependencies():
    return dict(**S.dependency_hashes(),**{str(p):R.digest(p.read_bytes()) for p in
        [Path(__file__),S.HERE/'DOSE_PROTOCOL.md',S.HERE.parent/'raw-embedding-infonce/run.py']})


def setup():
    ds,c,initial,ix,_=S.setup()
    u=torch.rand((4000,1024),dtype=torch.float64,generator=torch.Generator().manual_seed(8500131))
    raw=ds[131]['x'];mean=raw.mean(0);std=raw.std(0,unbiased=False).clamp_min(1e-6)
    torch.manual_seed(6100131);critic=N.critic().state_dict()
    return ds,c,initial,ix,u,mean,std,critic


def fit(near,far,alpha,method,ds,c,initial,ix,u,mean,std,critic_state):
    name=f'n{near}-f{far}-a{alpha:g}-{method}'
    path=OUT/'runs'/f'{name}.pt';x=torch.cat([ds[near]['near'],ds[far]['far']])
    ids=S.indices(alpha,len(ds[near]['near']),len(ds[far]['far']),u)
    identity=dict(near=near,far=far,alpha=alpha,method=method,initial_hash=R.state_hash(initial),
        alignment_plan_hash=R.digest(ix.numpy().tobytes()),uniform_plan_hash=R.digest(u.numpy().tobytes()),
        endpoint_index_hash=R.digest(ids.numpy().tobytes()),critic_initial_hash=R.state_hash(critic_state),
        source_hashes={s:ds[s]['source_hash'] for s in set([near,far,131])},dependencies=dependencies())
    if path.exists():
        assert R.trials.load(path)['identity']==identity
        return
    m=B.Encoder();m.load_state_dict(initial);critic=N.critic();critic.load_state_dict(critic_state)
    parameters=list(m.parameters())+(list(critic.parameters()) if method=='raw_infonce' else [])
    opt=torch.optim.Adam(parameters,lr=.001);reg=B.SIGReg1D();data=ds[131]['data']
    frames=[dict(step=0,**R.geometry(m,c,1025))];start=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        raw=x[ids[step,:256] if method=='raw_infonce' else ids[step]]
        code=m(raw)
        if method=='raw_infonce':loss=N.objective((raw-mean)/std,code,critic)
        else:
            ya=m(torch.cat([data['x'][ix[step]],data['xp'][ix[step]]]))
            alignment=(ya[:512]-ya[512:]).square().mean()
            if method=='sigreg':loss=.9*alignment+.1*reg(code)
            else:
                variance=.5*(torch.relu(1-torch.sqrt(code[:512].var()+1e-4))+
                              torch.relu(1-torch.sqrt(code[512:].var()+1e-4)))
                loss=25*alignment+25*variance
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,loss=float(loss.detach()),**R.geometry(m,c,1025)))
    d=1/x[:,1];lo,hi=float(d.min()),float(d.max())
    saved=dict(identity=identity,model=m.state_dict(),initial=initial,optimizer=opt.state_dict(),
        critic=critic.state_dict() if method=='raw_infonce' else None,critic_initial=critic_state,
        raw_mean=mean,raw_std=std,config=vars(c),frames=frames,seconds=time.monotonic()-start,
        target_support=[lo,hi],realized_near_mass=float((u<alpha).double().mean()))
    R.trials.save_torch(path,saved)
    check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001)
    saved['geometry']['roots_in_source_support']=[d for d in saved['geometry']['roots'] if lo<=d<=hi]
    R.trials.save_torch(path,saved)
    R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,**saved['geometry']))
    print(json.dumps(dict(name=name,**saved['geometry'],seconds=saved['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);args=setup()
    jobs=[(n,f,alpha,method) for alpha in MASSES for n in (130,136) for f in (130,136) for method in METHODS]
    for k,job in enumerate(jobs):
        if k%a.shards==a.shard:fit(*job,*args)


if __name__=='__main__':main()

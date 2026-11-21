"""Joint-transition dose bridge for the canonical four scalar objectives."""
import argparse
import copy
import json
from pathlib import Path
import time
import fresh_fit as F

D,S,R,torch,np,B=F.D,F.S,F.R,F.torch,F.D.np,F.B
OUT=S.OUT/'joint'
METHODS=('sigreg','infonce','raw_infonce','vicreg')


def pair_indices(weights,counts,u):
    ids=torch.empty_like(u,dtype=torch.long);lower=0.;offset=0
    for w,n in zip(weights,counts):
        if w>0:
            assert n>0
            mask=(u>=lower)&(u<lower+w if offset+n<sum(counts) else u<=1)
            ids[mask]=offset+((u[mask]-lower)/w*n).long().clamp(0,n-1)
        lower+=w;offset+=n
    assert abs(lower-1)<1e-12
    return ids


def prepare(s,alpha,u):
    data=s['data'];bn=(1/data['x'][:,1])<2;bp=(1/data['xp'][:,1])<2
    masks=[bn&bp,bn!=bp,~bn&~bp];groups=[torch.nonzero(m).flatten() for m in masks]
    counts=[len(g) for g in groups];gamma=counts[1]/sum(counts)
    native=(counts[0]+counts[1]/2)/sum(counts)
    weights=[alpha-gamma/2,gamma,1-alpha-gamma/2]
    assert min(weights)>=0
    table=torch.cat(groups);ids=pair_indices(weights,counts,u)
    return {k:data[k][table] for k in ['x','xp']},ids,dict(gamma=gamma,native_mass=native,
        counts=counts,weights=weights,near_shape_tv_bound=gamma/2*abs(1/alpha-1/native),
        far_shape_tv_bound=gamma/2*abs(1/(1-alpha)-1/(1-native)))


def dependencies():
    return dict(**D.dependencies(),**{str(p):R.digest(p.read_bytes()) for p in [Path(__file__),S.HERE/'JOINT_PROTOCOL.md',S.HERE/'fresh_fit.py']})


def fit(seed,alpha,method,ds,c,initial,mean,std,critic_state,u):
    name=f'{seed}-a{alpha:g}-{method}';path=OUT/'runs'/f'{name}.pt'
    data,ids,measure=prepare(ds[seed],alpha,u)
    identity=dict(source=seed,alpha=alpha,method=method,source_hash=ds[seed]['source_hash'],
        initial_hash=R.state_hash(initial),critic_initial_hash=R.state_hash(critic_state),
        plan_hash=R.digest(ids.numpy().tobytes()),uniform_hash=R.digest(u.numpy().tobytes()),dependencies=dependencies())
    if path.exists():assert R.trials.load(path)['identity']==identity;return
    model=B.Encoder();model.load_state_dict(initial);critic=D.N.critic();critic.load_state_dict(critic_state)
    opt=torch.optim.Adam(list(model.parameters())+(list(critic.parameters()) if method=='raw_infonce' else []),lr=.001)
    reg=B.SIGReg1D();frames=[dict(step=0,**R.geometry(model,c,1025))];start=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        ix=ids[step];raw=torch.cat([data['x'][ix],data['xp'][ix]])
        y=model(raw);ya,yp=y[:512],y[512:]
        if method=='raw_infonce':
            rr=torch.cat([raw[:128],raw[512:640]])
            zz=torch.cat([ya[:128],yp[:128]])
            loss=D.N.objective((rr-mean)/std,zz,critic)
        else:loss=D.N.base.base_loss(ya,yp,method,reg)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,loss=float(loss.detach()),**R.geometry(model,c,1025)))
    d=1/torch.cat([data['x'],data['xp']])[:,1];lo,hi=float(d.min()),float(d.max())
    saved=dict(identity=identity,initial=initial,model=model.state_dict(),optimizer=opt.state_dict(),
        critic=critic.state_dict() if method=='raw_infonce' else None,config=vars(c),raw_mean=mean,raw_std=std,
        measure=measure,frames=frames,support=[lo,hi],seconds=time.monotonic()-start)
    R.trials.save_torch(path,saved);check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001)
    saved['geometry']['roots_in_source_support']=[d for d in saved['geometry']['roots'] if lo<=d<=hi]
    R.trials.save_torch(path,saved)
    R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,measure=measure,**saved['geometry']))
    print(json.dumps(dict(name=name,**saved['geometry'],seconds=saved['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,initial,_,_,mean,std,critic=F.setup()
    u=torch.rand((4000,512),dtype=torch.float64,generator=torch.Generator().manual_seed(9000131))
    if a.validate:
        records=[]
        for seed in range(9100,9110):
            n=len(ds[seed]['data']['x']);mids=(torch.arange(n,dtype=torch.float64)+.5)/n
            _,native_ids,measure=prepare(ds[seed],ds[seed]['mass'],mids)
            assert torch.equal(native_ids,torch.arange(n))
            for alpha in [ds[seed]['mass'],.5,.75,.9375]:
                _,_,measure=prepare(ds[seed],alpha,mids)
                assert abs(measure['weights'][0]+measure['weights'][1]/2-alpha)<1e-12
                records.append(dict(source=seed,alpha=alpha,**measure))
        m=B.Encoder();m.load_state_dict(initial);cr=D.N.critic();cr.load_state_dict(critic)
        calls=[];hook=m.register_forward_hook(lambda *args:calls.append(1))
        d=ds[9100]['data'];raw=torch.cat([d['x'][:512],d['xp'][:512]]);y=m(raw)
        assert len(calls)==1;hook.remove();gradient_checks={}
        for method in METHODS:
            if method=='raw_infonce':
                rr=torch.cat([raw[:128],raw[512:640]]);zz=torch.cat([y[:128],y[512:640]])
                loss=D.N.objective((rr-mean)/std,zz,cr)
            else:loss=D.N.base.base_loss(y[:512],y[512:],method,B.SIGReg1D())
            gy=torch.autograd.grad(loss,y,retain_graph=True)[0]
            assert torch.isfinite(gy).all() and gy[:512].abs().sum()>0 and gy[512:].abs().sum()>0
            gradient_checks[method]=[float(gy[:512].abs().sum()),float(gy[512:].abs().sum())]
        R.trials.atomic_json(OUT/'preflight.json',dict(passed=True,native_joint_corners_exact=True,
            target_masses_exact=True,shared_forward_calls=len(calls),both_endpoint_gradients=gradient_checks,
            dependencies=dependencies(),records=records));print('Joint-law, endpoint-mass and shared-forward gradient checks passed');return
    pre=json.loads((OUT/'preflight.json').read_text());assert pre['passed'] and pre['dependencies']==dependencies() and pre['shared_forward_calls']==1
    jobs=[(s,alpha,method) for s in range(9100,9110) for alpha in sorted(set([ds[s]['mass'],.5,.75,.9375])) for method in METHODS]
    for k,job in enumerate(jobs):
        if k%a.shards==a.shard:fit(*job,ds,c,initial,mean,std,critic,u)


if __name__=='__main__':main()

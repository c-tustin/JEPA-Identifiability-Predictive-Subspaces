"""Attached versus detached own-observation discrimination in SIGReg training."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import time
import joint as J

F,D,S,R,torch,np,B=J.F,J.D,J.S,J.R,J.torch,J.np,J.B
OUT=S.OUT/'interaction'
ARMS=((0.,False),(.1,False),(1.,False),(10.,False),(100.,False),(100.,True))


def fit(source,init,lam,detach,ds,c,initial,critic_state,u,mean,std):
    name=f's{source}-i{init}-l{lam:g}'+('-detached' if detach else '')
    path=OUT/'runs'/f'{name}.pt';data,ids,measure=J.prepare(ds[source],.75,u)
    identity=dict(source=source,initialization=init,lambda_info=lam,detached=detach,
        source_hash=ds[source]['source_hash'],initial_hash=R.state_hash(initial),
        critic_initial_hash=R.state_hash(critic_state),plan_hash=R.digest(ids.numpy().tobytes()),
        code_hash=R.digest(Path(__file__).read_bytes()),protocol_hash=R.digest((S.HERE/'INTERACTION_PROTOCOL.md').read_bytes()),
        dependencies=J.dependencies())
    if path.exists():assert R.trials.load(path)['identity']==identity;return
    model=B.Encoder();model.load_state_dict(initial);critic=D.N.critic();critic.load_state_dict(critic_state)
    opt=torch.optim.Adam(list(model.parameters())+list(critic.parameters()),lr=.001);reg=B.SIGReg1D()
    frames=[dict(step=0,**R.geometry(model,c,1025))];start=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        ix=ids[step];raw=torch.cat([data['x'][ix],data['xp'][ix]]);y=model(raw)
        align=(y[:512]-y[512:]).square().mean();penalty=reg(y)
        zz=torch.cat([y[:128],y[512:640]]);rr=torch.cat([raw[:128],raw[512:640]])
        info=D.N.objective((rr-mean)/std,zz.detach() if detach else zz,critic)
        loss=.9*align+.1*penalty+lam*info
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,alignment=float(align.detach()),
            sigreg=float(penalty.detach()),infonce=float(info.detach()),**R.geometry(model,c,1025)))
    d=1/torch.cat([data['x'],data['xp']])[:,1];lo,hi=float(d.min()),float(d.max())
    saved=dict(identity=identity,initial=initial,model=model.state_dict(),critic=critic.state_dict(),
        critic_initial=critic_state,optimizer=opt.state_dict(),frames=frames,measure=measure,config=vars(c),
        raw_mean=mean,raw_std=std,support=[lo,hi],seconds=time.monotonic()-start)
    R.trials.save_torch(path,saved);check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001)
    saved['geometry']['roots_in_source_support']=[d for d in saved['geometry']['roots'] if lo<=d<=hi]
    R.trials.save_torch(path,saved)
    R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,**saved['geometry']))
    print(name,saved['geometry'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,_,_,_,mean,std,_=F.setup()
    initial={};critics={}
    for seed in [131,132]:
        torch.manual_seed(40000+seed);initial[seed]=B.Encoder().state_dict()
        torch.manual_seed(6100000+seed);critics[seed]=D.N.critic().state_dict()
    u=torch.rand((4000,512),dtype=torch.float64,generator=torch.Generator().manual_seed(9300131))
    jobs=[(s,i,lam,detach) for s in [131,136,9109] for i in [131,132] for lam,detach in ARMS]
    for k,(s,i,lam,detach) in enumerate(jobs):
        if k%a.shards==a.shard:fit(s,i,lam,detach,ds,c,initial[i],critics[i],u,mean,std)


if __name__=='__main__':main()

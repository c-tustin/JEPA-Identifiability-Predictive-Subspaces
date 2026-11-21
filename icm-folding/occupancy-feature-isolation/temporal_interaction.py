"""Causal temporal-discrimination gradient intervention under Gaussian pressure."""
import argparse
import json
from pathlib import Path
import time
import interaction as I

F,D,S,R,torch,np,B,J=I.F,I.D,I.S,I.R,I.torch,I.np,I.B,I.J
OUT=S.OUT/'temporal-interaction'
ARMS=((0.,False),(1.,False),(10.,False),(100.,False),(100.,True))


def fit(source,init,lam,detach,ds,c,initial,u):
    name=f's{source}-i{init}-l{lam:g}'+('-detached' if detach else '')
    path=OUT/'runs'/f'{name}.pt';data,ids,measure=J.prepare(ds[source],.75,u)
    identity=dict(source=source,initialization=init,lambda_info=lam,detached=detach,
        source_hash=ds[source]['source_hash'],initial_hash=R.state_hash(initial),
        plan_hash=R.digest(ids.numpy().tobytes()),code_hash=R.digest(Path(__file__).read_bytes()),
        protocol_hash=R.digest((S.HERE/'TEMPORAL_INTERACTION_PROTOCOL.md').read_bytes()),dependencies=J.dependencies())
    if path.exists():assert R.trials.load(path)['identity']==identity;return
    m=B.Encoder();m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
    frames=[dict(step=0,**R.geometry(m,c,1025))];start=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        ix=ids[step];y=m(torch.cat([data['x'][ix],data['xp'][ix]]))
        align=(y[:512]-y[512:]).square().mean();penalty=reg(y)
        z=y.detach() if detach else y
        info=D.N.base.base_loss(z[:512],z[512:],'infonce',reg)
        loss=.9*align+.1*penalty+lam*info
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,alignment=float(align.detach()),
             sigreg=float(penalty.detach()),infonce=float(info.detach()),**R.geometry(m,c,1025)))
    d=1/torch.cat([data['x'],data['xp']])[:,1];lo,hi=float(d.min()),float(d.max())
    saved=dict(identity=identity,initial=initial,model=m.state_dict(),optimizer=opt.state_dict(),
        frames=frames,measure=measure,config=vars(c),support=[lo,hi],seconds=time.monotonic()-start)
    R.trials.save_torch(path,saved);check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001)
    saved['geometry']['roots_in_source_support']=[r for r in saved['geometry']['roots'] if lo<=r<=hi]
    if lam==0:
        ref=R.trials.load(I.OUT/'runs'/f's{source}-i{init}-l0.pt')
        error=max(float((v-ref['model'][k]).abs().max()) for k,v in saved['model'].items());assert error==0
        saved['cross_implementation_baseline_error']=error
    R.trials.save_torch(path,saved)
    R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,**saved['geometry']))
    print(name,saved['geometry'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,_,_,_,_,_,_=F.setup();initial={}
    for seed in [131,132]:
        torch.manual_seed(40000+seed);initial[seed]=B.Encoder().state_dict()
    u=torch.rand((4000,512),dtype=torch.float64,generator=torch.Generator().manual_seed(9300131))
    jobs=[(s,i,lam,dt) for s in [131,136,9109] for i in [131,132] for lam,dt in ARMS]
    for k,(s,i,lam,dt) in enumerate(jobs):
        if k%a.shards==a.shard:fit(s,i,lam,dt,ds,c,initial[i],u)


if __name__=='__main__':main()

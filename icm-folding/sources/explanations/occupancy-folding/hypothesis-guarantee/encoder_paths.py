"""Near/far Adam force decomposition and concentration-switch tests."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from guaranteed_models import old,dataset,target_measure,penalty,OUT,COMPARISON,key


def setup(level,tail):
    ds=dataset(131);c=ds['c'];data=ds['data']
    with old.encoder_gain(4):model,plan=old.trials.study.make_plan(c,131,131,'icm')
    rg=torch.Generator().manual_seed(1900000+131);rp=torch.randint(len(data['x']),plan['indices'].shape,generator=rg)
    mass=target_measure(ds,level,tail);ratio=np.zeros_like(mass);mask=ds['p']>0;ratio[mask]=mass[mask]/ds['p'][mask]
    w=torch.from_numpy(ratio.astype(np.float32))[ds['bins']]
    opt=torch.optim.Adam(model.parameters(),lr=.001);reg=old.trials.behavior.SIGReg1D()
    return ds,model,plan,rp,w,opt,reg


def profile(model,c,parameters=None,n=1025):
    q=torch.linspace(0,np.log(16),n,dtype=torch.float64,requires_grad=True)
    probe=old.ScaledEncoder(4).double()
    pars={name:p.detach().double() for name,p in (model.named_parameters() if parameters is None else parameters.items())}
    y=functional_call(probe,pars,(old.trials.behavior.sensor(q.exp(),c),))
    s=torch.autograd.grad(y.sum(),q)[0]
    from run_comparison import metrics
    return dict(q=q.detach().numpy(),y=y.detach().numpy(),slope=s.detach().numpy(),**metrics(q.detach().numpy(),y.detach().numpy(),s.detach().numpy()))


def endpoint_slopes(model,c):
    q=torch.tensor([np.log(1.2),np.log(16)],dtype=torch.float32,requires_grad=True)
    y=model(old.trials.behavior.sensor(q.exp(),c));s=torch.autograd.grad(y.sum(),q)[0]
    return s.detach().numpy()


def trace(level,tail):
    path=OUT/'paths'/f'trace-a{int(level*10000)}-{tail}.pt'
    if path.exists():return
    ds,model,plan,rp,w,opt,reg=setup(level,tail);data=ds['data'];c=ds['c']
    named=list(model.named_parameters());ps=[p for _,p in named]
    moment={term:{name:torch.zeros_like(p,dtype=torch.float64) for name,p in named} for term in ('alignment','near','far')}
    frames=[dict(step=0,**profile(model,c))];snapshots=[];starts={};events=[]
    max_grad_error=max_moment_error=max_update_error=max_gradient_relative_error=0.
    previous=endpoint_slopes(model,c)
    for step in range(4000):
        lr=old.trials.study.controls.learning_rate(step,4000);opt.param_groups[0]['lr']=lr
        ix=plan['indices'][step];ri=torch.cat([rp[step],rp[step]+len(data['x'])])
        y=model(torch.cat([data['x'][ix],data['xp'][ix]]));a=(y[:512]-y[512:]).square().mean()
        z=model(ds['endpoints'][ri]);r=penalty(z,w[ri],'weighted_moments',reg)
        ga=torch.autograd.grad(.9*a,ps,allow_unused=True,retain_graph=True)
        gy=torch.autograd.grad(.1*r,z,retain_graph=True)[0].detach()
        near=(ds['d'][ri]<2).float()
        gn=torch.autograd.grad(z,ps,grad_outputs=gy*near,allow_unused=True,retain_graph=True)
        gf=torch.autograd.grad(z,ps,grad_outputs=gy*(1-near),allow_unused=True,retain_graph=True)
        gradients={}
        for term,values in (('alignment',ga),('near',gn),('far',gf)):
            gradients[term]={name:torch.zeros_like(p) if g is None else g.detach() for (name,p),g in zip(named,values)}
        loss=.9*a+.1*r;opt.zero_grad(set_to_none=True);loss.backward()
        scale=max(1.,max(float(p.grad.abs().max()) for name,p in named))
        for name,p in named:
            total=sum(gradients[t][name] for t in gradients)
            max_grad_error=max(max_grad_error,float((total-p.grad).abs().max()))
            max_gradient_relative_error=max(max_gradient_relative_error,float((total-p.grad).abs().max())/scale)
            for term in moment:moment[term][name].mul_(.9).add_(gradients[term][name].double(),alpha=.1)
        before={name:p.detach().clone() for name,p in named}
        opt.step();current=endpoint_slopes(model,c)
        for name,p in named:
            error=(sum(moment[t][name] for t in moment)-opt.state[p]['exp_avg']).abs().max()
            max_moment_error=max(max_moment_error,float(error))
            # Reconcile float32 accumulation/reduction roundoff into the final
            # group. The actual optimizer and model updates remain untouched.
            moment['far'][name]=opt.state[p]['exp_avg'].double()-moment['alignment'][name]-moment['near'][name]
        event=bool(np.any(previous*current<0))
        if event:events.append(dict(step=step+1,before_slopes=previous.tolist(),after_slopes=current.tolist()))
        selected=step+1 in (100,200,250,500,1000,2000,4000) or (event and len(events)<=20)
        if selected:
            after={name:p.detach().clone() for name,p in named};directions={t:{} for t in moment}
            for name,p in named:
                state=opt.state[p];denom=((state['exp_avg_sq']/(1-.999**(step+1))).sqrt()+1e-8).double()
                for term in moment:directions[term][name]=-lr*moment[term][name]/(1-.9**(step+1))/denom
            error=max(float((after[name]-before[name]-sum(directions[t][name] for t in directions)).abs().max()) for name in before)
            max_update_error=max(max_update_error,error)
            probes={'before':profile(model,c,before),'actual':profile(model,c,after)}
            largest=max(float(directions[t][name].abs().max()) for t in directions for name in before)
            fraction=min(1.,1e-4/max(largest,1e-30))
            for term in directions:probes[term]=profile(model,c,{name:before[name].double()+fraction*directions[term][name] for name in before})
            probe_model=old.ScaledEncoder(4).double()
            pars={name:value.double().requires_grad_() for name,value in before.items()}
            projections={t:[] for t in directions}
            for distance in (1.2,16.):
                qq=torch.tensor([np.log(distance)],dtype=torch.float64,requires_grad=True)
                zz=functional_call(probe_model,pars,(old.trials.behavior.sensor(qq.exp(),c),))
                slope=torch.autograd.grad(zz.sum(),qq,create_graph=True)[0]
                jac=torch.autograd.grad(slope.sum(),tuple(pars.values()),allow_unused=True)
                for term in directions:projections[term].append(sum(float((g*directions[term][name]).sum()) for name,g in zip(pars,jac) if g is not None))
            snapshots.append(dict(step=step+1,event=event,probes=probes,before=before,after=after,
                directions=directions,update_error=error,component_fraction=fraction,linearized_slope_changes=projections))
        if (step+1)%100==0:frames.append(dict(step=step+1,**profile(model,c)))
        if step+1 in (100,500):starts[step+1]=dict(encoder=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
        previous=current
    reference=old.trials.load(COMPARISON/'runs'/f'{key(131,"weighted",level,tail)}.pt')
    reproduction=max(float((v-reference['encoder'][n]).abs().max()) for n,v in model.state_dict().items())
    assert reproduction==0,reproduction
    print('audit',level,tail,'gradient abs/rel',max_grad_error,max_gradient_relative_error,'moment roundoff',max_moment_error,'step error',max_update_error,flush=True)
    assert max_update_error<1e-6 and max_gradient_relative_error<1e-5
    old.trials.save_torch(path,dict(level=level,tail=tail,encoder=model.state_dict(),frames=frames,snapshots=snapshots,
        starts=starts,events=events,reproduction_max_error=reproduction,max_gradient_error=max_grad_error,
        max_moment_error=max_moment_error,max_update_error=max_update_error,max_gradient_relative_error=max_gradient_relative_error))
    print(path.stem,'events',events[:3],'final roots',frames[-1]['roots'],'exact reproduction',reproduction,flush=True)


def switch(old_level,new_level,at,reset=False):
    path=OUT/'paths'/f'switch-{int(old_level*10000)}-to-{int(new_level*10000)}-at-{at}{"-reset" if reset else ""}.pt'
    if path.exists():return
    origin=old.trials.load(OUT/'paths'/f'trace-a{int(old_level*10000)}-one.pt')
    start=origin['starts'][at]
    ds,model,plan,rp,w,opt,reg=setup(new_level,'one');c=ds['c'];data=ds['data']
    model.load_state_dict(start['encoder'])
    if not reset:opt.load_state_dict(start['optimizer'])
    initial_optimizer=copy.deepcopy(opt.state_dict())
    frames=[dict(step=at,**profile(model,c))]
    for step in range(at,4000):
        opt.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,4000)
        ix=plan['indices'][step];ri=torch.cat([rp[step],rp[step]+len(data['x'])])
        y=model(torch.cat([data['x'][ix],data['xp'][ix]]));a=(y[:512]-y[512:]).square().mean()
        r=penalty(model(ds['endpoints'][ri]),w[ri],'weighted_moments',reg)
        opt.zero_grad(set_to_none=True);(.9*a+.1*r).backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,**profile(model,c)))
    old.trials.save_torch(path,dict(old_level=old_level,new_level=new_level,at=at,reset=reset,frames=frames,
        encoder=model.state_dict(),start_encoder=start['encoder'],start_optimizer=initial_optimizer))
    print(path.stem,'final roots',frames[-1]['roots'],'R2',frames[-1]['r2'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--switches',action='store_true');p.add_argument('--reset',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.switches:
        jobs=[(a,b,at) for a,b in ((.875,.9375),(.9375,.875)) for at in (100,500)]
        if args.reset:jobs=[(a,b,500,True) for a,b in ((.875,.9375),(.9375,.875))]
        for i,job in enumerate(jobs):
            if i%args.shards==args.shard:switch(*job)
    else:
        jobs=[(a,t) for a in (.875,.9375) for t in ('one','two')]
        for i,job in enumerate(jobs):
            if i%args.shards==args.shard:trace(*job)

"""Matched gradient interventions, hard gain-8 tests, and late fold repair."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
# The older reliability package also has a module named run_study. Load the
# occupancy driver under a unique name so its imports cannot shadow that one.
spec=importlib.util.spec_from_file_location('occupancy_experiment_base',HERE.parent/'run_study.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
old,PREVIOUS,penalty,curve,roots=base.old,base.OUT,base.penalty,base.curve,base.roots
import numpy as np
import torch

OUT=HERE/'output'
FOLDED=(131,132,133,136,139)


def source(seed):
    saved=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
    c=old.trials.behavior.Config(**saved['config'])
    data=saved['data']
    endpoints=torch.cat([data['x'],data['xp']])
    bins=(old.trials.distances(endpoints).log()/np.log(16)*32).long().clamp(0,31)
    mass=torch.bincount(bins,minlength=32).float()/len(bins)
    inverse=1/mass[bins]
    return c,data,endpoints,inverse


def order_loss(model,c,sign):
    q=torch.linspace(0,np.log(16),65)
    y=model(old.trials.behavior.sensor(q.exp(),c))
    slope=sign*torch.diff(y)/(q[1]-q[0])
    return torch.relu(-slope).square().mean()


def strong_order_loss(model,c):
    """An explicit order prior, allowing either orientation and a slope margin."""
    q=torch.linspace(0,np.log(16),65)
    y=model(old.trials.behavior.sensor(q.exp(),c))
    delta=torch.diff(y)/(q[1]-q[0])
    return 100*torch.minimum(torch.relu(.1-delta).square().mean(),
                             torch.relu(.1+delta).square().mean())


def stored_order_probes(endpoints):
    q=old.trials.distances(endpoints).log()
    sorted_q,indices=q.sort()
    targets=torch.linspace(0,np.log(16),65)
    hi=torch.searchsorted(sorted_q,targets).clamp(1,len(q)-1)
    choose=torch.where((sorted_q[hi]-targets).abs()<(sorted_q[hi-1]-targets).abs(),hi,hi-1)
    choose=torch.unique(choose,sorted=True)
    return endpoints[indices[choose]],sorted_q[choose]


def stored_order_loss(model,xx,q):
    y=model(xx);delta=torch.diff(y)/torch.diff(q).clamp_min(1e-7)
    return 100*torch.minimum(torch.relu(.1-delta).square().mean(),torch.relu(.1+delta).square().mean())


def frame(model,c,sign,data,endpoints,inverse,seed,step,log,initial=False):
    q,y,s=curve(model,c)
    generator=torch.Generator().manual_seed(3100000+seed)
    a,r,w=[],[],[]
    reg=old.trials.behavior.SIGReg1D()
    with torch.no_grad():
        for _ in range(4):
            ix=torch.randint(len(data['x']),(512,),generator=generator)
            aa=model(data['x'][ix]); bb=model(data['xp'][ix])
            a.append(float((aa-bb).square().mean()))
            ri=torch.randint(len(endpoints),(1024,),generator=generator)
            yy=model(endpoints[ri])
            r.append(float(reg(yy)))
            w.append(float(penalty(yy,inverse[ri],'weighted_moments',reg)))
    total=np.sum(abs(np.diff(y)))
    backwards=float(np.maximum(-sign*np.diff(y),0).sum())
    return dict(step=step,q=q,y=y,slope=s,roots=roots(q,s),folded=bool(roots(q,s)),
        backward_variation=backwards,backward_fraction=backwards/max(float(total),1e-12),
        r2=float(np.corrcoef(q,y)[0,1]**2),grid_std=float(np.std(y)),
        alignment=float(np.mean(a)),ordinary_sigreg=float(np.mean(r)),
        corrected_sigreg=float(np.mean(w)),**log)


def fit(stage,gain,seed,method):
    target=OUT/'runs'/f'{stage}-g{gain}-{method}-{seed}.pt'
    if target.exists():return
    c,data,endpoints,inverse=source(seed)
    with old.encoder_gain(gain):
        model,plan=old.trials.study.make_plan(c,seed,seed,'icm')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    if stage=='repair':
        start=old.trials.load(PREVIOUS/'runs'/f'native-ordinary-{seed}.pt')
        model.load_state_dict(start['encoder'])
        if method!='weighted_reset':optimizer.load_state_dict(start['final_optimizer'])
    q,y,s=curve(model,c)
    sign=1 if np.cov(q,y)[0,1]>=0 else -1
    reg=old.trials.behavior.SIGReg1D()
    probe_x,probe_q=stored_order_probes(endpoints)
    rg=torch.Generator().manual_seed(1900000+seed if stage=='scratch' else 2200000+seed)
    ag=torch.Generator().manual_seed(2100000+seed)
    ri_plan=torch.randint(len(data['x']),plan['indices'].shape,generator=rg) if stage=='scratch' else None
    nsteps=4000 if stage=='scratch' else 8000
    if method=='tempered':weights=inverse.sqrt()
    elif method=='capped':weights=(inverse/inverse.mean()).clamp_max(20)
    else:weights=inverse
    ordinary=method in ('ordinary','ordinary_order')
    use_order=method.endswith('_order') or method in ('weighted_strong','weighted_stored')
    frames=[frame(model,c,sign,data,endpoints,inverse,seed,0,{})]
    states={0:copy.deepcopy(model.state_dict())}
    history=[]
    started=time.perf_counter()
    for step in range(nsteps):
        lr=old.trials.study.controls.learning_rate(step,nsteps) if stage=='scratch' else .0001
        optimizer.param_groups[0]['lr']=lr
        ix=plan['indices'][step] if stage=='scratch' else torch.randint(len(data['x']),(c.batch,),generator=ag)
        ri=ri_plan[step] if stage=='scratch' else torch.randint(len(data['x']),(c.batch,),generator=rg)
        ri=torch.cat([ri,ri+len(data['x'])])
        z=model(torch.cat([data['x'][ix],data['xp'][ix]]))
        alignment=(z[:c.batch]-z[c.batch:]).square().mean()
        zreg=model(endpoints[ri])
        regularizer=penalty(zreg,weights[ri],'ordinary' if ordinary else 'weighted_moments',reg)
        if method=='weighted_stored':ordering=stored_order_loss(model,probe_x,probe_q)
        else:ordering=(strong_order_loss(model,c) if method=='weighted_strong' else order_loss(model,c,sign)) if use_order else torch.tensor(0.)
        loss=.9*alignment+.1*regularizer+ordering
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step+1)%25==0:
            history.append(dict(step=step+1,alignment=float(alignment.detach()),
                sigreg=float(regularizer.detach()),order_penalty=float(ordering.detach()),
                objective=float(loss.detach())))
        interval=100 if stage=='scratch' else 200
        if (step+1)%interval==0:
            frames.append(frame(model,c,sign,data,endpoints,inverse,seed,step+1,{}))
            states[step+1]=copy.deepcopy(model.state_dict())
    # Dense independent geometry grid; do not infer success from loss values.
    denseq=torch.linspace(0,np.log(16),10001,requires_grad=True)
    densey=model(old.trials.behavior.sensor(denseq.exp(),c))
    denses=torch.autograd.grad(densey.sum(),denseq)[0]
    dense_roots=roots(denseq.detach().numpy(),denses.detach().numpy())
    result=dict(stage=stage,gain=gain,seed=seed,method=method,orientation=sign,
        frames=frames,states=states,history=history,encoder=model.state_dict(),
        optimizer=optimizer.state_dict(),config=vars(c),dense_roots=dense_roots,
        dense_min_oriented_slope=float((sign*denses).min()),
        seconds=time.perf_counter()-started)
    if method=='weighted_stored':result['stored_probe_q']=probe_q.numpy()
    if stage=='scratch' and gain==4 and method in ('ordinary','weighted'):
        name='ordinary' if ordinary else 'weighted_moments'
        reference=old.trials.load(PREVIOUS/'runs'/f'native-{name}-{seed}.pt')
        error=max(float((v-reference['encoder'][k]).abs().max()) for k,v in model.state_dict().items())
        assert error==0, error
        result['baseline_max_parameter_error']=error
    old.trials.save_torch(target,result)
    print(target.stem,'folded=',bool(dense_roots),'r2=',frames[-1]['r2'],
        'backward=',frames[-1]['backward_fraction'],'seconds=',result['seconds'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--strong-only',action='store_true');p.add_argument('--stored-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[('scratch',4,131,m) for m in ('ordinary','weighted')]
    jobs += [('scratch',4,s,m) for s in range(130,140) for m in ('tempered','capped','weighted_order','ordinary_order')]
    jobs += [('scratch',8,s,m) for s in range(130,140) for m in ('weighted','weighted_order')]
    jobs += [('repair',4,s,m) for s in FOLDED for m in ('ordinary','weighted','weighted_reset','weighted_order')]
    if args.strong_only:
        jobs=[('scratch',8,s,'weighted_strong') for s in range(130,140)]
        jobs += [('repair',4,s,'weighted_strong') for s in FOLDED]
    if args.stored_only:jobs=[('repair',4,s,'weighted_stored') for s in FOLDED]
    for i,job in enumerate(jobs):
        if i%args.shards==args.shard:fit(*job)


if __name__=='__main__':main()

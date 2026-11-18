"""Change endpoint occupancy via stay placement; keep alignment identical."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/shared-occupancy-ou-mpl')
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dwell_reference',HERE.parent/'icm-matched-ou/run.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
old,base,shared=reference.old,reference.base,reference.shared
sha=shared.sha
OUT=HERE/'output'
SEEDS=tuple(range(130,140))
FRACTIONS=(.5,.25)
ALLOCATIONS=('native','balanced')
EDGES=np.geomspace(1,16,33)
PROBES=(2.,4.,8.,16.)

def array_sha(a):
    if torch.is_tensor(a):a=a.numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()

def setup(seed,fraction):
    path,data,c=reference.source(seed)
    pool,d=reference.sorted_pool(data)
    bins=np.clip(np.searchsorted(EDGES,d,side='right')-1,0,31)
    counts=np.bincount(bins,minlength=32)
    probabilities={'native':np.full(len(pool),1/len(pool))}
    p=1/counts[bins].astype(np.float64)
    probabilities['balanced']=p/p.sum()
    ns=int(c.batch*fraction)
    ix=torch.randint(len(data['x']),(c.jepa_steps,ns),generator=torch.Generator().manual_seed(2500000+seed))
    u=np.random.default_rng(2600000+seed).random((c.jepa_steps,c.batch-ns))
    holds={}
    for allocation,p in probabilities.items():
        cdf=np.cumsum(p);cdf[-1]=1.
        holds[allocation]=np.searchsorted(cdf,u,side='right')
    source_bins=[]
    for key in ('x','xp'):
        dd=old.trials.distances(data[key]).numpy()
        source_bins.append(np.clip(np.searchsorted(EDGES,dd,side='right')-1,0,31))
    moving_counts=sum(np.bincount(b[ix.numpy().reshape(-1)],minlength=32) for b in source_bins)
    native_mass=counts/counts.sum()
    exposure={}
    for allocation in ALLOCATIONS:
        actual=moving_counts+2*np.bincount(bins[holds[allocation].reshape(-1)],minlength=32)
        hold_mass=np.bincount(bins,weights=probabilities[allocation],minlength=32)
        exposure[allocation]=dict(expected=(fraction*native_mass+(1-fraction)*hold_mass).tolist(),
            realized=(actual/actual.sum()).tolist(),native=native_mass.tolist())
    return dict(path=path,data=data,c=c,pool=pool,d=d,indices=ix,holds=holds,exposure=exposure,
        source_plan_sha256=array_sha(ix),hold_plan_sha256={a:array_sha(h) for a,h in holds.items()})

def batch(s,allocation,step):
    ix=s['indices'][step]
    stays=s['pool'][torch.from_numpy(s['holds'][allocation][step])]
    return torch.cat((s['data']['x'][ix],stays)),torch.cat((s['data']['xp'][ix],stays))

def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])

def shared_loss(model,x,xp,reg,nsource):
    # Encode a stay once and reuse its activation at both endpoint positions.
    # This makes the analytic cancellation exact in floating point as well.
    h=model(torch.cat((x[:nsource],xp[:nsource],x[nsource:])))
    stays=h[2*nsource:]
    y=torch.cat((h[:nsource],stays,h[nsource:2*nsource],stays))
    b=len(x)
    a=(y[:b]-y[b:]).square().mean()
    r=reg(y)
    return .9*a+.1*r,a,r,y

def probe_slopes(model,c):
    q=torch.tensor(np.log(PROBES),dtype=next(model.parameters()).dtype,requires_grad=True)
    y=model(old.trials.behavior.sensor(q.exp(),c))
    return torch.autograd.grad(y.sum(),q)[0].detach().numpy()

def fork_step(model,opt,s,step):
    records={}
    before=probe_slopes(model,s['c'])
    for allocation in ALLOCATIONS:
        m=copy.deepcopy(model)
        o=torch.optim.Adam(m.parameters(),lr=.001)
        o.load_state_dict(copy.deepcopy(opt.state_dict()))
        o.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,s['c'].jepa_steps)
        x,xp=batch(s,allocation,step)
        loss,a,r,_=shared_loss(m,x,xp,old.trials.behavior.SIGReg1D(),s['indices'].shape[1])
        parameters=list(m.parameters())
        ga=flat(torch.autograd.grad(a,parameters,retain_graph=True))
        gr=flat(torch.autograd.grad(r,parameters,retain_graph=True))
        o.zero_grad(set_to_none=True);loss.backward()
        g=flat([p.grad for p in parameters]).detach()
        state_before=flat([p.detach() for p in parameters]).clone()
        o.step()
        records[allocation]=dict(alignment=float(a.detach()),sigreg=float(r.detach()),ga=ga,gr=gr,g=g,
            update=flat([p.detach() for p in parameters])-state_before,slopes_after=probe_slopes(m,s['c']))
    a,b=(records[k] for k in ALLOCATIONS)
    alignment_error=float((a['ga']-b['ga']).abs().max())
    total_error=float(((a['g']-b['g'])-.1*(a['gr']-b['gr'])).abs().max())
    assert abs(a['alignment']-b['alignment'])<1e-7
    assert alignment_error<1e-6
    assert total_error<1e-4
    return dict(step=step,alignment_value_difference=a['alignment']-b['alignment'],
        alignment_gradient_max_error=alignment_error,gradient_channel_max_error=total_error,
        regularizer_gradient_difference_norm=float((a['gr']-b['gr']).norm()),
        adam_update_difference_norm=float((a['update']-b['update']).norm()),
        slopes_before=before.tolist(),slopes_after={k:r['slopes_after'].tolist() for k,r in records.items()})

def fit(seed,fraction,allocation):
    path=OUT/f'{seed}-f{fraction:g}-{allocation}.pt'
    if path.exists():
        r=old.trials.load(path)
        assert r['code_sha256']==sha(__file__)
        assert r['protocol_sha256']==sha(HERE/'PROTOCOL.md')
        return
    s=setup(seed,fraction);c=s['c']
    torch.manual_seed(40000+seed)
    model=old.ScaledEncoder(4)
    initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    reg=old.trials.behavior.SIGReg1D()
    history=[];forks=[]
    for step in range(c.jepa_steps):
        opt.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,c.jepa_steps)
        x,xp=batch(s,allocation,step)
        loss,a,r,_=shared_loss(model,x,xp,reg,s['indices'].shape[1])
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (step+1)%100==0:
            q,y,slope=base.curve(model,c)
            history.append(dict(step=step+1,y=y,slope=slope,roots=base.roots(q,slope),
                alignment=float(a.detach()),sigreg=float(r.detach())))
        if allocation=='native' and step+1 in (100,200,400):
            forks.append(fork_step(model,opt,s,step+1))
    old.trials.save_torch(path,dict(seed=seed,fraction=fraction,allocation=allocation,config=vars(c),
        encoder=model.state_dict(),initial=initial,history=history,forks=forks,
        exposure=s['exposure'][allocation],source_plan_sha256=s['source_plan_sha256'],
        hold_plan_sha256=s['hold_plan_sha256'][allocation],source_sha256=sha(s['path']),
        source_support=[float(s['d'].min()),float(s['d'].max())],code_sha256=sha(__file__),
        protocol_sha256=sha(HERE/'PROTOCOL.md'),shared_code_sha256=sha(shared.__file__)))
    print(seed,fraction,allocation,'final roots',history[-1]['roots'],flush=True)

def audit():
    errors=[]
    for seed in SEEDS:
        for fraction in FRACTIONS:
            s=setup(seed,fraction)
            torch.manual_seed(40000+seed)
            m=old.ScaledEncoder(4).double()
            outputs={}
            for allocation in ALLOCATIONS:
                x,xp=batch(s,allocation,0)
                calls=[];seen=[]
                handle=m.register_forward_hook(lambda *args:calls.append(1))
                reg=old.trials.behavior.SIGReg1D().double()
                def spy(y):
                    seen.append(y)
                    return reg(y)
                loss,a,r,y=shared_loss(m,x.double(),xp.double(),spy,s['indices'].shape[1])
                handle.remove()
                assert len(calls)==1 and seen[0] is y
                nsource=int(s['c'].batch*fraction);b=s['c'].batch
                assert torch.equal(y[nsource:b],y[b+nsource:])
                gy=torch.autograd.grad(a,y,retain_graph=True)[0]
                assert torch.count_nonzero(gy[nsource:b])==0
                assert torch.count_nonzero(gy[b+nsource:])==0
                ga=flat(torch.autograd.grad(a,list(m.parameters()),retain_graph=True))
                outputs[allocation]=(float(a.detach()),ga)
            va,ga=outputs['native'];vb,gb=outputs['balanced']
            error=float((ga-gb).abs().max())
            assert abs(va-vb)<1e-14 and error<1e-12
            errors.append(dict(seed=seed,fraction=fraction,alignment_value_error=abs(va-vb),alignment_gradient_max_error=error))
    old.trials.atomic_json(OUT/'audit.json',dict(cells=errors,one_forward_and_shared_tensor=True,
        stay_alignment_gradients_exactly_zero=True,alignment_function_numeric_checks=True))
    print('Same-input and invariant-alignment audits passed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.check:audit();return
    for seed in SEEDS:
        for fraction in FRACTIONS:
            for allocation in ALLOCATIONS:fit(seed,fraction,allocation)

if __name__=='__main__':main()

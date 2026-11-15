"""Same near concentration/support; change only regularizer tail structure."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'gradient-followup'))
from train_gradients import old,penalty,curve,roots,PREVIOUS
import numpy as np
from scipy.optimize import root
from scipy.special import softmax
import torch

OUT=HERE/'output';BINS=128;SPLIT=32;EPS=.02
LEVELS=(.25,.5,.875,.9375)
TAILS=('native','diffuse','one','two')


def dataset(seed):
    saved=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
    c=old.trials.behavior.Config(**saved['config']);data=saved['data']
    endpoints=torch.cat([data['x'],data['xp']])
    d=old.trials.distances(endpoints);q=d.log()
    bins=(q/np.log(16)*BINS).long().clamp(0,BINS-1)
    # Use the physical threshold consistently even at a floating-point edge.
    bins=torch.where(d<2,bins.clamp_max(SPLIT-1),bins.clamp_min(SPLIT))
    counts=torch.bincount(bins,minlength=BINS)
    p=counts.double().numpy()/len(bins);occupied=p>0
    uniform=occupied.astype(float)/occupied.sum()
    near=p.copy();near[SPLIT:]=0;near/=near.sum()
    native=p.copy();native[:SPLIT]=0;native/=native.sum()
    diffuse=occupied.astype(float);diffuse[:SPLIT]=0;diffuse/=diffuse.sum()
    centers=(np.arange(BINS)+.5)*np.log(16)/BINS
    far_ids=np.flatnonzero(occupied & (np.arange(BINS)>=SPLIT))
    z=centers[far_ids]
    qarr=q.double().numpy()
    mean=np.zeros(BINS);second=np.zeros(BINS)
    for j in np.flatnonzero(occupied):
        vals=qarr[bins.numpy()==j];mean[j]=vals.mean();second[j]=(vals**2).mean()
    nominal=np.stack([np.exp(-.5*((z-np.log(x))/.12)**2) for x in (4.,8.)])
    # Equal component weights after each is restricted to available bins.
    two_far=(nominal/nominal.sum(1,keepdims=True)).mean(0)
    target=np.array([two_far@mean[far_ids],two_far@second[far_ids]])
    offset=np.log(np.sqrt(32));xx=z-offset
    def residual(coef):
        mass=softmax(coef[0]*xx+coef[1]*xx**2)
        return np.array([mass@mean[far_ids],mass@second[far_ids]])-target
    sol=root(residual,np.array([0.,-1/(2*(.12**2+np.log(2)**2/4))]),tol=1e-11)
    if np.max(abs(residual(sol.x)))>1e-9 or sol.x[1]>=0:
        raise RuntimeError(f'Moment matching failed seed {seed}: {sol.message}, {sol.x}, {residual(sol.x)}')
    one=np.zeros(BINS);two=np.zeros(BINS)
    one[far_ids]=softmax(sol.x[0]*xx+sol.x[1]*xx**2);two[far_ids]=two_far
    tails=dict(native=native,diffuse=diffuse,one=one,two=two)
    table=torch.zeros((BINS,int(counts.max())),dtype=torch.long)
    for j in np.flatnonzero(occupied):
        ids=torch.nonzero(bins==j).flatten();table[j,:len(ids)]=ids
    return dict(c=c,data=data,endpoints=endpoints,d=d,q=q,bins=bins,counts=counts,
        p=p,uniform=uniform,near=near,tails=tails,table=table,
        mean=mean,second=second,moment_coefficients=sol.x)


def target_measure(ds,level,tail):
    u=ds['uniform'];n=ds['near'];f=ds['tails'][tail]
    a=level-EPS*u[:SPLIT].sum();b=1-level-EPS*u[SPLIT:].sum()
    assert a>0 and b>0
    return EPS*u+a*n+b*f


def jobs():
    result=[]
    for seed in range(130,140):
        result += [(seed,'native_paired',None,None),(seed,'native_floor',None,None)]
        result += [(seed,'weighted',a,t) for a in LEVELS for t in TAILS]
        result += [(seed,'replay',.875,t) for t in TAILS]
        result += [(seed,'native_replay',None,None)]
    return result


def key(seed,estimator,level,tail):
    return f'{seed}-{estimator}'+('' if level is None else f'-a{int(level*10000):04d}-{tail}')


def draw(ds,mass,n,bin_u,within_u,start,end):
    probabilities=mass[start:end].copy();probabilities/=probabilities.sum()
    cdf=torch.from_numpy(np.cumsum(probabilities));cdf[-1]=1.
    selected=torch.searchsorted(cdf,bin_u.double(),right=True).clamp_max(end-start-1)+start
    offsets=(within_u*ds['counts'][selected]).long()
    assert torch.all(offsets<ds['counts'][selected])
    return ds['table'][selected,offsets],selected


def metrics(q,y,s):
    dy=np.diff(y);up=np.maximum(dy,0).sum();down=np.maximum(-dy,0).sum()
    return dict(roots=roots(q,s),folded=bool(roots(q,s)),
        backward_fraction=float(min(up,down)/max(up+down,1e-12)),
        r2=float(np.corrcoef(q,y)[0,1]**2),std=float(y.std()))


def evaluate(model,c,step):
    q,y,s=curve(model,c)
    return dict(step=step,q=q,y=y,slope=s,**metrics(q,y,s))


def audit_distributions():
    audits=[]
    for seed in range(130,140):
        ds=dataset(seed);p=ds['p'];near_mass=float(p[:SPLIT].sum())
        rows=[]
        for a in LEVELS:
            reference=None
            for tail in TAILS:
                r=target_measure(ds,a,tail)
                assert abs(r.sum()-1)<1e-12 and abs(r[:SPLIT].sum()-a)<1e-12
                assert np.array_equal(r>0,p>0)
                assert np.all(r[p>0]>=EPS*ds['uniform'][p>0])
                if reference is None:reference=r[:SPLIT].copy()
                else:np.testing.assert_array_equal(r[:SPLIT],reference)
                rows.append(dict(level=a,tail=tail,mass_below2=float(r[:SPLIT].sum()),
                    mean_log_distance=float(r@ds['mean']),second_log_moment=float(r@ds['second']),mass=r.tolist()))
            one=target_measure(ds,a,'one');two=target_measure(ds,a,'two')
            assert abs((one-two)@ds['mean'])<1e-9
            assert abs((one-two)@ds['second'])<1e-9
        audits.append(dict(seed=seed,native_mass_below2=near_mass,
            support=[float(ds['d'].min()),float(ds['d'].max())],occupied_bins=int((p>0).sum()),
            native_mass=p.tolist(),uniform_background=(EPS*ds['uniform']).tolist(),
            moment_coefficients=ds['moment_coefficients'].tolist(),targets=rows))
    old.trials.atomic_json(OUT/'distribution-audit.json',audits)
    print('All 160 target measures pass concentration, near-profile, support, floor and moment checks.',flush=True)


def fit(seed,estimator,level,tail):
    name=key(seed,estimator,level,tail);path=OUT/'runs'/f'{name}.pt'
    if path.exists():return
    started=time.perf_counter();ds=dataset(seed);c=ds['c'];data=ds['data'];p=ds['p']
    if estimator in ('native_paired','native_replay'):target=p.copy()
    elif estimator=='native_floor':target=EPS*ds['uniform']+(1-EPS)*p
    else:target=target_measure(ds,level,tail)
    with old.encoder_gain(4):model,plan=old.trials.study.make_plan(c,seed,seed,'icm')
    rg=torch.Generator().manual_seed(1900000+seed)
    reg_plan=torch.randint(len(data['x']),plan['indices'].shape,generator=rg)
    ng=torch.Generator().manual_seed(1710000+seed)
    fg=torch.Generator().manual_seed(1810000+seed)
    point_weights=torch.from_numpy((target[p>0]/p[p>0]).astype(np.float32))
    by_bin=torch.zeros(BINS)
    by_bin[torch.from_numpy(p>0)]=point_weights
    weights=by_bin[ds['bins']]
    reg=old.trials.behavior.SIGReg1D();opt=torch.optim.Adam(model.parameters(),lr=.001)
    frames=[evaluate(model,c,0)];states={0:copy.deepcopy(model.state_dict())};history=[]
    actual_mass=np.zeros(BINS);ess=[];near_hash=hashlib.sha256()
    for step in range(c.jepa_steps):
        opt.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,c.jepa_steps)
        ix=plan['indices'][step]
        y=model(torch.cat([data['x'][ix],data['xp'][ix]]))
        alignment=(y[:c.batch]-y[c.batch:]).square().mean()
        if estimator in ('replay','native_replay'):
            alpha=target[:SPLIT].sum();nn=int(round(2*c.batch*alpha));nf=2*c.batch-nn
            ni,nb=draw(ds,target,nn,torch.rand(nn,generator=ng),torch.rand(nn,generator=ng),0,SPLIT)
            fi,fb=draw(ds,target,nf,torch.rand(nf,generator=fg),torch.rand(nf,generator=fg),SPLIT,BINS)
            ri=torch.cat([ni,fi]);used_bins=torch.cat([nb,fb])
            near_hash.update(ni.numpy().tobytes())
            actual_mass+=torch.bincount(used_bins,minlength=BINS).numpy()/(2*c.batch)
            penalty_value=reg(model(ds['endpoints'][ri]));ess.append(float(2*c.batch))
        else:
            ri=torch.cat([reg_plan[step],reg_plan[step]+len(data['x'])])
            yy=model(ds['endpoints'][ri]);w=weights[ri]
            if estimator=='native_paired':penalty_value=reg(yy)
            else:penalty_value=penalty(yy,w,'weighted_moments',reg)
            normalized=w/w.sum()
            actual_mass+=torch.bincount(ds['bins'][ri],weights=normalized.double(),minlength=BINS).numpy()
            ess.append(float(w.sum().square()/w.square().sum()))
        loss=.9*alignment+.1*penalty_value
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (step+1)%100==0:
            frames.append(evaluate(model,c,step+1))
            history.append(dict(step=step+1,alignment=float(alignment.detach()),sigreg=float(penalty_value.detach())))
        if step+1 in (100,250,500,1000,2000,4000):
            states[step+1]=copy.deepcopy(model.state_dict())
    q=torch.linspace(0,np.log(16),10001,requires_grad=True)
    y=model(old.trials.behavior.sensor(q.exp(),c));s=torch.autograd.grad(y.sum(),q)[0]
    dense=metrics(q.detach().numpy(),y.detach().numpy(),s.detach().numpy())
    lower,upper=float(ds['d'].min()),float(ds['d'].max())
    dense['in_support_roots']=[r for r in dense['roots'] if lower<=r<=upper]
    dense['extrapolation_roots']=[r for r in dense['roots'] if not lower<=r<=upper]
    result=dict(seed=seed,estimator=estimator,level=level,tail=tail,target_mass=target,
        observed_mass=actual_mass/c.jepa_steps,mean_ess=float(np.mean(ess)),min_ess=float(min(ess)),
        near_draw_sha256=near_hash.hexdigest() if estimator in ('replay','native_replay') else None,
        frames=frames,states=states,encoder=model.state_dict(),optimizer=opt.state_dict(),
        config=vars(c),history=history,dense=dense,support=[lower,upper],
        index_sha256=hashlib.sha256(plan['indices'].numpy().tobytes()+reg_plan.numpy().tobytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.perf_counter()-started)
    if estimator=='native_paired':
        reference=old.trials.load(PREVIOUS/'runs'/f'native-ordinary-{seed}.pt')
        error=max(float((v-reference['encoder'][n]).abs().max()) for n,v in model.state_dict().items())
        assert error==0,(seed,error)
        result['native_reproduction_max_error']=error
    old.trials.save_torch(path,result)
    print(name,'folded=',dense['folded'],'turns=',dense['roots'],'R2=',round(dense['r2'],5),'seconds=',round(result['seconds'],1),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--case',nargs=4)
    args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.audit_only:audit_distributions();return
    if args.case:
        s,e,a,t=args.case;fit(int(s),e,None if a=='none' else float(a),None if t=='none' else t);return
    for i,job in enumerate(jobs()):
        if i%args.shards==args.shard:fit(*job)


if __name__=='__main__':main()

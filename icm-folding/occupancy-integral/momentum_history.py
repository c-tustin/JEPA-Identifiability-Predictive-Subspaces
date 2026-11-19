"""Attribute the historical regularizer numerator and intervene at fold birth."""
import copy
import json
from pathlib import Path
import run as R
import numpy as np
import torch
from torch.func import functional_call, jvp


def main():
    torch.set_num_threads(1)
    L=R.legacy
    birth=L.trials.load(L.OUT/'birth-event.pt')
    source_path=L.ORIGINAL/'sources/icm-131.pt'
    source=L.trials.load(source_path)
    c=L.trials.behavior.Config(**source['config']);data=source['data']
    with L.encoder_gain(4):model,plan=L.trials.study.make_plan(c,131,131,'icm')
    ri=torch.randint(len(data['x']),plan['indices'].shape,generator=torch.Generator().manual_seed(1900131))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    reg=L.trials.behavior.SIGReg1D()
    names=list(dict(model.named_parameters()))
    ma={k:torch.zeros_like(p) for k,p in model.named_parameters()}
    mr=copy.deepcopy(ma)
    # Final denominator is known from the exact saved update, not recomputed
    # per component. Historical gradient states stay on their original path.
    after=L.ScaledEncoder(4);after.load_state_dict(birth['before'])
    lastopt=torch.optim.Adam(after.parameters(),lr=.001)
    lastopt.load_state_dict(copy.deepcopy(birth['optimizer_before']))
    ia,ir=plan['indices'][204],ri[204]
    yy=after(torch.cat([data['x'][ia],data['xp'][ia]]))
    aa=torch.autograd.grad(.9*(yy[:c.batch]-yy[c.batch:]).square().mean(),tuple(after.parameters()))
    rr=torch.autograd.grad(.1*reg(after(torch.cat([data['x'][ir],data['xp'][ir]]))),tuple(after.parameters()))
    for p,a,r in zip(after.parameters(),aa,rr):p.grad=a+r
    lastopt.step()
    assert all(torch.equal(p,birth['after'][k]) for k,p in after.state_dict().items())
    probe=L.ScaledEncoder(4);probe.load_state_dict(birth['before'])
    # Use the replayed combined second moment, with the saved decomposition's
    # float32 bias-corrected denominator convention.
    before_double=copy.deepcopy(probe).double()
    _,mixed=R.slope_and_mixed(before_double,c,16)
    direction={}
    for k,p in after.named_parameters():
        v=lastopt.state[p]['exp_avg_sq']
        denom=((v/(1-.999**205)).sqrt()+1e-8).double()
        direction[k]=-.001/(1-.9**205)*mixed[k]/denom
    history=[];bin_sum=np.zeros(32);max_gradient_sum_error=0.
    for step in range(205):
        ix,rx=plan['indices'][step],ri[step]
        ya=model(torch.cat([data['x'][ix],data['xp'][ix]]))
        yr=model(torch.cat([data['x'][rx],data['xp'][rx]]))
        la=.9*(ya[:c.batch]-ya[c.batch:]).square().mean()
        lr=.1*reg(yr)
        ga=torch.autograd.grad(la,tuple(model.parameters()))
        gy=torch.autograd.grad(lr,yr,retain_graph=True)[0]
        gr=torch.autograd.grad(lr,tuple(model.parameters()))
        if step==204:
            assert all(torch.equal(p,birth['before'][k]) for k,p in model.state_dict().items())
            history_before=copy.deepcopy(mr)
            ga_last,gr_last=ga,gr
        # Differentiate each historical output along one fixed final-state
        # sensitivity vector; this contracts per-sample gradients exactly.
        old=copy.deepcopy(model).double();params=dict(old.named_parameters())
        xx=torch.cat([data['x'][rx],data['xp'][rx]]).double()
        _,response=jvp(lambda p:functional_call(old,p,(xx,)),(params,),(direction,))
        terms=gy.detach().double()*response.detach()*.1*(.9**(204-step))
        contracted=sum(float((g.double()*direction[k]).sum()) for k,g in zip(names,gr))*.1*(.9**(204-step))
        max_gradient_sum_error=max(max_gradient_sum_error,abs(float(terms.sum())-contracted))
        bins=((1/xx[:,1]).log()/np.log(16)*32).long().clamp(0,31)
        by_bin=torch.bincount(bins,weights=terms,minlength=32).numpy()
        bin_sum+=by_bin
        history.append(dict(update=step+1,contribution=float(terms.sum()),by_bin=by_bin.tolist()))
        opt.zero_grad()
        for (k,p),a,r in zip(model.named_parameters(),ga,gr):
            p.grad=a+r;ma[k].mul_(.9).add_(a,alpha=.1);mr[k].mul_(.9).add_(r,alpha=.1)
        opt.step()
    assert all(torch.equal(p,birth['after'][k]) for k,p in model.state_dict().items())
    # Numerator-only optimizer-state surgery: same before weights, current
    # gradients, second moments, step counter, rate and batch.
    forks={};states={}
    for mode in ['full','remove_regularizer_history','remove_all_history','remove_alignment_history']:
        m=L.ScaledEncoder(4);m.load_state_dict(birth['before'])
        o=torch.optim.Adam(m.parameters(),lr=.001);o.load_state_dict(copy.deepcopy(birth['optimizer_before']))
        for (k,p),a,r in zip(m.named_parameters(),ga_last,gr_last):
            if mode=='remove_regularizer_history':o.state[p]['exp_avg'].sub_(history_before[k])
            if mode=='remove_all_history':o.state[p]['exp_avg'].zero_()
            if mode=='remove_alignment_history':o.state[p]['exp_avg'].copy_(history_before[k])
            p.grad=a+r
        o.step()
        # Use the common independent dense evaluator, with gain-4 architecture.
        q=torch.linspace(0,np.log(16),20001,dtype=torch.float64,requires_grad=True)
        md=copy.deepcopy(m).double();y=md(L.trials.behavior.sensor(q.exp(),c));s=torch.autograd.grad(y.sum(),q)[0].detach().numpy()
        ids=np.flatnonzero(s[:-1]*s[1:]<0);qq=q.detach().numpy()
        roots=np.exp(qq[ids]-s[ids]*np.diff(qq)[ids]/np.diff(s)[ids]).tolist()
        forks[mode]=dict(far_slope=float(s[-1]),near_slope=float(s[0]),roots=roots)
        states[mode]=m.state_dict()
        if mode=='full':assert all(torch.equal(p,birth['after'][k]) for k,p in m.state_dict().items())
    old=copy.deepcopy(probe).double()
    _,sensitivity=R.slope_and_mixed(old,c,16)
    saved_historical=R.dot(sensitivity,{k:v.double() for k,v in birth['direction_r'].items()})-history[-1]['contribution']
    historical=sum(h['contribution'] for h in history[:-1])
    assert abs(historical-saved_historical)<2e-7
    result=dict(seed=131,update=205,history=history,total_regularizer_contraction=float(bin_sum.sum()),
                historical_contraction=historical,current_contraction=history[-1]['contribution'],
                saved_historical_contraction=saved_historical,bin_contributions=bin_sum.tolist(),
                sample_vs_parameter_contraction_max_error=max_gradient_sum_error,
                numerator_surgery=forks,exact_replay=True,
                provenance={str(p):R.hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_path,L.OUT/'birth-event.pt',Path(__file__)]},
                scope='Historical contractions use final shared denominator and before-birth slope sensitivity. Numerator surgery is one actual Adam step with fixed second moments; no claim of permanent prevention or universal cause.')
    out=R.HERE/'output';L.trials.atomic_json(out/'momentum_history.json',result)
    L.trials.save_torch(out/'momentum_history_forks.pt',dict(forks=states,config=vars(c)))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    axes[0].bar(np.arange(1,206),[h['contribution'] for h in history]);axes[0].set(xlabel='Past update',ylabel='Contribution to slope change at update 205',title='SIGReg numerator history through final denominator')
    axes[1].bar(np.exp((np.arange(32)+.5)*np.log(16)/32),bin_sum,width=.9*np.diff(np.exp(np.arange(33)*np.log(16)/32)));axes[1].set(xscale='log',xlabel='Source distance (log bins)',title='Signed historical + current contributions')
    axes[2].bar(['Full','No SIGReg\nhistory','No first-moment\nhistory','No alignment\nhistory'],[v['far_slope'] for v in forks.values()]);axes[2].set(ylabel='Actual post-step slope at d=16',title='One-step Adam numerator interventions')
    axes[2].tick_params(axis='x',labelsize=8)
    for ax in axes:ax.axhline(0,color='black',lw=.6)
    fig.tight_layout();fig.savefig(out/'momentum_history.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['history','provenance','bin_contributions']},indent=2))


if __name__=='__main__':main()

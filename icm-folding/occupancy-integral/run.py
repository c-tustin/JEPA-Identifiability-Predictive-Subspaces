"""Evaluate the empirical occupancy/slope integral at seed 131's fold birth.

No training sweep: replay one saved update and differentiate its before-state.
The original study remains canonical; all new outputs live beside this script.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/occupancy-integral-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/occupancy-integral-cache')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MECH = HERE.parent/'sources/predictive-closure-workspace/behavior-policy-experiments/qualitative-study/fold-location/training-mechanism'
sys.path.insert(0, str(MECH))
import run_mechanism as legacy
import numpy as np
import torch
from torch.func import functional_call, jvp


def slope_and_mixed(model, config, distance):
    q = torch.tensor([math.log(distance)], dtype=torch.float64, requires_grad=True)
    y = model(legacy.trials.behavior.sensor(q.exp(), config))
    slope = torch.autograd.grad(y.sum(), q, create_graph=True)[0]
    named = dict(model.named_parameters())
    raw = torch.autograd.grad(slope.sum(), tuple(named.values()), allow_unused=True)
    return float(slope.detach()), {
        k: torch.zeros_like(p) if a is None else a.detach()
        for (k, p), a in zip(named.items(), raw)
    }


def dot(a, b):
    return sum(float((a[k]*b[k]).sum()) for k in a)


def force(y, alpha, reg, batch_scale=1024):
    phase = y[:, None]*reg.frequency
    cosine = (alpha[:, None]*phase.cos()).sum(0)
    sine = (alpha[:, None]*phase.sin()).sum(0)
    g = 2*batch_scale*((reg.weight*reg.frequency)*(
        -(cosine-reg.target)*phase.sin()+sine*phase.cos())).sum(1)
    return g, cosine, sine


def objective(model, x, alpha, reg):
    # Fixed scale 1024: the historical regularizer's endpoint batch size.
    y = model(x)
    phase = y[:, None]*reg.frequency
    c = (alpha[:, None]*phase.cos()).sum(0)
    s = (alpha[:, None]*phase.sin()).sum(0)
    loss = 1024*(reg.weight*((c-reg.target).square()+s.square())).sum()
    raw = torch.autograd.grad(loss, tuple(model.parameters()))
    return float(loss.detach()), {k:g.detach() for k,g in zip(dict(model.named_parameters()),raw)}


def main():
    torch.set_num_threads(1)
    out = HERE/'output'
    out.mkdir(exist_ok=True)
    birth_path = MECH/'output/birth-event.pt'
    source_path = legacy.ORIGINAL/'sources/icm-131.pt'
    birth = legacy.trials.load(birth_path)
    saved = legacy.trials.load(source_path)
    config = legacy.trials.behavior.Config(**saved['config'])
    data = saved['data']
    model = legacy.ScaledEncoder(4).double()
    model.load_state_dict({k:v.double() for k,v in birth['before'].items()})
    reg = legacy.trials.behavior.SIGReg1D().double()
    x = torch.cat([data['x'],data['xp']]).double()
    qdata = (1/x[:,1]).log()
    bins = (qdata/math.log(16)*32).long().clamp(0,31)
    counts = torch.bincount(bins, minlength=32)
    native = torch.ones(len(x),dtype=torch.float64)/len(x)
    balanced = 1/counts[bins].double()
    balanced /= balanced.sum()
    y = model(x).detach()
    gn, cn, sn = force(y,native,reg)
    gb, cb, sb = force(y,balanced,reg)
    before_slope, mixed = slope_and_mixed(model,config,16)
    parameters = dict(model.named_parameters())
    # d_q K(q,s) = J_theta h(s) dot grad_theta h'(q).
    kernel = []
    for chunk in x.split(2048):
        _, derivative = jvp(lambda p:functional_call(model,p,(chunk,)),
                            (parameters,), (mixed,))
        kernel.append(derivative.detach())
    kernel = torch.cat(kernel)
    cases = {}
    gradients = {}
    for name,alpha,g in [('native',native,gn),('equal_log_bins',balanced,gb)]:
        terms = alpha*g*kernel
        loss, grad = objective(model,x,alpha,reg)
        integral = float(terms.sum())
        contraction = dot(mixed,grad)
        assert math.isclose(integral,contraction,rel_tol=1e-10,abs_tol=1e-10)
        cases[name] = dict(integral=integral,autograd_contraction=contraction,
            absolute_check_error=abs(integral-contraction),regularizer_loss=loss,
            predicted_regularizer_sgd_slope_change=-.001*.1*integral,
            slope_after_regularizer_sgd_linearization=before_slope-.001*.1*integral,
            occupancy_mass_by_bin=torch.bincount(bins,weights=alpha,minlength=32).tolist(),
            integral_by_bin=torch.bincount(bins,weights=terms,minlength=32).tolist(),
            cosine= (cn if name=='native' else cb).tolist(),
            sine=(sn if name=='native' else sb).tolist())
        stepped = copy.deepcopy(model)
        with torch.no_grad():
            for k,p in stepped.named_parameters():
                p.add_(grad[k],alpha=-.001*.1)
        cases[name]['finite_regularizer_sgd_slope_at_16'] = slope_and_mixed(stepped,config,16)[0]
        gradients[name]=grad
    # These hybrid contractions separate outer sampling and inner moments.
    hybrids = dict(
        native_moments_balanced_outer=float((balanced*gn*kernel).sum()),
        balanced_moments_native_outer=float((native*gb*kernel).sum()))

    # Recover exactly the historical minibatches and perform its real Adam step.
    with legacy.encoder_gain(4):
        _,plan = legacy.trials.study.make_plan(config,131,131,'icm')
    ix = plan['indices'][204]
    rix = torch.randint(len(data['x']),plan['indices'].shape,
        generator=torch.Generator().manual_seed(1900000+131))[204]
    xm = torch.cat([data['x'][ix],data['xp'][ix]])
    xr = torch.cat([data['x'][rix],data['xp'][rix]])
    replay = legacy.ScaledEncoder(4)
    replay.load_state_dict(birth['before'])
    opt = torch.optim.Adam(replay.parameters(),lr=.001)
    opt.load_state_dict(copy.deepcopy(birth['optimizer_before']))
    yy = replay(xm)
    align = (yy[:len(ix)]-yy[len(ix):]).square().mean()
    rr = legacy.trials.behavior.SIGReg1D()(replay(xr))
    ga = torch.autograd.grad(.9*align,tuple(replay.parameters()))
    gr = torch.autograd.grad(.1*rr,tuple(replay.parameters()))
    opt.zero_grad(set_to_none=True)
    for p,a,r in zip(replay.parameters(),ga,gr):
        p.grad = a+r
    opt.step()
    replay_error = max(float((p-birth['after'][k]).abs().max())
                       for k,p in replay.state_dict().items())
    assert replay_error == 0, replay_error
    # Double precision formula checks for the actual minibatch integral.
    batch_x = xr.double()
    alpha = torch.ones(len(batch_x),dtype=torch.float64)/len(batch_x)
    batch_y = model(batch_x).detach()
    batch_g,_,_ = force(batch_y,alpha,reg)
    _,batch_k = jvp(lambda p:functional_call(model,p,(batch_x,)),
                   (parameters,), (mixed,))
    _,batch_grad = objective(model,batch_x,alpha,reg)
    batch_integral = float((alpha*batch_g*batch_k.detach()).sum())
    assert math.isclose(batch_integral,dot(mixed,batch_grad),rel_tol=1e-10,abs_tol=1e-10)
    # Isolate current regularizer gradient within Adam's shared denominator.
    preconditioned_mixed = {}
    current_direction = {}
    for (name,p),r in zip(replay.named_parameters(),gr):
        state = opt.state[p]
        step = int(state['step'])
        denom = ((state['exp_avg_sq']/(1-.999**step)).sqrt()+1e-8).double()
        coefficient = -.001*(1-.9)/(1-.9**step)*.1
        preconditioned_mixed[name] = coefficient*mixed[name]/denom
        current_direction[name] = (coefficient/.1)*r.double()/denom
    _,adam_k = jvp(lambda p:functional_call(model,p,(batch_x,)),
                  (parameters,), (preconditioned_mixed,))
    current_terms = alpha*batch_g*adam_k.detach()
    current_analytic = float(current_terms.sum())
    current_replayed = dot(mixed,current_direction)
    assert math.isclose(current_analytic,current_replayed,rel_tol=2e-4,abs_tol=2e-7)
    da = {k:v.double() for k,v in birth['direction_a'].items()}
    dr = {k:v.double() for k,v in birth['direction_r'].items()}
    update = {k:birth['after'][k].double()-birth['before'][k].double() for k in parameters}
    after = copy.deepcopy(model)
    after.load_state_dict({k:v.double() for k,v in birth['after'].items()})
    after_slope,_ = slope_and_mixed(after,config,16)
    # Central finite difference independently checks the mixed-derivative contraction.
    fd = []
    for sign in [-1,1]:
        probe = copy.deepcopy(model)
        with torch.no_grad():
            for k,p in probe.named_parameters():
                p.add_(update[k],alpha=sign*.001)
        fd.append(slope_and_mixed(probe,config,16)[0])
    finite_difference = (fd[1]-fd[0])/.002
    assert math.isclose(finite_difference,dot(mixed,update),rel_tol=1e-6,abs_tol=1e-10)
    batch_bins = ((1/batch_x[:,1]).log()/math.log(16)*32).long().clamp(0,31)
    actual = dict(before_slope=before_slope,after_slope=after_slope,
        measured_slope_change=after_slope-before_slope,
        linear_actual_update=dot(mixed,update),
        central_finite_difference_actual_direction=finite_difference,
        linearization_absolute_error=abs(after_slope-before_slope-dot(mixed,update)),
        saved_component_update_max_error=max(float((da[k]+dr[k]-update[k]).abs().max()) for k in update),
        linear_alignment_adam_component=dot(mixed,da),
        linear_regularizer_adam_component=dot(mixed,dr),
        linear_current_regularizer_adam=current_analytic,
        linear_current_regularizer_float32_check=current_replayed,
        linear_historical_regularizer_adam=dot(mixed,dr)-current_analytic,
        batch_integral=batch_integral,
        batch_predicted_regularizer_sgd_change=-.001*.1*batch_integral,
        current_regularizer_adam_by_bin=torch.bincount(batch_bins,weights=current_terms,minlength=32).tolist(),
        replay_parameter_max_error=replay_error)
    # Direction profiles use mixed derivatives at fixed before-state weights.
    distances=np.geomspace(1,16,65)
    profile=[]
    for d in distances:
        s,a = slope_and_mixed(model,config,float(d))
        profile.append(dict(distance=float(d),before_slope=s,
            native_integral=dot(a,gradients['native']),
            balanced_integral=dot(a,gradients['equal_log_bins']),
            predicted_actual_adam_slope_change=dot(a,update)))
    result = dict(seed=131,gain=4,update='204 -> 205',query_distance=16,
        endpoint_observations=len(x),frequency_knots=17,batch_scale=1024,
        observed_distance_min=float(qdata.exp().min()),observed_distance_max=float(qdata.exp().max()),
        occupied_bins=int((counts>0).sum()),empty_bins=torch.where(counts==0)[0].tolist(),
        regularizer_coefficient=.1,sgd_comparison_lr=.001,bins=32,
        empirical_occupancy=cases,hybrid_integrals=hybrids,actual_update=actual,
        profile=profile,
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [birth_path,source_path,Path(__file__)]},
        scope=['Full-data empirical measure, not an exact unknown population density.',
               'Equal occupied log bins at fixed before-state weights; privileged geometry.',
               'Plug-in full-data moments differ from the expectation of finite-batch moments squared.',
               'SGD linearization is not an Adam prediction or a finite-step fold certificate.',
               'Adam attribution retains the actual shared denominator; historical term aggregates past regularizer gradients.',
               'One inspected source and update; no universal occupancy-only threshold.'])
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    centers=np.exp((np.arange(32)+.5)*math.log(16)/32)
    for name,label in [('native','Native occupancy'),('equal_log_bins','Equal occupied log-bin mass')]:
        axes[0].plot(centers,cases[name]['occupancy_mass_by_bin'],label=label)
        axes[1].plot(centers,-.0001*np.array(cases[name]['integral_by_bin']),label=label)
    axes[0].set(ylabel='Probability per log-distance bin',title='Same observations, different weights')
    axes[1].set(ylabel='Contribution to slope change at d=16',title='Regularizer-only SGD linearization')
    values=[actual['linear_alignment_adam_component'],actual['linear_current_regularizer_adam'],
            actual['linear_historical_regularizer_adam'],actual['measured_slope_change']]
    axes[2].bar(['Alignment\n(history included)','Current\nSIGReg','Historical\nSIGReg','Measured\nfull update'],values)
    axes[2].axhline(-before_slope,color='r',ls='--',label='Change needed to cross zero')
    axes[2].set(ylabel='Slope change at d=16',title='Actual Adam update at fold birth')
    for ax in axes[:2]:
        ax.set_xscale('log');ax.set_xlabel('Source observation distance');ax.legend(fontsize=8)
    axes[1].axhline(0,color='k',lw=.5)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'integral.png',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(empirical_occupancy={k:{a:v for a,v in x.items() if a in ('integral','predicted_regularizer_sgd_slope_change','absolute_check_error')} for k,x in cases.items()},actual_update=actual,hybrid_integrals=hybrids),indent=2))


if __name__=='__main__':
    main()

"""Concrete graphs, independent geometry checks, and experiment summaries."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_gradients import OUT,HERE,PREVIOUS,old,source,curve,roots,order_loss,stored_order_probes,FOLDED
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RED='#c13c45';TEAL='#147b79';ORANGE='#d48a18';BLUE='#345e91';PURPLE='#8758a3'
LABEL={'ordinary':'Original SIGReg','weighted':'Balance each distance bin',
 'tempered':'Gentler correction (square root)','capped':'Full correction, cap extreme weights',
 'weighted_order':'Balance bins + discourage doubling back',
 'ordinary_order':'Original SIGReg + discourage doubling back',
 'weighted_reset':'Balance bins + reset Adam',
 'weighted_strong':'Balance bins + strong order prior',
 'weighted_stored':'Strong order prior on stored observations'}


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def geometry(q,y,s):
    # Sign is arbitrary: use the smaller of upward and downward variation.
    increments=np.diff(y)
    up=float(np.maximum(increments,0).sum());down=float(np.maximum(-increments,0).sum())
    return dict(roots=roots(q,s),folded=bool(roots(q,s)),
        backward_fraction=min(up,down)/max(up+down,1e-12),
        r2=float(np.corrcoef(q,y)[0,1]**2),grid_std=float(np.std(y)))


def normalize(q,y):
    sign=1 if np.cov(q,y)[0,1]>=0 else -1
    return sign*(y-y.mean())/max(float(y.std()),1e-12)


def dense_turns(model,c):
    q=torch.linspace(0,np.log(16),10001,requires_grad=True)
    y=model(old.trials.behavior.sensor(q.exp(),c))
    s=torch.autograd.grad(y.sum(),q)[0]
    return roots(q.detach().numpy(),s.detach().numpy())


def overview(runs):
    rows=[]
    for r in runs:
        f=r['frames'][-1]
        g=geometry(f['q'],f['y'],f['slope'])
        g['roots']=r['dense_roots'];g['folded']=bool(r['dense_roots'])
        rows.append(dict(stage=r['stage'],gain=r['gain'],method=r['method'],seed=r['seed'],
            **g,dense_roots=r['dense_roots']))
    # Retain native ordinary/full-weight comparisons, using existing models.
    for seed in range(130,140):
        for method,oldname in (('ordinary','ordinary'),('weighted','weighted_moments')):
            if any(r['stage']=='scratch' and r['gain']==4 and r['seed']==seed and r['method']==method for r in rows):continue
            a=old.trials.load(PREVIOUS/'runs'/f'native-{oldname}-{seed}.pt')
            f=a['frames'][-1]
            c,_,_,_=source(seed);model=old.ScaledEncoder(4);model.load_state_dict(a['encoder'])
            dense=dense_turns(model,c)
            g=geometry(a['q'],f['y'],f['slope']);g['roots']=dense;g['folded']=bool(dense)
            rows.append(dict(stage='scratch',gain=4,method=method,seed=seed,
                **g,dense_roots=dense))
        # Hard gain-8 originals are established source-model controls.
        p=old.HERE.parent/'output/gain-8/encoders'/f'icm-{seed}.pt'
        if not p.exists():raise RuntimeError(f'Missing gain-8 baseline {p}')
        a=old.trials.load(p);c,_,_,_=source(seed)
        model=old.ScaledEncoder(8);model.load_state_dict(a['encoder'])
        q,y,s=curve(model,c)
        dense=dense_turns(model,c);g=geometry(q,y,s);g['roots']=dense;g['folded']=bool(dense)
        rows.append(dict(stage='scratch',gain=8,method='ordinary',seed=seed,
            **g,dense_roots=dense))
    summary=[]
    for stage,gain in (('scratch',4),('scratch',8),('repair',4)):
        for method in LABEL:
            group=[r for r in rows if r['stage']==stage and r['gain']==gain and r['method']==method]
            if group:summary.append(dict(stage=stage,gain=gain,method=method,n=len(group),
                folded=sum(bool(r['roots']) for r in group),mean_r2=float(np.mean([r['r2'] for r in group])),
                mean_backward_fraction=float(np.mean([r['backward_fraction'] for r in group])),
                minimum_grid_std=min(r['grid_std'] for r in group)))
    return rows,summary


def weighting_picture():
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    bars=[([900,100],'Collected observations','Observation count'),
          ([.9,.1],'Ordinary SIGReg','Share of averaging weight'),
          ([.5,.5],'Inverse-occupancy weighting','Share of averaging weight')]
    for ax,(values,title,ylabel) in zip(axes,bars):
        ax.bar(['Crowded range','Rare range'],values,color=[ORANGE,BLUE],width=.6)
        ax.set(title=title,ylabel=ylabel)
        for i,v in enumerate(values):ax.text(i,v,str(int(v)) if v>1 else f'{v:.0%}',ha='center',va='bottom',fontsize=12)
        ax.set_ylim(0,max(values)*1.23)
    fig.suptitle('Give the two ranges equal total influence, despite unequal observation counts',fontsize=13)
    fig.text(.5,-.01,'Each crowded-range observation gets 1/9 the weight of each rare-range observation.\nThe sine/cosine averages are then computed using those weights.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.9));save(fig,'01_what_weighting_means')


def ambiguity_picture():
    raw=old.trials.load(PREVIOUS/'runs/native-ordinary-131.pt')
    fixed=old.trials.load(PREVIOUS/'runs/native-weighted_moments-131.pt')
    q=raw['q'];d=np.exp(q)
    rr=normalize(q,raw['frames'][-1]['y']);ff=normalize(q,fixed['frames'][-1]['y'])
    turn=raw['metrics']['roots'][0];left=1.5
    value=np.interp(left,d,rr);ix=np.flatnonzero(d>turn)
    right=float(np.interp(value,rr[ix],d[ix]))
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
    for ax,y,color,title in zip(axes,(rr,ff),(RED,TEAL),('Original: two distances share a code','Balanced influence: the codes stay distinct')):
        ax.plot(d,y,c=color,lw=3)
        yy=np.interp([left,right],d,y)
        ax.scatter([left,right],yy,color=color,s=65,zorder=4)
        for xx,v in zip((left,right),yy):
            ax.axvline(xx,color='gray',ls=':',alpha=.5)
            ax.annotate(f'd = {xx:.2f}',(xx,v),xytext=(8,14),textcoords='offset points',fontsize=10)
        ax.set(xlim=(1,7),xlabel='Actual distance to object',ylabel='Learned code (display scale)',title=title)
    axes[0].axhline(value,c=RED,ls='--',alpha=.6)
    fig.suptitle('What a fold loses · same ICM observations, same starting weights, seed 131',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'02_fold_makes_distances_ambiguous')
    return dict(seed=131,left_distance=left,right_distance=right,shared_display_code=float(value))


def inward_picture():
    run=old.trials.load(PREVIOUS/'runs/native-ordinary-131.pt')
    q=run['q'];d=np.exp(q);final=run['frames'][-1]['y']
    sign=1 if np.cov(q,final)[0,1]>=0 else -1
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),sharex=True,sharey=True)
    for ax,step in zip(axes,(250,500,1000,4000)):
        f=next(f for f in run['frames'] if f['step']==step)
        y=sign*(f['y']-final.mean())/final.std()
        root=f['roots'][0]
        ax.plot(d,y,c=RED,lw=2.5)
        ax.scatter([root],[np.interp(root,d,y)],c=RED,s=55,zorder=4)
        ax.axvline(root,c=RED,ls=':',alpha=.5)
        ax.set(title=f'Update {step}\nTurn at distance {root:.2f}',xlabel='Actual distance',xlim=(1,16))
    axes[0].set_ylabel('Learned code (one fixed scale)')
    fig.suptitle('The fold moves inward as the curve bends · seed 131',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.88));save(fig,'03_fold_moves_inward')


def concrete_repair_picture():
    raw=old.trials.load(PREVIOUS/'runs/native-ordinary-131.pt')
    repaired=old.trials.load(OUT/'runs/repair-g4-weighted_stored-131.pt')
    q=raw['q'];d=np.exp(q);yy=normalize(q,raw['frames'][-1]['y'])
    root=raw['metrics']['roots'][0];left=1.5;level=np.interp(left,d,yy)
    ix=np.flatnonzero(d>root);right=float(np.interp(level,yy[ix],d[ix]))
    end=repaired['frames'][-1];fixed=normalize(end['q'],end['y'])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,y,color,title in zip(axes,(yy,fixed),(RED,TEAL),
        ('Before repair: two distances share a code','After repair: the two codes are distinct')):
        ax.plot(d,y,c=color,lw=3)
        ys=np.interp([left,right],d,y)
        ax.scatter([left,right],ys,c=color,s=65,zorder=4)
        for x,v in zip((left,right),ys):
            ax.axvline(x,c='gray',ls=':',alpha=.5)
            ax.annotate(f'd = {x:.2f}',(x,v),xytext=(8,14),textcoords='offset points')
        ax.set(xlim=(1,7),xlabel='Actual distance to object',ylabel='Learned code (display scale)',title=title)
    axes[0].axhline(level,c=RED,ls='--',alpha=.5)
    fig.suptitle('Repair an existing fold with an order-based gradient · seed 131',fontsize=14)
    fig.text(.5,-.02,'Same stored observations; 8,000 additional updates. The added gradient uses known distance ordering.\nEach panel rescales its code for display; only ordering and equality within a panel are compared.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.89));save(fig,'12_before_and_after_actual_repair')


def counts_picture(summary):
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,stage,gain,title in zip(axes,('scratch','scratch','repair'),(4,8,4),
        ('Prevent folds: usual input scale','Prevent folds: harder input scale','Repair an existing fold')):
        group=[s for s in summary if s['stage']==stage and s['gain']==gain]
        display={'ordinary':'Original','weighted':'Balance bins','tempered':'Gentler weights','capped':'Capped weights',
           'weighted_order':'Balance + order penalty','ordinary_order':'Original + order penalty','weighted_reset':'Balance + reset Adam',
           'weighted_strong':'Balance + strong order prior','weighted_stored':'Strong prior, stored observations'}
        ax.barh([display[s['method']] for s in group],[s['folded']/s['n'] for s in group],
            color=[RED if s['method']=='ordinary' else TEAL if s['method'] in ('weighted','capped') else BLUE for s in group])
        for i,s in enumerate(group):ax.text(s['folded']/s['n']+.02,i,f"{s['folded']}/{s['n']}",va='center')
        ax.set(xlim=(0,1.22),xticks=(0,.5,1),xticklabels=('0%','50%','100%'),title=title,xlabel='Runs with a final fold')
        ax.invert_yaxis()
    fig.suptitle('Gradient interventions: count every run, including failures',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'04_gradient_results')


def run_curves(runs):
    selected=[r for r in runs if r['stage']=='scratch' and r['gain']==4 and r['seed']==131
              and r['method'] in ('ordinary','weighted','weighted_order')]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    colors={'ordinary':RED,'weighted':TEAL,'weighted_order':BLUE}
    for r in selected:
        f=r['frames'];x=[a['step'] for a in f]
        back=[100*geometry(a['q'],a['y'],a['slope'])['backward_fraction'] for a in f]
        axes[0].plot(x,back,c=colors[r['method']],label=LABEL[r['method']])
        axes[1].plot(x,[a['alignment'] for a in f],c=colors[r['method']])
        axes[2].plot(x,[a['corrected_sigreg']/1024 for a in f],c=colors[r['method']])
    axes[0].set(title='Does the curve still double back?',ylabel='Backward part of curve (%)',ylim=(-1,51))
    axes[0].axhline(0,c=TEAL,ls=':',lw=1)
    axes[1].set(title='Pair-alignment error',ylabel='Mean squared code change',yscale='log')
    axes[2].set(title='Distribution mismatch after balancing',ylabel='SIGReg / endpoint count',yscale='log')
    for ax in axes:ax.set_xlabel('Representation-training updates')
    axes[0].legend(fontsize=8)
    fig.suptitle('Representation training · seed 131\nPair alignment is not the ICM forward predictor',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'05_representation_training')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    repair=[r for r in runs if r['stage']=='repair' and r['seed']==131]
    palette={'ordinary':RED,'weighted':TEAL,'weighted_reset':ORANGE,'weighted_order':BLUE,'weighted_strong':PURPLE,'weighted_stored':'#734f32'}
    for r in repair:
        f=r['frames'];x=[4000+a['step'] for a in f]
        axes[0].plot(x,[100*geometry(a['q'],a['y'],a['slope'])['backward_fraction'] for a in f],
            c=palette[r['method']],label=LABEL[r['method']])
        axes[1].plot(np.exp(f[-1]['q']),normalize(f[-1]['q'],f[-1]['y']),c=palette[r['method']])
    axes[0].set(title='Start with the same folded model',xlabel='Total representation updates',ylabel='Backward part of curve (%)',ylim=(-1,51))
    axes[0].axhline(0,c=TEAL,ls=':',lw=1)
    axes[1].set(title='Shape after another 8,000 updates',xlabel='Actual distance',ylabel='Learned code (display scale)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Can changed gradients undo an existing fold? · seed 131',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'06_late_fold_repair')


def predictor_graphs():
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed in range(130,140):
        saved=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
        h=saved['history'];x=[a['update']+1 for a in h];y=[a['forward_error'] for a in h]
        for ax in axes:ax.plot(x,y,alpha=.6,lw=1.2)
    axes[0].set(title='First 600 actor/ICM updates',ylabel='Training forward prediction error',xlabel='Actor/ICM updates')
    axes[1].set(title='Same runs, logarithmic error scale',ylabel='Training forward prediction error',xlabel='Actor/ICM updates',yscale='log')
    fig.suptitle('All ten original ICM runs: error falls early, then rises again',fontsize=14)
    fig.text(.5,-.02,'Error = ½ × sum of squared errors over the 32 learned features, averaged over transitions. Each line is one seed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.9));save(fig,'07_all_original_predictor_runs')
    seeds=(130,131,136);colors=(PURPLE,TEAL,ORANGE)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    summaries=[]
    for seed,color in zip(seeds,colors):
        r=old.trials.load(OUT/f'joint-predictor-{seed}.pt');f=r['frames'];x=[a['update'] for a in f]
        axes[0].plot(x,[a['on_policy']['error'] for a in f],c=color,label=f'Seed {seed}')
        axes[1].plot(x,[a['on_policy']['target_variance'] for a in f],c=color)
        axes[2].plot(x,[a['on_policy']['normalized_error'] for a in f],c=color)
        for a in f:summaries.append(dict(seed=seed,**a))
    for ax in axes:
        ax.set(xlabel='Actor/ICM updates',yscale='log');ax.axvline(600,c='gray',ls='--',lw=1)
    axes[0].set(title='Prediction error on fresh trajectories',ylabel='Held-out forward prediction error')
    axes[1].set(title='The feature targets change scale',ylabel='½ × total feature variance')
    axes[2].set(title='Error relative to target variation',ylabel='Error / feature variance')
    axes[2].axhline(1,c='black',lw=.7,ls=':')
    axes[0].legend(fontsize=8)
    fig.suptitle('Continue ICM training to 1,800 updates: measured error does not go to zero\nDashed line = original training cutoff; tiny early error coincides with nearly constant features',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.85));save(fig,'08_predictor_error_and_moving_targets')
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharex=True,sharey=True)
    frozen=[]
    for ax,seed in zip(axes,seeds):
        r=old.trials.load(OUT/f'frozen-predictor-{seed}.pt');f=r['frames'];x=[a['step'] for a in f]
        ax.plot(x,[a['train_error'] for a in f],c=BLUE,label='Stored training transitions')
        ax.plot(x,[a['heldout_error'] for a in f],c=TEAL,label='Independent held-out trajectories')
        ax.set(title=f'Seed {seed}',xlabel='Additional forward-model updates',yscale='log')
        ax.text(.96,.94,f"Final held-out: {f[-1]['heldout_error']:.4g}",ha='right',va='top',transform=ax.transAxes)
        frozen.append(dict(seed=seed,initial=f[0],final=f[-1]))
    axes[0].set_ylabel('Forward prediction error');axes[0].legend(fontsize=8)
    fig.suptitle('Freeze the actor and feature encoder; train only the forward predictor\nError decreases further, but remains nonzero in this finite run',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.85));save(fig,'09_frozen_target_predictor')
    # Individual extended training curves, including fixed all-action probes.
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,seed in zip(axes,seeds):
        r=old.trials.load(OUT/f'joint-predictor-{seed}.pt');f=r['frames'];h=r['history']
        ax.plot([a['update']+1 for a in h],[a['forward_error'] for a in h],c='gray',alpha=.55,label='Current training batch')
        ax.plot([a['update'] for a in f],[a['on_policy']['error'] for a in f],c=TEAL,label='Fresh on-policy trajectories')
        ax.plot([a['update'] for a in f],[a['fixed_grid']['error'] for a in f],c=ORANGE,label='Fixed distances, all actions')
        ax.set(title=f'Seed {seed}',xlabel='Actor/ICM updates',yscale='log');ax.axvline(600,c='gray',ls=':')
    axes[0].set_ylabel('Forward prediction error');axes[0].legend(fontsize=8)
    fig.suptitle('Training and held-out predictor error through the extended runs',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'10_training_and_heldout_predictor')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for seed,color in zip(seeds,colors):
        r=old.trials.load(OUT/f'joint-predictor-{seed}.pt');f=r['frames']
        x=[a['update'] for a in f];y=[a['on_policy']['error'] for a in f]
        ax.plot(x,y,c=color,lw=2,label=f'Seed {seed}')
        ax.annotate(f'{y[-1]:.4f}',(x[-1],y[-1]),xytext=(12,{130:20,131:0,136:-17}[seed]),textcoords='offset points',c=color,fontsize=11)
    ax.axvline(600,c='gray',ls='--',lw=1)
    ax.text(620,.35,'Original training cutoff',fontsize=10,color='gray')
    ax.set(xlabel='Actor/ICM training updates',ylabel='Prediction error on new trajectories',yscale='log',
        xlim=(0,2050),ylim=(.0005,1.5),yticks=(.001,.01,.1,1),yticklabels=('0.001','0.01','0.1','1'))
    ax.legend(fontsize=9,loc='lower right')
    ax.set_title('Prediction error falls early, rises, and remains nonzero after longer training',pad=16)
    fig.text(.5,-.01,'Logarithmic error axis: lower is better. All recorded errors are positive.\nThe learned feature targets change during these runs; raw error alone does not measure useful dynamics learning.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,'11_does_predictor_error_vanish')
    old.trials.atomic_json(OUT/'predictor-summary.json',dict(joint=summaries,frozen=frozen))
    return summaries,frozen


def verify(runs):
    torch.set_num_threads(1)
    result=[]
    for r in runs:
        c,_,_,_=source(r['seed']);m=old.ScaledEncoder(r['gain']);m.load_state_dict(r['encoder'])
        q,y,s=curve(m,c);f=r['frames'][-1]
        np.testing.assert_array_equal(y,f['y']);np.testing.assert_array_equal(s,f['slope'])
        assert roots(q,s)==f['roots']
        dq=torch.linspace(0,np.log(16),10001,requires_grad=True)
        dy=m(old.trials.behavior.sensor(dq.exp(),c));ds=torch.autograd.grad(dy.sum(),dq)[0]
        assert roots(dq.detach().numpy(),ds.detach().numpy())==r['dense_roots']
        if r['stage']=='repair':
            original=old.trials.load(PREVIOUS/'runs'/f"native-ordinary-{r['seed']}.pt")
            assert all(torch.equal(v,original['encoder'][k]) for k,v in r['states'][0].items())
        if r['method']=='weighted_stored':
            _,_,endpoints,_=source(r['seed']);_,probe_q=stored_order_probes(endpoints)
            np.testing.assert_array_equal(probe_q.numpy(),r['stored_probe_q'])
        result.append(dict(stage=r['stage'],gain=r['gain'],seed=r['seed'],method=r['method'],
            geometry_exact=True,dense_roots_exact=True,fold_count_agrees=bool(f['roots'])==bool(r['dense_roots'])))
    # Test the order penalty's gradient independently of the real data.
    c,_,_,_=source(131);torch.manual_seed(123);model=old.ScaledEncoder(4).double()
    q=torch.linspace(0,np.log(16),65,dtype=torch.float64)
    xx=old.trials.behavior.sensor(q.exp(),c)
    yy=model(xx);diff=torch.diff(yy)/(q[1]-q[0])
    sign=-1 if float(diff.detach().mean())>0 else 1
    val=torch.relu(-sign*diff).square().mean()
    parameter=model.net[4].weight
    grad=torch.autograd.grad(val,parameter)[0]
    ix=np.unravel_index(int(grad.abs().argmax()),tuple(grad.shape))
    analytic=grad[ix].item()
    assert abs(analytic)>1e-8
    original=parameter[ix].item();eps=1e-5;values=[]
    for shift in (-eps,eps):
        with torch.no_grad():parameter[ix]=original+shift
        delta=torch.diff(model(xx))/(q[1]-q[0]);values.append(torch.relu(-sign*delta).square().mean().item())
    fd=(values[1]-values[0])/(2*eps)
    assert np.isclose(fd,analytic,rtol=1e-4,atol=1e-8)
    with torch.no_grad():parameter[ix]=original
    def strong_penalty():
        delta=torch.diff(model(xx))/(q[1]-q[0])
        return 100*torch.minimum(torch.relu(.1-delta).square().mean(),torch.relu(.1+delta).square().mean())
    grad2=torch.autograd.grad(strong_penalty(),parameter)[0]
    ix2=np.unravel_index(int(grad2.abs().argmax()),tuple(grad2.shape))
    analytic2=grad2[ix2].item();assert abs(analytic2)>1e-8
    original2=parameter[ix2].item();vals=[]
    for shift in (-eps,eps):
        with torch.no_grad():parameter[ix2]=original2+shift
        vals.append(float(strong_penalty().detach()))
    fd2=(vals[1]-vals[0])/(2*eps)
    assert np.isclose(fd2,analytic2,rtol=1e-4,atol=1e-8)
    from predictor_training import evaluate,fixed_grid,B
    predictor_checks=[]
    for seed in (130,131,136):
        r=old.trials.load(OUT/f'joint-predictor-{seed}.pt')
        c=B.Config(**r['config']);policy,icm=B.Policy(),B.ICM()
        for step,state in r['states'].items():
            policy.load_state_dict(state['policy']);icm.load_state_dict(state['icm'])
            with torch.random.fork_rng():heldout=B.collect(c,'icm',64,5100000+seed,policy)
            expected=next(a for a in r['frames'] if a['update']==step)
            for label,data in (('on_policy',heldout),('fixed_grid',fixed_grid(c))):
                values=evaluate(icm,data)
                for key in values:assert np.isclose(values[key],expected[label][key],rtol=0,atol=1e-8),(seed,step,label,key)
            predictor_checks.append(dict(seed=seed,updates=step,heldout_and_fixed_grid_recomputed=True))
        frozen=old.trials.load(OUT/f'frozen-predictor-{seed}.pt')
        original=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
        policy.load_state_dict(original['policy']);icm.load_state_dict(frozen['icm'])
        assert all(torch.equal(v,original['icm'][k]) for k,v in icm.state_dict().items() if k.startswith('features.'))
        with torch.random.fork_rng():heldout=B.collect(c,'icm',64,5200000+seed,policy)
        actual=evaluate(icm,heldout)['error']
        assert np.isclose(actual,frozen['frames'][-1]['heldout_error'],rtol=0,atol=1e-8)
    old.trials.atomic_json(OUT/'verification.json',dict(models=result,
        order_gradient=dict(analytic=analytic,finite_difference=fd),
        strong_order_gradient=dict(analytic=analytic2,finite_difference=fd2),
        predictor_measurements=predictor_checks,
        predictor_update600_exact={str(s):old.trials.load(OUT/f'joint-predictor-{s}.pt')['update600_exact'] for s in (130,131,136)}))


def write_gallery():
    names=['01_what_weighting_means','02_fold_makes_distances_ambiguous','12_before_and_after_actual_repair','03_fold_moves_inward',
       '04_gradient_results','05_representation_training','06_late_fold_repair','07_all_original_predictor_runs',
       '11_does_predictor_error_vanish','08_predictor_error_and_moving_targets','09_frozen_target_predictor','10_training_and_heldout_predictor']
    html=['<!doctype html><meta charset="utf-8"><title>Gradient repairs and predictor learning curves</title><style>body{font:18px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}img{max-width:100%}section{margin:50px 0}</style><h1>Gradient repairs and predictor learning curves</h1><p><a href="../README.md">Read the explanation and experimental limits</a></p>']
    for name in names:html.append(f'<section><h2>{name[3:].replace("_"," ")}</h2><img src="{name}.png"></section>')
    (OUT/'index.html').write_text('\n'.join(html))


def main():
    torch.set_num_threads(1)
    runs=[old.trials.load(p) for p in sorted((OUT/'runs').glob('*.pt'))]
    if len(runs)!=102:raise RuntimeError(f'Need all 102 new runs; have {len(runs)}')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    verify(runs)
    rows,summary=overview(runs)
    old.trials.atomic_json(OUT/'gradient-summary.json',dict(summary=summary,runs=rows))
    weighting_picture();ambiguity=ambiguity_picture();inward_picture();concrete_repair_picture();counts_picture(summary);run_curves(runs)
    predictor,frozen=predictor_graphs()
    old.trials.atomic_json(OUT/'ambiguity-example.json',ambiguity)
    write_gallery()
    lines=['# Gradient intervention results','',
       '| Stage | Input gain | Method | Final folded runs | Mean log-distance R² | Mean backward travel (%) | Minimum output std |',
       '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for a in summary:lines.append(f"| {a['stage']} | {a['gain']} | {LABEL[a['method']]} | {a['folded']}/{a['n']} | {a['mean_r2']:.4f} | {100*a['mean_backward_fraction']:.5f} | {a['minimum_grid_std']:.4f} |")
    lines+=['','## Held-out predictor error: same evolving policy, new trajectories','',
       '| Seed | Update 100 | Update 600 | Update 1,800 | Frozen-target fitting: start → end |',
       '| ---: | ---: | ---: | ---: | ---: |']
    for seed in (130,131,136):
        vals=[next(a for a in predictor if a['seed']==seed and a['update']==step)['on_policy']['error'] for step in (100,600,1800)]
        ff=next(a for a in frozen if a['seed']==seed)
        lines.append(f"| {seed} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {ff['initial']['heldout_error']:.6f} → {ff['final']['heldout_error']:.6f} |")
    lines+=['','Error means half the summed squared error over 32 features, averaged over transitions. The frozen test has fixed targets and a separate held-out dataset; it is not a continuation of the moving-policy error curve.',
       '', 'Backward variation in summary metrics is sign-invariant: min(total upward variation, total downward variation) / total variation. The stored training frames also retain the original fixed-orientation backward measure; output sign is arbitrary.',
       '', '[Explanation](../README.md) · [Graph gallery](index.html) · [Gradient metrics](gradient-summary.json) · [Predictor metrics](predictor-summary.json) · [Verification](verification.json)','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    sources=[HERE/'PROTOCOL.md',HERE/'README.md',HERE/'train_gradients.py',HERE/'predictor_training.py',HERE/'report_followup.py',HERE.parent/'run_study.py']
    sources += [old.ORIGINAL/'sources'/f'icm-{seed}.pt' for seed in range(130,140)]
    old.trials.atomic_json(OUT/'source-hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

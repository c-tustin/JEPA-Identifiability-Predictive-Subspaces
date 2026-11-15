"""Reload checkpoints, verify curves, render requested occupancy galleries."""
from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
old=base.old

def load(distribution,method,seed):
    return old.trials.load(base.OUT/'runs'/f'{distribution}-{method}-{seed}.pt')
def scaled(r):
    y=r['frames'][-1]['y']; q=r['q']
    return (1 if np.cov(q,y)[0,1]>=0 else -1)*(y-y.mean())/y.std()
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)
def galleries():
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for col,seed in enumerate((130,131,134,136)):
        r=load('native','ordinary',seed)
        data=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')['data']
        d=old.trials.distances(torch.cat([data['x'],data['xp']])).numpy()
        bins=np.geomspace(1,16,33); count,_=np.histogram(d,bins)
        axes[0,col].stairs(count/count.sum(),bins,fill=True,alpha=.75,color='#435774')
        axes[0,col].set(title=f"Seed {seed}: {'folded' if r['metrics']['folded'] else 'non-folded'}",ylabel='Probability per log bin',xlabel='Distance',xscale='log',ylim=(0,.8),xticks=[1,2,4,8,16])
        axes[0,col].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        for method,color,label in [('ordinary','#c13c45','Ordinary'),('inverse_gradient','#d78a19','Inverse gradient'),('weighted_moments','#147b79','Weighted moments')]:
            rr=load('native',method,seed)
            axes[1,col].plot(np.exp(rr['q']),scaled(rr),color=color,label=label)
        axes[1,col].set(xlabel='Distance',ylabel='Encoder (standardized)',ylim=(-4,4))
        for root in r['metrics']['roots']: axes[1,col].axvline(root,color='#c13c45',alpha=.4,ls=':')
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Stored ICM occupancy and learned scalar encoders\nFold status refers to ordinary SIGReg; same frozen occupancy for all three training methods')
    save(fig,'native_occupancy_encoders')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,dist in enumerate(('bimodal','trimodal')):
        r=load(dist,'ordinary',131); bins=np.geomspace(1,16,65)
        axes[0,col].stairs(r['mass'],bins,fill=True,alpha=.7,color='#435774')
        axes[0,col].set(title='Peaks at '+(', '.join(str(int(x)) for x in base.PEAKS[dist])),xlabel='Distance',ylabel='Probability per log bin')
        for seed,style in zip(SEEDS,('-','--',':')):
            for method,color,label in [('ordinary','#c13c45','Ordinary'),('inverse_gradient','#d78a19','Inverse gradient'),('weighted_moments','#147b79','Weighted moments')]:
                rr=load(dist,method,seed)
                axes[1,col].plot(np.exp(rr['q']),scaled(rr),style,color=color,label=f'{label}, {seed}',lw=1.5)
        axes[1,col].set(xlabel='Distance',ylabel='Encoder (standardized)')
    axes[1,1].legend(fontsize=7,ncol=2)
    fig.suptitle('Bimodal and trimodal SIGReg occupancy with corresponding learned encoders\nControlled endpoint distributions; original ICM alignment pairs remain fixed')
    save(fig,'bimodal_trimodal_encoders')

def results():
    runs=[]
    for path in sorted(OUT.glob('*.pt')):
        r=old.trials.load(path); model=old.ScaledEncoder(4); model.load_state_dict(r['encoder'])
        c=old.trials.behavior.Config(**r['config']); q,y,s=base.curve(model,c)
        np.testing.assert_array_equal(y,r['frames'][-1]['y']); np.testing.assert_array_equal(s,r['frames'][-1]['slope'])
        assert base.roots(q,s)==r['metrics']['roots']
        assert abs(float(np.corrcoef(q,y)[0,1]**2)-r['metrics']['r2']) < 1e-12
        runs.append(r)
    assert len(runs)==15, f'Expected 15 new completed fits, got {len(runs)}'
    for seed in SEEDS:
        for method in ('ordinary','inverse_gradient','weighted_moments'):
            r=load('native',method,seed); r.update(factor=1.,compensate=False); runs.append(r)
    summary=[]
    for method in ('ordinary','inverse_gradient','weighted_moments'):
        for factor,comp in ((.25,False),(1.,False),(4.,False),(1.,True)):
            group=[r for r in runs if r['method']==method and r['factor']==factor and r['compensate']==comp]
            if not group: continue
            summary.append(dict(method=method,lr_factor=factor,norm_compensation=comp,n=len(group),folds=sum(r['metrics']['folded'] for r in group),mean_r2=float(np.mean([r['metrics']['r2'] for r in group])),seeds=[dict(seed=r['seed'],**r['metrics']) for r in group]))
    old.trials.atomic_json(OUT/'summary.json',dict(summary=summary,new_checkpoints_verified=15))
    lines=['# Learning-rate follow-up results','','Three paired historical sources, seeds 130, 131, 136; 4,000 updates from initialization. 15 new fits, nine reused original controls. All 15 new checkpoint curves/slopes reproduced exactly after reload.','','| Method | Rate multiplier | Final folds | Mean log-distance R² |','| --- | --- | --- | --- |']
    for g in summary:
        rate='Per-step norm matched' if g['norm_compensation'] else str(g['lr_factor'])
        lines.append(f"| {g['method']} | {rate} | {g['folds']}/{g['n']} | {g['mean_r2']:.4f} |")
    lines+=['','Norm matching compensates the full raw gradient norm before Adam, not the final Adam step. Folds are sampled sign reversals. These are prevention experiments with privileged occupancy estimation; no fresh actors or late-repair runs.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for col,seed in enumerate(SEEDS):
        for row,method in enumerate(('inverse_gradient','weighted_moments')):
            for r in [r for r in runs if r['seed']==seed and r['method']==method]:
                label='Norm matched' if r['compensate'] else f"{r['factor']:g} × LR"
                axes[row,col].plot(np.exp(r['q']),scaled(r),label=label)
            axes[row,col].set(title=f'{method}, seed {seed}',xlabel='Distance',ylabel='Encoder (standardized)')
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('Inverse-occupancy training with learning-rate controls · 4,000 updates')
    save(fig,'learning_rate_comparison')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    torch.set_num_threads(1);galleries()
    if len(list(OUT.glob('*.pt')))==15:results()

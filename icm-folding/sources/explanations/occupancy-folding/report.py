"""Recompute diagnostics from saved models, then build figures and report."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtri
from run_study import OUT, HERE, MECH, ROOT, old, METHODS, PEAKS, curve, roots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'ordinary': '#c13c45', 'inverse_gradient': '#d78a19', 'weighted_moments': '#147b79'}
LABELS = {'ordinary': 'Ordinary SIGReg', 'inverse_gradient': 'Inverse gradient only', 'weighted_moments': 'Inverse-weighted moments'}


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def standardized(run, frame):
    final = run['frames'][-1]['y']
    sign = 1 if np.cov(run['q'], final)[0,1] >= 0 else -1
    return sign*(frame['y']-final.mean()) / final.std()


def report():
    torch.set_num_threads(1)
    runs = [old.trials.load(p) for p in sorted((OUT/'runs').glob('*.pt'))]
    if len(runs) != 60:
        raise RuntimeError(f'Expected 60 complete runs, found {len(runs)}')
    lookup = {(r['distribution'], r['method'], r['seed']): r for r in runs}
    checks = []
    for r in runs:
        c = old.trials.behavior.Config(**r['config'])
        model = old.ScaledEncoder(4); model.load_state_dict(r['encoder'])
        q,y,s = curve(model, c)
        np.testing.assert_array_equal(y, r['frames'][-1]['y'])
        np.testing.assert_array_equal(s, r['frames'][-1]['slope'])
        assert roots(q,s) == r['metrics']['roots']
        assert abs(np.corrcoef(q,y)[0,1]**2-r['metrics']['r2']) < 1e-12
        err = None
        if r['distribution'] == 'native' and r['method'] == 'ordinary':
            baseline = old.trials.load(MECH/'output/runs'/f"ordinary-g4-{r['seed']}.pt") if (MECH/'output/runs'/f"ordinary-g4-{r['seed']}.pt").exists() else old.trials.load(old.ORIGINAL/'encoders'/f"icm-{r['seed']}.pt")
            err = max(float((v-baseline['encoder'][k]).abs().max()) for k,v in r['encoder'].items())
            assert err == 0, (r['seed'], err)
        checks.append(dict(seed=r['seed'], distribution=r['distribution'], method=r['method'],
                           final_curve_and_slope_exact=True, baseline_parameter_error=err))
    summary = []
    for d in ('native', *PEAKS):
        for method in METHODS:
            group = [r for r in runs if r['distribution']==d and r['method']==method]
            if group:
                summary.append(dict(distribution=d, method=method, n=len(group),
                    folded=sum(r['metrics']['folded'] for r in group),
                    mean_r2=float(np.mean([r['metrics']['r2'] for r in group])),
                    mean_reversal=float(np.mean([r['metrics']['reversal_fraction'] for r in group]))))
    old.trials.atomic_json(OUT/'summary.json', dict(summary=summary,
        runs=[dict(seed=r['seed'], distribution=r['distribution'],method=r['method'],**r['metrics']) for r in runs]))
    lines=['# Measured occupancy interventions', '',
        'All runs: gain 4, 4,000 updates, pure ICM alignment. R² uses a uniform-log-distance evaluation grid. Controlled distributions change only SIGReg sampling; seeds 130, 131, 136 are diagnostic.', '',
        '| Occupancy | Method | Final folded runs | Mean log-distance R² |',
        '| --- | --- | ---: | ---: |']
    for item in summary:
        lines.append(f"| {item['distribution']} | {LABELS[item['method']]} | {item['folded']}/{item['n']} | {item['mean_r2']:.4f} |")
    lines += ['', '## Ordinary-SIGReg trajectory details', '',
        'All turning distances are reported; a blank means no turn in [1,16]. Transient turns can disappear. Snapshots are 25 updates apart.', '',
        '| Occupancy | Seed | At 250 | At 500 | At 1,000 | At 4,000 |',
        '| --- | ---: | --- | --- | --- | --- |']
    for d in ('native', *PEAKS):
        for seed in (130,131,136):
            r=lookup[d,'ordinary',seed]
            vals=[]
            for step in (250,500,1000,4000):
                f=next(f for f in r['frames'] if f['step']==step)
                vals.append(', '.join(f'{v:.3f}' for v in f['roots']) or '—')
            lines.append(f'| {d} | {seed} | '+ ' | '.join(vals)+' |')
    lines += ['', '[Explanation and limitations](../README.md) · [Graph gallery](index.html) · [Machine-readable metrics](summary.json)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    old.trials.atomic_json(OUT/'verification.json', checks)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    # One column per occupancy. All roots shown, including transient ones.
    fig, axes = plt.subplots(3,4,figsize=(16,10), sharex='row')
    for col,d in enumerate(PEAKS):
        r = lookup[d,'ordinary',131]
        centers = np.exp((np.arange(64)+.5)*np.log(16)/64)
        axes[0,col].plot(centers,r['mass'],color='#435774')
        axes[0,col].set_title({'near2':'One peak: 2','bimodal':'Two peaks: 2, 8','trimodal':'Three peaks: 2, 5, 12','uniform':'Uniform in log-distance'}[d])
        axes[0,col].set_ylabel('Probability per log bin' if col==0 else '')
        for seed,marker in zip((130,131,136),('o','s','^')):
            rr = lookup[d,'ordinary',seed]
            for f in rr['frames']:
                axes[1,col].scatter([f['step']]*len(f['roots']),f['roots'], s=4,
                    c={130:'#8b65a8',131:'#c13c45',136:'#368b76'}[seed], marker=marker, alpha=.5)
        axes[1,col].set(ylim=(1,16),xlim=(0,4000),xlabel='Encoder updates', ylabel='Turning distance' if col==0 else '')
        for step in (0,100,250,500,1000,4000):
            f = next(f for f in r['frames'] if f['step']==step)
            axes[2,col].plot(np.exp(r['q']),standardized(r,f),label=str(step), alpha=.85)
        axes[2,col].set(xlabel='Physical distance', ylabel='Representation (fixed scale)' if col==0 else '')
        axes[2,col].set_ylim(-5,5)
    axes[2,3].legend(title='Updates',fontsize=8)
    axes[1,0].text(.04,.96,'130 purple · 131 red · 136 green',transform=axes[1,0].transAxes,va='top',fontsize=8)
    fig.suptitle('Change only SIGReg occupancy; keep ICM alignment pairs fixed\nDots show every detected turn, not agent movement',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94)); save(fig,'occupancy_and_fold_motion')

    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for col,(d,title) in enumerate((('native','Stored ICM occupancy'),('bimodal','Two controlled peaks'),('trimodal','Three controlled peaks'))):
        for method in METHODS:
            r = lookup[d,method,131]
            axes[0,col].plot(np.exp(r['q']),standardized(r,r['frames'][-1]), color=COLORS[method],label=LABELS[method])
            for f in r['frames']:
                axes[1,col].scatter([f['step']]*len(f['roots']), f['roots'],s=5,color=COLORS[method],alpha=.6)
        axes[0,col].set(title=title,xlabel='Physical distance',ylabel='Final representation' if col==0 else '')
        axes[1,col].set(xlabel='Encoder updates',xlim=(0,4000),ylim=(1,16),ylabel='Turning distance' if col==0 else '')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Inverse-occupancy gradients versus inverse-weighted SIGReg moments · seed 131',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'inverse_occupancy_comparison')

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    native=lookup['native','ordinary',131]
    for step in (0,100,225,250,500,1000,2000,4000):
        frame=next(f for f in native['frames'] if f['step']==step)
        axes[0].plot(np.exp(native['q']),standardized(native,frame),label=str(step))
        if frame['roots']:
            axes[0].scatter(frame['roots'],np.interp(frame['roots'],np.exp(native['q']),standardized(native,frame)),s=20)
    axes[0].set(xlabel='Physical distance',ylabel='Representation (one fixed scale)',ylim=(-4,4),title='The learned curve bends and its turn moves')
    axes[0].legend(title='Encoder updates',fontsize=8)
    for method in METHODS:
        r=lookup['native',method,131]
        for f in r['frames']:
            axes[1].scatter([f['step']]*len(f['roots']),f['roots'],s=7,c=COLORS[method])
        axes[1].plot([],[],'.',color=COLORS[method],label=LABELS[method])
    axes[1].set(xlabel='Encoder updates',ylabel='Turning distance',xlim=(0,4000),ylim=(1,16),title='All detected turns; sampled every 25 updates')
    axes[1].legend(fontsize=8)
    fig.suptitle('Native ICM seed 131: inward fold migration and inverse-occupancy interventions')
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'native_inward_migration')

    fig,axes=plt.subplots(2,5,figsize=(17,7),sharex=True,sharey='row')
    for col,seed in enumerate(range(130,135)):
        for row,ss in enumerate((seed,seed+5)):
            for method in METHODS:
                r=lookup['native',method,ss]
                axes[row,col].plot(np.exp(r['q']),standardized(r,r['frames'][-1]),color=COLORS[method],label=LABELS[method])
            axes[row,col].set(title=f'Seed {ss}',xlabel='Distance',ylim=(-4,4))
    axes[0,0].legend(fontsize=7)
    fig.suptitle('All ten stored ICM sources · identical initialization within each seed',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'all_native_runs')

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,d in zip(axes,('near2','bimodal','trimodal')):
        mass=lookup[d,'ordinary',131]['mass']
        centers=np.exp((np.arange(64)+.5)*np.log(16)/64)
        quantile=ndtri(np.cumsum(mass)-mass/2)
        ax.plot(centers,quantile,lw=2,label='Monotone Gaussianizing map')
        ax.set(title=d,xlabel='Physical distance',ylabel='Output Gaussian quantile')
        for peak in PEAKS[d]: ax.axvline(peak,color='gray',alpha=.3,ls='--')
    fig.suptitle('Even multimodal occupancy permits an ordered representation; folding is not required')
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'monotone_gaussianization')

    policies = old.trials.load(OUT/'policy-maps.pt')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    policy_summary=[]
    for r in policies:
        d=r['d']; reward=(r['reward']*r['probs']).sum(1)
        near=(d>=1.5)&(d<=3);far=(d>=8)&(d<=14)
        attractors=roots(np.log(d),r['drift'])
        ix=np.flatnonzero((r['drift'][:-1]>0)&(r['drift'][1:]<0))
        stable=(d[ix]-r['drift'][ix]*(d[ix+1]-d[ix])/(r['drift'][ix+1]-r['drift'][ix])).tolist()
        policy_summary.append(dict(seed=r['seed'],stable_drift_roots=stable,
            mean_drift_8_14=float(r['drift'][far].mean()),
            mean_reward_1_5_3=float(reward[near].mean()),mean_reward_8_14=float(reward[far].mean()),
            stored_fraction_below4=float(np.mean(r['stored_distances']<4)),
            stationary_fraction_below4=float(r['stationary'][r['lattice']<4].sum()),
            expected_steps_2_to_4=r['expected_steps_2_to_4'],
            stationary_residual=r['stationary_residual']))
        for ax,y in ((axes[0,0],r['drift']),(axes[0,1],reward),(axes[0,2],r['probs'][:,1])):
            ax.plot(d,y,alpha=.4,lw=1)
    for col,seed in enumerate((130,131,136)):
        r=next(r for r in policies if r['seed']==seed)
        bins=np.geomspace(1,16,33)
        hist,_=np.histogram(r['stored_distances'],bins)
        axes[1,col].stairs(hist/hist.sum(),bins,label='Stored trajectory occupancy',color='#243e60')
        projected,_=np.histogram(r['lattice'],bins,weights=r['chain_collection'])
        stat,_=np.histogram(r['lattice'],bins,weights=r['stationary'])
        axes[1,col].plot(np.sqrt(bins[:-1]*bins[1:]),projected,label='Lattice chain: collection window',color='#d78a19')
        axes[1,col].plot(np.sqrt(bins[:-1]*bins[1:]),stat,label='Lattice chain: stationary',color='#147b79',ls='--')
        axes[1,col].set(title=f'Seed {seed}',xlabel='Physical distance',ylabel='Probability per log bin')
    axes[0,0].axhline(0,c='black',lw=.7)
    axes[0,0].set(title='Expected reflected distance change',xlabel='Physical distance',ylabel='Distance per action')
    axes[0,1].set(title='Current expected intrinsic reward',xlabel='Physical distance',ylabel='Reward')
    axes[0,2].set(title='Probability of stay action',xlabel='Physical distance',ylabel='Probability')
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Why the actor remains near the object · ten frozen policies',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'policy_attraction')
    old.trials.atomic_json(OUT/'policy-summary.json',policy_summary)
    trace_summary=[]

    fig,axes=plt.subplots(3,3,figsize=(14,11),sharex=True)
    for col,seed in enumerate((130,131,136)):
        trace=old.trials.load(OUT/f'policy-trace-{seed}.pt')
        for frame in trace['frames']:
            label=str(frame['update']+1)
            d=frame['d']
            reward=(frame['reward']*frame['probs']).sum(1)
            near=(d>=1.5)&(d<=3); far=(d>=8)&(d<=14)
            trace_summary.append(dict(seed=seed, updates=frame['update']+1,
                near_reward=float(reward[near].mean()),far_reward=float(reward[far].mean()),
                far_drift=float(frame['drift'][far].mean()),
                fraction_below4=float((frame['distances']<4).mean())))
            axes[0,col].plot(d,(frame['reward']*frame['probs']).sum(1),label=label)
            axes[1,col].plot(d,frame['drift'],label=label)
            bins=np.geomspace(1,16,33); hist,_=np.histogram(frame['distances'],bins)
            axes[2,col].plot(np.sqrt(bins[:-1]*bins[1:]),hist/hist.sum(),label=label)
        axes[0,col].set(title=f'Seed {seed}',yscale='log')
        axes[1,col].axhline(0,c='black',lw=.7)
        axes[2,col].set_xlabel('Physical distance')
    axes[0,0].set_ylabel('Expected intrinsic reward')
    axes[1,0].set_ylabel('Expected distance change')
    axes[2,0].set_ylabel('Fresh frozen-policy occupancy')
    axes[0,2].legend(title='Actor updates',fontsize=8)
    fig.suptitle('Training history: prediction error, inward drift, then concentrated occupancy\nInstrumentation reproduces all three original final actor and ICM weights exactly',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'policy_training_history')
    old.trials.atomic_json(OUT/'policy-history-summary.json',trace_summary)

    files=['native_inward_migration','occupancy_and_fold_motion','inverse_occupancy_comparison','all_native_runs','monotone_gaussianization','policy_attraction','policy_training_history']
    html=['<!doctype html><meta charset="utf-8"><title>Occupancy and folding</title><style>body{font:18px system-ui;max-width:1500px;margin:40px auto;padding:0 20px}img{max-width:100%}</style><h1>Occupancy, SIGReg gradients, and ICM attraction</h1>']
    for name in files: html.append(f'<h2>{name.replace("_"," ")}</h2><img src="{name}.png">')
    (OUT/'index.html').write_text('\n'.join(html))
    sources=[HERE/'run_study.py',HERE/'policy_diagnostics.py',HERE/'PROTOCOL.md', MECH/'output/settings.json']
    sources += [old.ORIGINAL/'sources'/f'icm-{seed}.pt' for seed in range(130,140)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    old.trials.atomic_json(OUT/'source-hashes.json',hashes)
    print(json.dumps(summary,indent=2))
    print(json.dumps(policy_summary,indent=2))


if __name__=='__main__': report()

"""Reload final models, verify paired controls, and make diagnostic figures."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/icm-gradient-cache')
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import HERE,OUT,METHODS,B,trials,source,geometry,standardized_successors,collision_loss,common_data

LABELS=dict(sigreg='SIGReg',vicreg='VICReg (scalar)',infonce='InfoNCE (distance)',quantile='Gaussian quantiles')
COLORS={0.:'#b94348',1.:'#147b79'}


def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def load_all():
    rows=[];allruns={};checks=[]
    for seed in range(130,140):
        c,train,test,source_hash=source(seed)
        common=common_data(c)
        f,mean,std=standardized_successors(train)
        episode_count=c.train_episodes
        assert len(train['x'])==int(.8*episode_count)*c.horizon
        assert len(train['x'])+len(test['x'])==episode_count*c.horizon
        ref=trials.load(OUT/'references'/f'{seed}.pt')
        init=plan=None
        for method in METHODS:
            for coefficient in (0.,1.):
                path=OUT/'runs'/f'{seed}-{method}-c{coefficient:g}-n4000.pt'
                r=trials.load(path)
                assert r['source_hash']==source_hash==ref['source_hash']
                assert r['train_keys']==['action','x','xp']
                if init is None:
                    init=r['initial'];plan=r['plan_hash']
                assert r['plan_hash']==plan
                for k in init:
                    torch.testing.assert_close(r['initial'][k],init[k],rtol=0,atol=0)
                model=B.Encoder();model.load_state_dict(r['model'])
                g=geometry(model,c,points=20001)
                assert g['folded']==r['geometry']['folded'],path
                assert abs(g['backward_fraction']-r['geometry']['backward_fraction'])<1e-6
                assert abs(g['std']-r['geometry']['std'])<.005*max(g['std'],1e-6)
                assert [x['step'] for x in r['frames']]==[0,100,250,500,1000,2000,4000]
                with torch.no_grad():
                    gen=torch.Generator().manual_seed(6100000+seed)
                    idx=torch.randperm(len(test['x']),generator=gen)[:512]
                    y=model(test['x'][idx]);yp=model(test['xp'][idx])
                    target=(torch.tanh(4*test['xp'][idx])-mean)/std
                    conflict=float(collision_loss(y,yp,test['action'][idx],target))
                    ids=torch.randperm(len(common['x']),generator=gen)[:512]
                    yy=model(common['x'][ids]);zz=model(common['xp'][ids])
                    tar=(torch.tanh(4*common['xp'][ids])-mean)/std
                    common_conflict=float(collision_loss(yy,zz,common['action'][ids],tar))
                    code=model(train['x']);center=code.mean();scale=code.std(unbiased=False).clamp_min(.05)
                    decoder=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,6))
                    decoder.load_state_dict(r['prediction']['decoder'])
                    for label,data in [('native',test),('common',common)]:
                        z=(model(data['x'])-center)/scale
                        inputs=torch.cat([z[:,None],torch.nn.functional.one_hot(data['action'],3).float()],1)
                        target=(torch.tanh(4*data['xp'])-mean)/std
                        mse=float((decoder(inputs)-target).square().mean())
                        assert abs(mse-r['prediction'][label+'_mse'])<1e-8
                pred={k:v for k,v in r['prediction'].items() if k!='decoder'}
                row=dict(seed=seed,method=method,coefficient=coefficient,
                    **{k:v for k,v in g.items() if k not in ('q','y','slope')},
                    **pred,heldout_conflict=conflict,common_conflict=common_conflict,
                    raw_native_mse=ref['native_mse'],raw_common_mse=ref['common_mse'])
                rows.append(row);allruns[(seed,method,coefficient)]=r
                checks.append(dict(file=str(path.relative_to(HERE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    dense_grid_points=20001,paired_initialization_and_batches=True,prediction_recomputed=True))
    return rows,allruns,checks


def main():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows,runs,checks=load_all()
    summaries=[];pairs=[]
    for method in METHODS:
        for coefficient in (0.,1.):
            group=[r for r in rows if r['method']==method and r['coefficient']==coefficient]
            summaries.append(dict(method=method,coefficient=coefficient,n=len(group),
                folded=sum(r['folded'] for r in group),material_folded=sum(r['material_fold'] for r in group),
                **{k:float(np.mean([r[k] for r in group])) for k in
                   ('backward_fraction','affine_r2','std','native_mse','common_mse','heldout_conflict','common_conflict')}))
        base=[next(r for r in rows if r['seed']==s and r['method']==method and r['coefficient']==0) for s in range(130,140)]
        treated=[next(r for r in rows if r['seed']==s and r['method']==method and r['coefficient']==1) for s in range(130,140)]
        ratios=np.array([t['native_mse']/b['native_mse'] for b,t in zip(base,treated)])
        rng=np.random.default_rng(20261115)
        boot=np.exp(np.log(ratios)[rng.integers(10,size=(10000,10))].mean(1))
        pairs.append(dict(method=method,native_improved=int(sum(ratios<1)),
            native_geometric_mean_ratio=float(np.exp(np.log(ratios).mean())),
            native_ratio_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            material_folds_repaired=sum(b['material_fold'] and not t['material_fold'] for b,t in zip(base,treated)),
            material_folds_created=sum(not b['material_fold'] and t['material_fold'] for b,t in zip(base,treated))))
    trials.atomic_json(OUT/'summary.json',dict(summary=summaries,paired=pairs,runs=rows))
    trials.atomic_json(OUT/'verification.json',dict(count=len(checks),runs=checks,
        formula_checks=json.loads((OUT/'formula-checks.json').read_text()),source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'run.py',HERE/'report.py',HERE/'PROTOCOL.md')}))
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for coefficient in (0.,1.):
        group=[r for r in summaries if r['coefficient']==coefficient]
        x=np.arange(4)+(coefficient-.5)*.32
        for ax,metric in zip(axes,('folded','material_folded')):
            ax.bar(x,[r[metric] for r in group],width=.3,color=COLORS[coefficient],label='Baseline' if not coefficient else 'Transition gradient')
            ax.set(xticks=np.arange(4),xticklabels=[LABELS[m] for m in METHODS],ylim=(0,10.8),ylabel='Models / 10')
            ax.tick_params(axis='x',labelrotation=15)
    axes[0].set_title('Any derivative sign reversal');axes[1].set_title('Material folds: backward variation > 0.1%')
    axes[0].legend(frameon=False)
    save(fig,'01_fold_rates')

    fig,axes=plt.subplots(2,4,figsize=(14,6),sharex=True)
    for j,m in enumerate(METHODS):
        for i,c in enumerate((0.,1.)):
            for seed in range(130,140):
                g=runs[(seed,m,c)]['geometry'];y=g['y'];q=g['q']
                orient=1 if np.corrcoef(q,y)[0,1]>0 else -1
                axes[i,j].plot(np.exp(q),orient*(y-y.mean())/max(y.std(),1e-8),alpha=.55,lw=1)
            axes[i,j].set_title(LABELS[m]+(' + gradient' if c else ' baseline'))
            axes[i,j].set(xlabel='Physical distance (evaluation only)',ylabel='Standardized code')
    save(fig,'02_all_code_curves')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,metric in enumerate(('native_mse','common_mse')):
        for k,m in enumerate(METHODS):
            for seed in range(130,140):
                pair=[next(r for r in rows if r['seed']==seed and r['method']==m and r['coefficient']==c)[metric] for c in (0.,1.)]
                axes[j].plot([k-.12,k+.12],pair,color='gray',lw=.6,alpha=.6)
                axes[j].scatter([k-.12,k+.12],pair,c=[COLORS[0.],COLORS[1.]],s=13)
        axes[j].set(yscale='log',xticks=range(4),xticklabels=[LABELS[m] for m in METHODS],ylabel='Whitened next-observation MSE')
        axes[j].tick_params(axis='x',labelrotation=15)
    axes[0].set_title('Held-out ICM trajectories');axes[1].set_title('Uniform-distance evaluation: coverage stress')
    axes[0].scatter([],[],color=COLORS[0.],label='Baseline')
    axes[0].scatter([],[],color=COLORS[1.],label='Transition gradient')
    axes[0].legend(frameon=False,fontsize=8)
    save(fig,'03_prediction_errors')

    fig,axes=plt.subplots(1,4,figsize=(14,3.5),sharey=True)
    for ax,m in zip(axes,METHODS):
        for c in (0.,1.):
            for seed in range(130,140):
                frames=runs[(seed,m,c)]['frames']
                x=[];y=[]
                for f in frames:
                    if f['roots']:
                        ax.scatter([f['step']]*len(f['roots']),f['roots'],color=COLORS[c],s=7,alpha=.6)
                    x.append(f['step']);y.append(min(f['roots']) if f['roots'] else np.nan)
                ax.plot(x,y,color=COLORS[c],alpha=.3,lw=.8)
        ax.set(title=LABELS[m],xlabel='Encoder updates',xlim=(0,4000),yscale='log',ylim=(1,16),yticks=(1,2,4,8,16))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].set_ylabel('Turning-point distance')
    axes[0].plot([],[],color=COLORS[0.],label='Baseline')
    axes[0].plot([],[],color=COLORS[1.],label='Transition gradient')
    axes[0].legend(frameon=False,fontsize=8)
    save(fig,'04_fold_trajectories')

    lines=['# Original MLP: gradients and regularizers','',
        'Baseline and transition-gradient fits share initialization and training batches. Ten historical ICM sources; no ordering or dynamics labels enter encoder training. Counts are descriptive.','',
        '| Objective | Gradient | Any folds | Material folds | Mean backward fraction | Held-out MSE | Coverage-stress MSE |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {LABELS[r['method']]} | {'transition' if r['coefficient'] else 'baseline'} | {r['folded']}/10 | {r['material_folded']}/10 | {r['backward_fraction']:.4f} | {r['native_mse']:.6f} | {r['common_mse']:.6f} |")
    lines+=['','The MSE is next-observation prediction from a frozen scalar code and action, with targets whitened using training data. The common uniform-distance set is a coverage/extrapolation stress test, not another held-out draw from the native distribution. Scalar VICReg has no covariance term; InfoNCE uses a scalar RBF score. Material folds exclude reversals with backward variation <= 0.001.','',
        '## Paired prediction comparisons','',
        'Bootstrap intervals resample ten paired source seeds. They are exploratory intervals, without multiple-comparison correction.','']
    for r in pairs:
        lines.append(f"- {LABELS[r['method']]}: {r['native_improved']}/10 held-out prediction improvements; geometric-mean MSE ratio {r['native_geometric_mean_ratio']:.3f}, bootstrap interval {r['native_ratio_bootstrap_95']}; material folds repaired {r['material_folds_repaired']}, created {r['material_folds_created']}.")
    lines+=['','## Prediction references','',
        'The raw-input decoder is an optimization/conditioning reference, not a rigorous lower bound: a scalar feature can make this finite predictor easier to fit.','']
    for key in ('native_mse','common_mse'):
        raw=[r['raw_'+key] for r in rows if r['method']=='sigreg' and r['coefficient']==0]
        action=[r['action_'+key] for r in rows if r['method']=='sigreg' and r['coefficient']==0]
        lines.append(f'- {key}: mean raw-input decoder {np.mean(raw):.6f}; action-only mean predictor {np.mean(action):.6f}.')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>Original MLP gradients</title><style>body{font:18px system-ui;max-width:1400px;margin:40px auto;padding:20px}img{max-width:100%}</style><h1>Original MLP: gradients and regularizers</h1><p><a href="RESULTS.md">Measured tables</a> · <a href="../PROTOCOL.md">Protocol</a></p>']
    for name in ('01_fold_rates','02_all_code_curves','03_prediction_errors','04_fold_trajectories'):
        html.append(f'<h2>{name[3:].replace("_"," ")}</h2><img src="{name}.png">')
    (OUT/'index.html').write_text('\n'.join(html))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()

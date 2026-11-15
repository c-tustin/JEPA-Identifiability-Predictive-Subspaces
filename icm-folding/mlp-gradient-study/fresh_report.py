"""Independent final audit for the fixed-coefficient fresh-source experiment."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/icm-gradient-cache')
import hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import HERE,OUT,B,trials,source,geometry,standardized_successors,common_data

LABELS=('SIGReg','VICReg (scalar)','InfoNCE (distance)','Gaussian quantiles','SIGReg + global mining 100')


def main():
    torch.set_num_threads(1)
    rows=[];checks=[];saved_runs={}
    for seed in range(140,145):
        c,train,test,source_hash=source(seed)
        future,mean,std=standardized_successors(train)
        baseline=trials.load(OUT/'runs'/f'{seed}-sigreg-c0-n4000.pt')
        for label in LABELS:
            method=dict(zip(LABELS[:4],('sigreg','vicreg','infonce','quantile'))).get(label)
            path=OUT/'runs'/f'{seed}-{method}-c0-n4000.pt' if method else OUT/'hard-mining'/f'{seed}-c100.pt'
            r=trials.load(path)
            assert r['source_hash']==source_hash and r['plan_hash']==baseline['plan_hash']
            assert r['train_keys']==['action','x','xp']
            for k in r['initial']:
                torch.testing.assert_close(r['initial'][k],baseline['initial'][k],rtol=0,atol=0)
            model=B.Encoder();model.load_state_dict(r['model'])
            g=geometry(model,c,20001)
            assert g['folded']==r['geometry']['folded']
            assert abs(g['backward_fraction']-r['geometry']['backward_fraction'])<1e-6
            with torch.no_grad():
                zz=model(train['x']);center=zz.mean();norm=zz.std(unbiased=False).clamp_min(.05)
                decoder=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,6))
                decoder.load_state_dict(r['prediction']['decoder'])
                for name,data in [('native',test),('common',common_data(c))]:
                    code=(model(data['x'])-center)/norm
                    inp=torch.cat([code[:,None],torch.nn.functional.one_hot(data['action'],3).float()],1)
                    target=(torch.tanh(4*data['xp'])-mean)/std
                    mse=float((decoder(inp)-target).square().mean())
                    assert abs(mse-r['prediction'][name+'_mse'])<1e-8
            row=dict(seed=seed,label=label,
                **{k:v for k,v in g.items() if k not in ('q','y','slope')},
                **{k:v for k,v in r['prediction'].items() if k!='decoder'})
            rows.append(row);saved_runs[(seed,label)]=r
            checks.append(dict(file=str(path.relative_to(HERE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dense_grid_points=20001,paired_initialization_and_batches=True,prediction_recomputed=True))
    summary=[]
    for label in LABELS:
        group=[r for r in rows if r['label']==label]
        summary.append(dict(label=label,n=5,folded=sum(r['folded'] for r in group),
            material_folded=sum(r['material_fold'] for r in group),
            **{k:float(np.mean([r[k] for r in group])) for k in
               ('native_mse','common_mse','backward_fraction','affine_r2')}))
    base=[r for r in rows if r['label']==LABELS[0]]
    treated=[r for r in rows if r['label']==LABELS[-1]]
    pair=dict(prediction_improved=sum(t['native_mse']<b['native_mse'] for b,t in zip(base,treated)),
        folds_prevented=sum(b['material_fold'] and not t['material_fold'] for b,t in zip(base,treated)),
        folds_created=sum(not b['material_fold'] and t['material_fold'] for b,t in zip(base,treated)),
        native_geometric_mean_ratio=float(np.exp(np.mean([np.log(t['native_mse']/b['native_mse']) for b,t in zip(base,treated)]))))
    trials.atomic_json(OUT/'fresh-summary.json',dict(summary=summary,runs=rows,paired=pair,
        fixed_coefficient=100,fresh_policy_seeds=list(range(140,145)),no_reselection=True))
    trials.atomic_json(OUT/'fresh-verification.json',dict(count=25,runs=checks,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'fresh.py',HERE/'fresh_report.py',HERE/'run.py',HERE/'hard_mining.py')}))
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    colors=['#b94348','#8758a3','#345e91','#c69240','#147b79']
    axes[0].bar(range(5),[r['material_folded'] for r in summary],color=colors)
    display=('SIGReg','VICReg','InfoNCE','Quantiles','SIGReg + mining')
    axes[0].set(xticks=range(5),xticklabels=display,ylim=(0,5.7),ylabel='Models / 5',title='Fresh-source material folds')
    axes[0].tick_params(axis='x',labelrotation=20)
    for seed in range(140,145):
        for i,label in enumerate(LABELS):
            r=next(r for r in rows if r['seed']==seed and r['label']==label)
            axes[1].scatter(i,r['native_mse'],color=colors[i],s=22)
    axes[1].set(xticks=range(5),xticklabels=display,yscale='log',ylabel='Held-out next-observation MSE',title='Fresh sources: fixed hyperparameters')
    axes[1].tick_params(axis='x',labelrotation=20)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'07_fresh_sources.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(14,3.2),sharex=True)
    for ax,seed in zip(axes,range(140,145)):
        for label,color in ((LABELS[0],colors[0]),(LABELS[-1],colors[-1])):
            g=saved_runs[(seed,label)]['geometry'];q=g['q'];y=g['y']
            orient=1 if np.corrcoef(q,y)[0,1]>0 else -1
            ax.plot(np.exp(q),orient*(y-y.mean())/max(y.std(),1e-8),color=color,label=label)
        ax.set(title=f'Fresh source {seed}',xlabel='Distance (evaluation only)',ylabel='Standardized code')
    axes[0].legend(frameon=False,fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'08_fresh_code_curves.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Fresh-source fixed-hyperparameter check','',
        'Five newly trained ICM policy/trajectory sources, seeds 140--144. Global-mining coefficient 100 fixed before generating these sources; no retuning. Same original scalar MLP. Small validation in the same deterministic renderer, not a new environment family.','',
        '| Objective | Any folds | Material folds | Mean backward fraction | Held-out MSE | Coverage-stress MSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        lines.append(f"| {r['label']} | {r['folded']}/5 | {r['material_folded']}/5 | {r['backward_fraction']:.4f} | {r['native_mse']:.6f} | {r['common_mse']:.6f} |")
    lines+=['',f"Global-mining paired results vs SIGReg: {pair['prediction_improved']}/5 held-out prediction improvements; {pair['folds_prevented']} material folds prevented, {pair['folds_created']} created. Geometric-mean prediction MSE ratio {pair['native_geometric_mean_ratio']:.3f}.",'',
        'All final geometries and frozen-encoder decoder errors were independently recomputed from checkpoints. Counts are descriptive; five sources are too few to establish a general success rate. There is no hard no-fold guarantee.']
    (OUT/'FRESH.md').write_text('\n'.join(lines)+'\n')
    page=OUT/'index.html';text=page.read_text()
    for name in ('07_fresh_sources','08_fresh_code_curves'):
        text+=f'<h2>{name[3:].replace("_"," ")}</h2><img src="{name}.png">\n'
    page.write_text(text)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()

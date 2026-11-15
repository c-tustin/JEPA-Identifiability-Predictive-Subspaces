"""Check whether a stronger local gradient alone explains the mining result."""
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
from run import HERE,OUT,B,trials,source,geometry,standardized_successors,common_data


def main():
    torch.set_num_threads(1)
    rows=[];checks=[]
    for seed in range(130,140):
        path=OUT/'runs'/f'{seed}-sigreg-c100-n4000.pt'
        r=trials.load(path);b=trials.load(OUT/'runs'/f'{seed}-sigreg-c0-n4000.pt')
        c,train,test,source_hash=source(seed)
        assert r['source_hash']==source_hash and r['plan_hash']==b['plan_hash']
        for k in r['initial']:torch.testing.assert_close(r['initial'][k],b['initial'][k],rtol=0,atol=0)
        model=B.Encoder();model.load_state_dict(r['model'])
        g=geometry(model,c,20001)
        assert g['folded']==r['geometry']['folded']
        assert abs(g['backward_fraction']-r['geometry']['backward_fraction'])<1e-6
        _,mean,std=standardized_successors(train)
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
        rows.append(dict(seed=seed,**{k:v for k,v in g.items() if k not in ('q','y','slope')},
            **{k:v for k,v in r['prediction'].items() if k!='decoder'}))
        checks.append(dict(file=str(path.relative_to(HERE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            dense_grid_points=20001,prediction_recomputed=True,paired_controls=True))
    summary=dict(n=10,folded=sum(r['folded'] for r in rows),material_folded=sum(r['material_fold'] for r in rows),
        **{k:float(np.mean([r[k] for r in rows])) for k in ('backward_fraction','native_mse','common_mse')})
    trials.atomic_json(OUT/'strength-summary.json',dict(summary=summary,runs=rows))
    trials.atomic_json(OUT/'strength-verification.json',dict(count=10,runs=checks))
    original=json.loads((OUT/'summary.json').read_text())['summary']
    mined=json.loads((OUT/'mining-summary.json').read_text())['summary']
    groups=[next(r for r in original if r['method']=='sigreg' and r['coefficient']==0),
            next(r for r in original if r['method']=='sigreg' and r['coefficient']==1),
            summary,next(r for r in mined if r['coefficient']==100)]
    labels=['Baseline','Local 1','Local 100','Global 100']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,metric in zip(axes,('material_folded','native_mse')):
        ax.bar(range(4),[r[metric] for r in groups],color=['#b94348','#c69240','#8758a3','#147b79'])
        ax.set(xticks=range(4),xticklabels=labels)
    axes[0].set(ylabel='Models / 10',ylim=(0,10.8),title='Gradient-strength control: material folds')
    axes[1].set(ylabel='Held-out next-observation MSE',yscale='log',title='Prediction at the matched coefficient')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'09_strength_control.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Gradient-strength control','',
        'Ten development sources; original batch-local collision gradient raised to coefficient 100 without global mining. Same encoder initialization and base batches. This isolates auxiliary strength partly; global mining still has more observation exposure and compute.','',
        '| Method | Any folds | Material folds | Held-out MSE | Coverage-stress MSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for label,r in zip(labels,groups):
        lines.append(f"| {label} | {r['folded']}/10 | {r['material_folded']}/10 | {r['native_mse']:.6f} | {r['common_mse']:.6f} |")
    (OUT/'STRENGTH.md').write_text('\n'.join(lines)+'\n')
    page=OUT/'index.html';page.write_text(page.read_text()+'<h2>Gradient-strength control</h2><img src="09_strength_control.png">\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()

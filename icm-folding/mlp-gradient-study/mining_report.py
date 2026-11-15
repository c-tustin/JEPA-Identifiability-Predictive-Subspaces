"""Audit and summarize the adaptively chosen global-mining follow-up."""
import hashlib
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/icm-gradient-cache')
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import B,OUT,HERE,trials,source,geometry,standardized_successors,common_data
from hard_mining import mine


def main():
    torch.set_num_threads(1)
    rows=[];checks=[];curves={}
    main_rows=json.loads((OUT/'summary.json').read_text())['runs']
    for seed in range(130,140):
        c,train,test,source_hash=source(seed)
        future,mean,std=standardized_successors(train)
        baseline=trials.load(OUT/'runs'/f'{seed}-sigreg-c0-n4000.pt')
        for coefficient in (1.,10.,100.):
            path=OUT/'hard-mining'/f'{seed}-c{coefficient:g}.pt'
            r=trials.load(path)
            assert r['source_hash']==source_hash and r['plan_hash']==baseline['plan_hash']
            assert r['train_keys']==['action','x','xp']
            for k in r['initial']:
                torch.testing.assert_close(r['initial'][k],baseline['initial'][k],rtol=0,atol=0)
            model=B.Encoder();model.load_state_dict(r['model'])
            g=geometry(model,c,20001)
            assert g['folded']==r['geometry']['folded']
            assert abs(g['backward_fraction']-r['geometry']['backward_fraction'])<1e-6
            assert [f['step'] for f in r['frames']]==[0,100,250,500,1000,2000,4000]
            with torch.no_grad():
                ids,js,gaps,scale=mine(model,train,future)
                assert torch.all(train['action'][ids]==train['action'][js])
                zz=model(train['x'])
                violations=(gaps-(zz[ids]-zz[js]).abs()/scale).clamp_min(0)
                center=zz.mean();norm=zz.std(unbiased=False).clamp_min(.05)
                decoder=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,6))
                decoder.load_state_dict(r['prediction']['decoder'])
                for label,data in [('native',test),('common',common_data(c))]:
                    code=(model(data['x'])-center)/norm
                    inp=torch.cat([code[:,None],torch.nn.functional.one_hot(data['action'],3).float()],1)
                    target=(torch.tanh(4*data['xp'])-mean)/std
                    score=float((decoder(inp)-target).square().mean())
                    assert abs(score-r['prediction'][label+'_mse'])<1e-8
            rows.append(dict(seed=seed,coefficient=coefficient,
                **{k:v for k,v in g.items() if k not in ('q','y','slope')},
                **{k:v for k,v in r['prediction'].items() if k!='decoder'},
                max_mined_gap_violation=float(violations.max()),
                mean_mined_gap_violation=float(violations.mean())))
            curves[(seed,coefficient)]=g
            checks.append(dict(file=str(path.relative_to(HERE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dense_grid_points=20001,paired_model_and_base_batches=True,
                prediction_recomputed=True,mined_actions_match=True))
    summary=[]
    for coeff in (0.,1.,10.,100.):
        group=([r for r in main_rows if r['method']=='sigreg' and r['coefficient']==0.] if coeff==0 else
            [r for r in rows if r['coefficient']==coeff])
        summary.append(dict(coefficient=coeff,n=len(group),folded=sum(r['folded'] for r in group),
            material_folded=sum(r['material_fold'] for r in group),
            **{k:float(np.mean([r[k] for r in group])) for k in ('native_mse','common_mse','backward_fraction','affine_r2')},
            prediction_improved=sum(r['native_mse']<next(b for b in main_rows if b['seed']==r['seed'] and b['method']=='sigreg' and b['coefficient']==0)['native_mse'] for r in group)))
    trials.atomic_json(OUT/'mining-summary.json',dict(summary=summary,runs=rows,
        adaptive=True,coefficients_developed_on_historical_sources=True))
    trials.atomic_json(OUT/'mining-verification.json',dict(count=len(checks),runs=checks,
        code_sha256=hashlib.sha256((HERE/'hard_mining.py').read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256((HERE/'mining_report.py').read_bytes()).hexdigest()))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(range(4),[r['material_folded'] for r in summary],color=['#b94348','#c69240','#669b94','#147b79'])
    axes[0].set(xticks=range(4),xticklabels=['Baseline','Mine 1','Mine 10','Mine 100'],ylim=(0,10.8),ylabel='Models / 10',title='Material folds with global mining')
    for seed in range(130,140):
        base=next(r for r in main_rows if r['seed']==seed and r['method']=='sigreg' and r['coefficient']==0)
        values=[base['native_mse']]+[next(r for r in rows if r['seed']==seed and r['coefficient']==c)['native_mse'] for c in (1.,10.,100.)]
        axes[1].plot(range(4),values,'o-',ms=3,alpha=.6,lw=1)
    axes[1].set(xticks=range(4),xticklabels=['Baseline','Mine 1','Mine 10','Mine 100'],yscale='log',ylabel='Held-out next-observation MSE',title='Prediction: same sources, exploratory coefficients')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'05_global_mining.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(14,5.8),sharex=True)
    for ax,seed in zip(axes.flat,range(130,140)):
        b=trials.load(OUT/'runs'/f'{seed}-sigreg-c0-n4000.pt')['geometry']
        for label,g,color in [('SIGReg',b,'#b94348'),('Global mining 100',curves[(seed,100.)],'#147b79')]:
            q,y=g['q'],g['y'];orient=1 if np.corrcoef(q,y)[0,1]>0 else -1
            ax.plot(np.exp(q),orient*(y-y.mean())/max(y.std(),1e-8),color=color,label=label)
        ax.set(title=f'Source {seed}',xlabel='Distance (evaluation only)',ylabel='Standardized code')
    axes.flat[0].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'06_mining_code_curves.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Adaptive global transition-collision mining','',
        'Same scalar MLP, base SIGReg objective, random initialization and base batches. Mining and coefficients were chosen adaptively after inspecting development results. Extra computation/observation exposure is not matched. This is not a confirmatory ten-seed generalization test.','',
        '| Global mining coefficient | Any folds | Material folds | Mean backward fraction | Held-out MSE | Coverage-stress MSE | Held-out improvements vs baseline |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        lines.append(f"| {r['coefficient']:g} | {r['folded']}/10 | {r['material_folded']}/10 | {r['backward_fraction']:.4f} | {r['native_mse']:.6f} | {r['common_mse']:.6f} | {r['prediction_improved']}/10 |")
    lines+=['','No geometry labels or dynamics equations enter mining or the encoder gradient. Development did inspect geometry diagnostics. Zero observed folds is not an all-input guarantee; dense checks cover this simulator curve. Same-action successor differences may include nuisance or stochastic variation in other environments.']
    (OUT/'MINING.md').write_text('\n'.join(lines)+'\n')
    page=OUT/'index.html'
    text=page.read_text()
    for name in ('05_global_mining','06_mining_code_curves'):
        text+=f'<h2>{name[3:].replace("_"," ")}</h2><img src="{name}.png">\n'
    page.write_text(text)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()

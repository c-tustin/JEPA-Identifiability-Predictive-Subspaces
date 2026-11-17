"""Reload every final encoder; compare fold development and output variation."""
from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def samples(run,key):
    if key.startswith('average_'):return [h['averages'][key.removeprefix('average_')] for h in run['history']]
    return [h[key] for h in run['history']]
def label(run,key):
    if run['mode']=='ordinary':return 'Live original' if key=='student' else 'Weight EMA '+key.removeprefix('average_')
    return f"Target EMA {run['beta']:g}: {key}"
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def main():
    torch.set_num_threads(1)
    paths=sorted(OUT.glob('*.pt'));assert len(paths)==31,f'Expected 31 completed fits; found {len(paths)}'
    runs=[old.trials.load(p) for p in paths];lookup={(r['seed'],r['mode'],r['beta']):r for r in runs}
    checks=[];rows=[]
    for path,r in zip(paths,runs):
        assert r['source_sha256']==sha(old.ORIGINAL/'sources'/f"icm-{r['seed']}.pt")
        assert r['code_sha256']==sha(HERE/'run.py') and r['protocol_sha256']==sha(HERE/'PROTOCOL.md')
        ordinary=lookup[r['seed'],'ordinary',None]
        for k in ('source_sha256','alignment_sha256','regularizer_sha256'):assert r[k]==ordinary[k]
        assert all(torch.equal(v,ordinary['initial'][k]) for k,v in r['initial'].items())
        c=old.trials.behavior.Config(**r['config'])
        for key,state in r['states'].items():
            model=old.ScaledEncoder(4);model.load_state_dict(state);fresh=geometry(model,c);stored=r['final'][key]
            for k in ('q','y','slope'):np.testing.assert_array_equal(fresh[k],stored[k])
            for k in compact(fresh):assert fresh[k]==stored[k]
            history=samples(r,key);first=next((h['step'] for h,g in zip(r['history'],history) if g['folded']),None)
            rows.append(dict(seed=r['seed'],label=label(r,key),**compact(fresh),ever_folded=any(g['folded'] for g in history),first_sampled_fold=first))
        if r['mode']=='ordinary' or r['beta']==0:
            baseline=old.trials.load(base.OUT/'runs'/f"native-ordinary-{r['seed']}.pt")
            assert all(torch.equal(v,baseline['encoder'][k]) for k,v in r['states']['student'].items())
        checks.append(dict(file=path.name,sha256=sha(path),final_curves_slopes_exact=True,paired_initialization_plans_source=True,baseline_parameter_error=r['baseline_parameter_error']))
    summary=[]
    for condition in dict.fromkeys(row['label'] for row in rows):
        group=[row for row in rows if row['label']==condition]
        summary.append(dict(condition=condition,n=len(group),final_folds=sum(x['folded'] for x in group),material_folds=sum(x['material_fold'] for x in group),ever_folded=sum(x['ever_folded'] for x in group),mean_r2=float(np.mean([x['r2'] for x in group])),min_std=float(min(x['std'] for x in group)),max_std=float(max(x['std'] for x in group))))
    old.trials.atomic_json(OUT/'summary.json',dict(summary=summary,runs=rows))
    old.trials.atomic_json(OUT/'verification.json',dict(final_grid=20001,dtype='float64',baseline_exact=10,zero_lag_full_run_exact=True,checks=checks))
    lines=['# EMA and folding: measured results','','Ten original gain-4 ICM sources, seeds 130–139; unchanged scalar MLP, native occupancy, 4,000 updates. Parameter averaging for evaluation and EMA targets in training are distinct interventions. Historical diagnostic cohort; no new actors.','','| Encoder | Final folds | Material folds | Any sampled training fold | Mean log-distance R² | Min output std |','| --- | --- | --- | --- | --- | --- |']
    for g in summary:lines.append(f"| {g['condition']} | {g['final_folds']}/{g['n']} | {g['material_folds']}/{g['n']} | {g['ever_folded']}/{g['n']} | {g['mean_r2']:.4f} | {g['min_std']:.4f} |")
    lines+=['','All final geometries were independently recomputed after reload on 20,001 float64 grid points. Original live reruns reproduce all ten prior checkpoints bit-for-bit. The full zero-lag target run (seed 131) also matches its original checkpoint exactly. Formula checks verify target detachment, EMA recurrence and the gradient-matched alignment control.','', 'Training fold histories are sampled every 25 updates on 1,025 points. "Any sampled training fold" includes initialization. A final ordered EMA may have folded earlier; averaging need not repair a live fold or preserve monotonicity. Output variation and R² help identify stale/weak representations but do not establish downstream prediction utility.','', 'This EMA-target experiment uses the original scalar alignment model without a predictor, keeps SIGReg on student endpoints, and uses the symmetric sum alignment to match the original gradient at zero lag. It is not a full BYOL/image-JEPA implementation. No learning-rate retuning, gain-8 stress test, fresh-seed validation, or direct continuation from existing folded models is included. No analytic no-fold guarantee is claimed.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    # All native seeds: dashed averages versus unchanged live training.
    fig,axes=plt.subplots(2,5,figsize=(17,7))
    colors={'student':'#c13c45','average_0.9':'#ba8a25','average_0.99':'#377eb8','average_0.999':'#147b79'}
    for ax,seed in zip(axes.flat,range(130,140)):
        r=lookup[seed,'ordinary',None]
        for key,g in r['final'].items():
            sign=1 if np.cov(g['q'],g['y'])[0,1]>=0 else -1
            ax.plot(np.exp(g['q']),sign*(g['y']-g['y'].mean())/g['std'],color=colors[key],ls='-' if key=='student' else '--',label=key)
        ax.set(title=f'Seed {seed}',xlabel='Distance',ylabel='Standardized encoder')
    axes[0,0].legend(fontsize=7);fig.suptitle('EMA of weights for evaluation: does smoothing remove the live encoder fold?');save(fig,'weight_averages')
    fig,axes=plt.subplots(2,5,figsize=(17,7))
    for ax,seed in zip(axes.flat,range(130,140)):
        baseline=lookup[seed,'ordinary',None]['final']['student']
        for g,text,color,style in [(baseline,'Original','#777777',':')]+[(lookup[seed,'target',b]['final'][key],f'{key}, beta {b}',('#c13c45' if b==.99 else '#147b79'),('-' if key=='student' else '--')) for b in (.99,.999) for key in ('student','target')]:
            sign=1 if np.cov(g['q'],g['y'])[0,1]>=0 else -1
            ax.plot(np.exp(g['q']),sign*(g['y']-g['y'].mean())/g['std'],label=text,color=color,ls=style)
        ax.set(title=f'Seed {seed}',xlabel='Distance',ylabel='Standardized encoder')
    axes[0,0].legend(fontsize=7);fig.suptitle('EMA targets during training: live students and their moving-average targets');save(fig,'ema_targets')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ordinary=[r for r in runs if r['mode']=='ordinary']
    for key in ('student','average_0.9','average_0.99','average_0.999'):
        time=[h['step'] for h in ordinary[0]['history']]
        counts=np.sum([[g['folded'] for g in samples(r,key)] for r in ordinary],axis=0)
        axes[0].plot(time,counts,label=label(ordinary[0],key),color=colors[key])
    for b in (.99,.999):
        group=[r for r in runs if r['mode']=='target' and r['beta']==b]
        for key,style in (('student','-'),('target','--')):
            counts=np.sum([[g['folded'] for g in samples(r,key)] for r in group],axis=0)
            axes[1].plot(time,counts,ls=style,label=f'{key}, beta {b}')
    for ax in axes:ax.set(xlabel='Training update',ylabel='Folded encoders / 10',ylim=(-.2,10.2));ax.legend(fontsize=8)
    axes[0].set_title('Evaluation averages');axes[1].set_title('Targets alter training')
    fig.suptitle('Do folds develop during training? Sampled every 25 updates');save(fig,'fold_development')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

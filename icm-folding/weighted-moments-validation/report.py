"""Independent dense checkpoint checks and separate prevention/repair tables."""
from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
METRICS=('folded','material_fold','roots','backward_fraction','std','r2')
def group_summary(name,group):
    return dict(condition=name,n=len(group),folded=sum(r['final']['folded'] for r in group),
        material_folds=sum(r['final']['material_fold'] for r in group),mean_r2=float(np.mean([r['final']['r2'] for r in group])),
        min_std=float(min(r['final']['std'] for r in group)))
def main():
    torch.set_num_threads(1)
    runs=[];checks=[]
    for path in sorted(OUT.glob('*.pt')):
        r=old.trials.load(path)
        model=old.ScaledEncoder(r['gain']);model.load_state_dict(r['encoder'])
        c=old.trials.behavior.Config(**r['config']);g=geometry(model,c)
        for key in ('q','y','slope'):np.testing.assert_array_equal(g[key],r['final'][key])
        assert all(g[k]==r['final'][k] for k in METRICS)
        assert r['source_hash']==sha(FRESH/f"icm-{r['seed']}.pt")
        if r['stage']=='repair':
            startpath=OUT/f"scratch-g4-ordinary-{r['seed']}.pt";start=old.trials.load(startpath)
            assert r['start_hash']==sha(startpath)
            assert all(torch.equal(v,start['encoder'][k]) for k,v in r['initial'].items())
        checks.append(dict(file=path.name,sha256=sha(path),dense_recomputation_exact=True))
        runs.append(r)
    lookup={(r['stage'],r['gain'],r['method'],r['seed']):r for r in runs}
    for gain in (4,8):
        for seed in range(140,145):
            a=lookup['scratch',gain,'ordinary',seed];b=lookup['scratch',gain,'weighted',seed]
            for k in ('source_hash','alignment_hash','regularizer_hash'):assert a[k]==b[k]
            assert all(torch.equal(v,b['initial'][k]) for k,v in a['initial'].items())
    repair_seeds=[s for s in range(140,145) if lookup['scratch',4,'ordinary',s]['final']['folded']]
    for seed in repair_seeds:
        group=[lookup['repair',4,m,seed] for m in ('ordinary','weighted','weighted_reset')]
        assert len({r['alignment_hash'] for r in group})==len({r['regularizer_hash'] for r in group})==1
    assert len(runs)==20+3*len(repair_seeds)
    summary=[]
    for gain in (4,8):
        for method in ('ordinary','weighted'):
            summary.append(group_summary(f'gain {gain}, {method}, initialization',[r for r in runs if r['stage']=='scratch' and r['gain']==gain and r['method']==method]))
    if repair_seeds:
        for method in ('ordinary','weighted','weighted_reset'):
            summary.append(group_summary(f'gain 4, {method}, existing fold',[r for r in runs if r['stage']=='repair' and r['method']==method]))
    historical=[]
    # Historical checkpoints may be omitted from the portable snapshot. Preserve
    # the original re-audit if present, rather than pretending to rerun it.
    historical_paths=[]
    for seed in range(130,140):
        historical_paths.append(('gain 4 prevention',4,base.OUT/'runs'/f'native-weighted_moments-{seed}.pt'))
        historical_paths.append(('gain 8 prevention',8,HERE.parent/'fold-repair/output/runs'/f'scratch-g8-weighted-{seed}.pt'))
    for seed in (131,132,133,136,139):
        for method in ('weighted','weighted_reset'):
            historical_paths.append((f'late repair {method}',4,HERE.parent/'fold-repair/output/runs'/f'repair-g4-{method}-{seed}.pt'))
    if all(p.exists() for _,_,p in historical_paths):
        for label,gain,path in historical_paths:
            r=old.trials.load(path);model=old.ScaledEncoder(gain);model.load_state_dict(r['encoder'])
            c=old.trials.behavior.Config(**r['config']);g=geometry(model,c)
            historical.append(dict(condition=label,file=path.name,sha256=sha(path),**{k:g[k] for k in METRICS}))
        old.trials.atomic_json(OUT/'historical-reaudit.json',dict(points=20001,dtype='float64',runs=historical))
    elif (OUT/'historical-reaudit.json').exists():
        historical=json.loads((OUT/'historical-reaudit.json').read_text())['runs']
    old.trials.atomic_json(OUT/'verification.json',dict(points=20001,dtype='float64',checks=checks,paired_initialization_and_plans=True,repair_starts_exact=True))
    old.trials.atomic_json(OUT/'summary.json',dict(summary=summary,repair_seeds=repair_seeds,
        runs=[dict(seed=r['seed'],gain=r['gain'],method=r['method'],stage=r['stage'],**{k:r['final'][k] for k in METRICS}) for r in runs]))
    lines=['# Weighted-moment validation','','Five recently generated ICM sources (140–144), new to occupancy weighting. All hyperparameters fixed before this experiment. Known distance is used for occupancy weights.','','| Condition | Final folds | Material folds | Mean log-distance R² | Minimum output std |','| --- | --- | --- | --- | --- |']
    for g in summary:lines.append(f"| {g['condition']} | {g['folded']}/{g['n']} | {g['material_folds']}/{g['n']} | {g['mean_r2']:.4f} | {g['min_std']:.4f} |")
    lines+=['','All new checkpoints passed independent float64 20,001-point curve/derivative recomputation; paired initialization, training plans and direct-repair starting weights verified. No encoder reinitialization is counted as repair.','',f'New direct-repair starts: {repair_seeds}. Initial training: 4,000 updates; continuations: 8,000 updates at 0.0001. Gain-4 and gain-8 results are not pooled.','', '## Historical weighted-moment re-audit','', '| Historical condition | Final folds |','| --- | --- |']
    for label in dict.fromkeys(r['condition'] for r in historical):
        gg=[r for r in historical if r['condition']==label];lines.append(f"| {label} | {sum(r['folded'] for r in gg)}/{len(gg)} |")
    lines+=['','Historical curves were re-evaluated on the same float64 grid, not retrained. On a portable checkout without historical checkpoints, `historical-reaudit.json` is preserved evidence, not a new re-audit. Finite-grid checks are not continuous-domain certificates. Standard-gain prevention does not establish consistent direct repair or robustness to gain changes. No downstream prediction improvement is asserted.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(2,5,figsize=(17,7))
    for row,gain in enumerate((4,8)):
        for col,seed in enumerate(range(140,145)):
            for method,color in (('ordinary','#c13c45'),('weighted','#147b79')):
                r=lookup['scratch',gain,method,seed];g=r['final'];y=g['y'];sign=1 if np.cov(g['q'],y)[0,1]>=0 else -1
                axes[row,col].plot(np.exp(g['q']),sign*(y-y.mean())/y.std(),color=color,label=method)
            axes[row,col].set(title=f'Gain {gain}, source {seed}',xlabel='Distance',ylabel='Standardized encoder')
    axes[0,0].legend();fig.suptitle('Weighted moments: paired standard-gain and stress tests on five recent ICM sources')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'prevention.png',dpi=160,bbox_inches='tight');fig.savefig(OUT/'prevention.pdf',bbox_inches='tight');plt.close(fig)
    if repair_seeds:
        fig,axes=plt.subplots(1,len(repair_seeds),figsize=(6*len(repair_seeds),4),squeeze=False)
        for ax,seed in zip(axes[0],repair_seeds):
            start=lookup['scratch',4,'ordinary',seed]['final'];scale=start['std'];mean=start['y'].mean()
            sign=1 if np.cov(start['q'],start['y'])[0,1]>=0 else -1
            ax.plot(np.exp(start['q']),sign*(start['y']-mean)/scale,'k:',label='Starting fold')
            for method in ('ordinary','weighted','weighted_reset'):
                g=lookup['repair',4,method,seed]['final'];ax.plot(np.exp(g['q']),sign*(g['y']-mean)/scale,label=method)
            ax.set(title=f'Source {seed}',xlabel='Distance',ylabel='Encoder (fixed starting scale)');ax.legend(fontsize=8)
        fig.suptitle('Direct continuation from the exact folded encoder: no model reinitialization')
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'repair.png',dpi=160,bbox_inches='tight');fig.savefig(OUT/'repair.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

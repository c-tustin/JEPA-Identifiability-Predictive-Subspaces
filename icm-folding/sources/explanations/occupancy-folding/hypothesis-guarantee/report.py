"""Check the hard guarantees and summarize the mechanism interventions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from guaranteed_models import OUT,HERE,COMPARISON,old,dataset,target_measure,evaluation,PositiveSpline,ProjectedSpline,MARGIN,name,key
from actor_geometry import B,ORIGINAL_SENSOR,sensor_for,diagnostics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RED='#c13c45';TEAL='#147b79';BLUE='#345e91';ORANGE='#d48a18';PURPLE='#8758a3'


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def scalar_evaluation(e):
    return {k:v for k,v in e.items() if k not in ('q','y','slope')}


def check_models():
    files=list((OUT/'guaranteed').glob('*.pt'));assert len(files)==100
    rows=[];checks=[];free_models={}
    for seed in range(130,140):
        ds=dataset(seed)
        for level,tail in ((None,None),(.875,'one'),(.875,'two'),(.9375,'one'),(.9375,'two')):
            target=ds['p'] if level is None else target_measure(ds,level,tail)
            free_file=COMPARISON/'runs'/f'{key(seed,"native_paired" if level is None else "weighted",level,tail)}.pt'
            free=old.trials.load(free_file);fm=old.ScaledEncoder(4).double();fm.load_state_dict(free['encoder'])
            fe=evaluation(fm,ds,target,seed);free_models[(seed,level,tail)]=fe
            rows.append(dict(seed=seed,method='free',level=level,tail=tail,**scalar_evaluation(fe)))
            for method,cls in (('positive',PositiveSpline),('projected',ProjectedSpline)):
                r=old.trials.load(OUT/'guaranteed'/f'{name(seed,method,level,tail)}.pt')
                m=cls();m.load_state_dict(r['model'])
                knots=m.knots.detach().numpy();values=m.values().detach().numpy();slopes=np.diff(values)/np.diff(knots)
                assert np.all(slopes>=MARGIN-1e-10) and np.all(np.diff(values)>0)
                assert r['minimum_training_slope']>=MARGIN-1e-10
                assert len(r['history'])==40 and r['history'][-1]['step']==4000
                e=evaluation(m,ds,target,seed)
                for field in ('population_objective','batch_objective_mean','population_alignment','population_sigreg'):
                    assert abs(e[field]-r['evaluation'][field])<1e-10,(seed,method,field)
                np.testing.assert_array_equal(e['y'],r['evaluation']['y'])
                assert not e['geometry']['folded'] and np.all(e['slope']<0)
                assert np.min(-e['slope'])>=MARGIN/16-1e-10
                rows.append(dict(seed=seed,method=method,level=level,tail=tail,**scalar_evaluation(e),
                    minimum_segment_slope=float(slopes.min()),minimum_training_slope=r['minimum_training_slope']))
                checks.append(dict(seed=seed,method=method,level=level,tail=tail,
                    continuous_interval_certificate=True,independent_metrics=True))
    old.trials.save_torch(OUT/'free-evaluations.pt',free_models)
    summaries=[]
    for level in (None,.875,.9375):
        for method in ('free','positive','projected'):
            group=[r for r in rows if r['level']==level and r['method']==method]
            summaries.append(dict(level=level,method=method,n=len(group),
                folded=sum(r['geometry']['folded'] for r in group),
                batch_objective_mean=float(np.mean([r['batch_objective_mean'] for r in group])),
                population_objective_mean=float(np.mean([r['population_objective'] for r in group])),
                mean_r2=float(np.mean([r['geometry']['r2'] for r in group])),
                min_std=min(r['geometry']['std'] for r in group)))
    comparisons=[]
    for level in (.875,.9375):
        better_batch=better_population=0
        for r in rows:
            if r['method']!='projected' or r['level']!=level:continue
            f=next(f for f in rows if f['method']=='free' and f['seed']==r['seed'] and f['level']==level and f['tail']==r['tail'])
            better_batch+=r['batch_objective_mean']<f['batch_objective_mean']
            better_population+=r['population_objective']<f['population_objective']
        comparisons.append(dict(level=level,n=20,projected_lower_batch_loss=better_batch,projected_lower_population_loss=better_population))
    old.trials.atomic_json(OUT/'model-summary.json',dict(summary=summaries,comparisons=comparisons,runs=rows))
    return rows,summaries,checks,free_models


def check_actors():
    assert len(list((OUT/'actors').glob('*.pt')))==21
    actors=[];checks=[]
    for seed in range(130,140):
        original=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt');c=B.Config(**original['config'])
        for kind in ('native','mirror','equalized'):
            if kind=='native':saved=original
            else:saved=old.trials.load(OUT/'actors'/f'{kind}-{seed}.pt')
            policy,icm=B.Policy(),B.ICM();policy.load_state_dict(saved['policy']);icm.load_state_dict(saved['icm'])
            sensor=sensor_for(kind);prior=B.sensor;B.sensor=sensor
            try:final=diagnostics(policy,icm,c,sensor,seed,256)
            finally:B.sensor=prior
            if kind!='native':
                np.testing.assert_array_equal(final['distances'],saved['final']['distances'])
                np.testing.assert_array_equal(final['drift'],saved['final']['drift'])
            else:
                d=(4*torch.exp(c.sigma*original['data']['z'])).numpy()
                np.testing.assert_array_equal(final['distances'],d)
            reward=saved['history'][-1]['intrinsic_reward']
            actors.append(dict(seed=seed,kind=kind,final=final,final_training_reward=reward))
            checks.append(dict(seed=seed,kind=kind,physical_distance_and_policy_recomputed=True))
    gate=old.trials.load(OUT/'actors/native-131.pt');assert max(gate['reproduction'].values())==0
    summaries=[]
    for kind in ('native','mirror','equalized'):
        g=[r for r in actors if r['kind']==kind]
        summaries.append(dict(kind=kind,n=10,mean_distance=float(np.mean([r['final']['mean_distance'] for r in g])),
            fraction_below2=float(np.mean([r['final']['fraction_below2'] for r in g])),
            fraction_above15=float(np.mean([r['final']['fraction_above15'] for r in g])),
            typical_median_distance=float(np.mean([r['final']['median_distance'] for r in g])),
            final_training_reward=float(np.mean([r['final_training_reward'] for r in g]))))
    old.trials.atomic_json(OUT/'actor-summary.json',dict(summary=summaries,
        runs=[dict(seed=r['seed'],kind=r['kind'],**{k:r['final'][k] for k in ('mean_distance','median_distance','fraction_below2','fraction_above15')},final_training_reward=r['final_training_reward']) for r in actors]))
    old.trials.save_torch(OUT/'actor-evaluations.pt',actors)
    return actors,summaries,checks


def check_paths():
    traces=[old.trials.load(p) for p in sorted((OUT/'paths').glob('trace-*.pt'))];assert len(traces)==4
    switches=[old.trials.load(p) for p in sorted((OUT/'paths').glob('switch-*.pt'))];assert len(switches)==6
    checks=[];births=[]
    for r in traces:
        assert r['reproduction_max_error']==0 and r['max_update_error']<1e-6
        for s in r['snapshots']:
            actual=s['after'];before=s['before'];directions=s['directions']
            error=max(float((actual[n]-before[n]-sum(directions[t][n] for t in directions)).abs().max()) for n in before)
            assert error<1e-6
            if s['event']:
                births.append(dict(level=r['level'],tail=r['tail'],step=s['step'],
                    far_slope_before=float(s['probes']['before']['slope'][-1]),
                    far_slope_after=float(s['probes']['actual']['slope'][-1]),
                    near_contribution=s['linearized_slope_changes']['near'][1],
                    far_contribution=s['linearized_slope_changes']['far'][1],
                    alignment_contribution=s['linearized_slope_changes']['alignment'][1]))
        checks.append(dict(type='trace',level=r['level'],tail=r['tail'],original_reproduction=True,adam_component_reconstruction=True))
    switch_rows=[]
    for r in switches:
        origin=next(t for t in traces if t['level']==r['old_level'] and t['tail']=='one')
        assert all(torch.equal(v,origin['starts'][r['at']]['encoder'][n]) for n,v in r['start_encoder'].items())
        f=r['frames'][-1];model=old.ScaledEncoder(4).double();model.load_state_dict(r['encoder'])
        q=torch.from_numpy(f['q']).requires_grad_();y=model(old.trials.behavior.sensor(q.exp(),dataset(131)['c']))
        np.testing.assert_array_equal(y.detach().numpy(),f['y'])
        ds=dataset(131);loss=evaluation(model,ds,target_measure(ds,r['new_level'],'one'),131)
        if r.get('reset',False):assert not r['start_optimizer']['state']
        switch_rows.append(dict(old_level=r['old_level'],new_level=r['new_level'],at=r['at'],reset=r.get('reset',False),
            folded=f['folded'],roots=f['roots'],r2=f['r2'],batch_objective=loss['batch_objective_mean'],population_objective=loss['population_objective']))
        checks.append(dict(type='switch',old_level=r['old_level'],new_level=r['new_level'],at=r['at'],reset=r.get('reset',False),same_starting_model=True,final_curve_recomputed=True))
    old.trials.atomic_json(OUT/'path-summary.json',dict(births=births,switches=switch_rows,
        traces=[dict(level=r['level'],tail=r['tail'],events=r['events'],reproduction_error=r['reproduction_max_error'],
            gradient_relative_error=r['max_gradient_relative_error'],moment_roundoff=r['max_moment_error'],step_error=r['max_update_error']) for r in traces]))
    return traces,switches,births,switch_rows,checks


def normalize(q,y):
    sign=1 if np.cov(q,y)[0,1]>=0 else -1
    return sign*(y-y.mean())/y.std()


def plots(rows,actors,traces,switches,free_models):
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    color={'native':RED,'mirror':BLUE,'equalized':TEAL};labels={'native':'Original perspective','mirror':'Mirrored perspective','equalized':'Equal visual step size'}
    bins=np.linspace(1,16,61);centers=(bins[:-1]+bins[1:])/2
    for kind in color:
        g=[r for r in actors if r['kind']==kind]
        hist=np.mean([np.histogram(r['final']['distances'],bins)[0]/len(r['final']['distances']) for r in g],axis=0)
        grid=g[0]['final']['grid'];response=np.mean([r['final']['visual_response'] for r in g],axis=0)
        interior=(grid>=1.25)&(grid<=15.75)
        axes[0].plot(grid[interior],response[interior],c=color[kind],label=labels[kind],lw=2)
        axes[1].plot(centers,hist,c=color[kind],label=labels[kind],lw=2)
        axes[2].plot(grid,np.mean([r['final']['drift'] for r in g],axis=0),c=color[kind],lw=2)
    axes[0].set(yscale='log',title='Visual change produced by a physical step',ylabel='Mean squared neural-input change',xlabel='Physical distance')
    axes[1].set(title='Where the trained agent spends time',ylabel='Probability per 0.25-distance bin',xlabel='Physical distance')
    axes[2].set(title='Learned direction of movement',ylabel='Mean distance change per action',xlabel='Physical distance');axes[2].axhline(0,c='black',lw=.8)
    axes[0].legend(fontsize=8)
    fig.suptitle('ICM preference follows the observation geometry · averages over ten seeds',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'01_geometry_and_actor_preference')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,new in zip(axes,(.9375,.875)):
        baseline=next(r for r in traces if r['level']==new and r['tail']=='one')['frames'][-1]
        ax.plot(np.exp(baseline['q']),normalize(baseline['q'],baseline['y']),c='black',lw=2,label=f'From start: {100*new:g}%')
        for r in switches:
            if r['new_level']!=new:continue
            f=r['frames'][-1];label=f'Switch at {r["at"]}'+(' + reset Adam' if r.get('reset') else '')
            c=TEAL if r['at']==100 else ORANGE if not r.get('reset') else PURPLE
            ax.plot(np.exp(f['q']),normalize(f['q'],f['y']),c=c,label=label,lw=1.6)
        ax.set(title=f'Final target: {100*new:g}% below 2',xlabel='Physical distance',ylabel='Final code (display scale)');ax.legend(fontsize=8)
    fig.suptitle('Source 131: the same final sampling measure can leave different shapes after different early histories',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'02_distribution_switches')
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    selections=[(.875,'one','event'),(.875,'two','event'),(.9375,'one',250)]
    for ax,(level,tail,selector) in zip(axes,selections):
        r=next(r for r in traces if r['level']==level and r['tail']==tail)
        s=next(s for s in r['snapshots'] if (s['event'] if selector=='event' else s['step']==selector))
        values=[s['linearized_slope_changes'][t][1] for t in ('alignment','near','far')]
        actual=float(s['probes']['actual']['slope'][-1]-s['probes']['before']['slope'][-1]);values.append(actual)
        ax.bar(['Alignment','Near samples','Far samples','Actual step'],values,color=['gray',ORANGE,BLUE,'black'])
        ax.axhline(0,c='black',lw=.7);ax.tick_params(axis='x',labelrotation=25)
        ax.set(title=f'{100*level:g}%, {tail} bump(s), update {s["step"]}',ylabel='Change in far-distance slope')
    fig.suptitle('Near-sample gradients deform the far branch at fold birth\nSource 131. First three bars are linearized Adam-component effects; black is the actual finite change',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.85));save(fig,'03_near_far_gradient_balance')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    free=free_models[(131,.875,'one')]
    axes[0].plot(np.exp(free['q']),normalize(free['q'],free['y']),c=RED,label='Unconstrained encoder',lw=2)
    for method,cls,c in (('positive',PositiveSpline,BLUE),('projected',ProjectedSpline,TEAL)):
        r=old.trials.load(OUT/'guaranteed'/f'{name(131,method,.875,"one")}.pt');e=r['evaluation'];m=cls();m.load_state_dict(r['model'])
        axes[0].plot(np.exp(e['q']),normalize(e['q'],e['y']),c=c,label=method.capitalize(),lw=2)
        knots=m.knots.numpy();slopes=np.diff(m.values().detach().numpy())/np.diff(knots)
        axes[1].stairs(slopes,knots,color=c,label=method.capitalize(),lw=2)
    axes[0].set(title='Free and guaranteed representations',xlabel='Physical distance',ylabel='Learned code (display scale)');axes[0].legend(fontsize=8)
    axes[1].axhline(MARGIN,c='black',ls='--',label='Required slope floor 0.05')
    axes[1].set(title='Analytic slopes on every spline segment',xlabel='Observed apparent height = 1 / distance',ylabel='Code slope with apparent height',yscale='log');axes[1].legend(fontsize=8)
    fig.suptitle('Positive segment slopes certify no folds between sample points',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'04_hard_no_fold_guarantee')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,level in zip(axes,(.875,.9375)):
        for seed in range(130,140):
            for tail in ('one','two'):
                values=[next(r for r in rows if r['seed']==seed and r['tail']==tail and r['level']==level and r['method']==m)['batch_objective_mean'] for m in ('free','positive','projected')]
                ax.plot(range(3),values,c='gray',alpha=.25,lw=.8)
        for i,(method,c) in enumerate((('free',RED),('positive',BLUE),('projected',TEAL))):
            vals=[r['batch_objective_mean'] for r in rows if r['method']==method and r['level']==level]
            ax.scatter(np.full(len(vals),i),vals,c=c,s=15,alpha=.7)
        ax.set(xticks=range(3),xticklabels=['Free MLP','Positive slopes','Projected values'],yscale='log',title=f'{100*level:g}% near concentration',ylabel='Matched diagnostic objective')
    fig.suptitle('An explicit ordered solution can fit the loss better than the observed folded solution\nPaired lines: same source and target; model class and optimization geometry differ',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.85));save(fig,'05_objective_comparison')


def main():
    torch.set_num_threads(1);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows,models,mc,free=check_models();actors,actor_summary,ac=check_actors();traces,switches,births,switch_summary,pc=check_paths()
    old.trials.atomic_json(OUT/'verification.json',dict(models=mc,actors=ac,paths=pc,
        guaranteed_model_count=100,continuous_margin=MARGIN,native_actor_reproduction=True))
    plots(rows,actors,traces,switches,free)
    lines=['# Hypothesis tests and guarantees','',
        '| Concentration | Method | Models | Final folded | Mean fixed-batch objective | Mean population objective | Mean affine log-distance R² |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in models:lines.append(f"| {'native' if r['level'] is None else str(100*r['level'])+'%'} | {r['method']} | {r['n']} | {r['folded']} | {r['batch_objective_mean']:.4f} | {r['population_objective_mean']:.4f} | {r['mean_r2']:.4f} |")
    lines += ['', '## Actor geometry controls','',
        '| Sensor | Mean distance | Fraction below 2 | Fraction above 15 | Mean final training reward |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in actor_summary:lines.append(f"| {r['kind']} | {r['mean_distance']:.4f} | {r['fraction_below2']:.4f} | {r['fraction_above15']:.4f} | {r['final_training_reward']:.6f} |")
    lines += ['', '## Concentration switches: source 131, one far bump','',
        '| Old → new concentration | Switch update | Reset Adam | Final folded | Turning distances | Fixed-batch objective | Population objective |',
        '| --- | ---: | --- | --- | --- | ---: | ---: |']
    for r in switch_summary:lines.append(f"| {100*r['old_level']:g}% → {100*r['new_level']:g}% | {r['at']} | {r['reset']} | {r['folded']} | {', '.join(f'{v:.4f}' for v in r['roots']) or '—'} | {r['batch_objective']:.4f} | {r['population_objective']:.4f} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    html=['<!doctype html><meta charset="utf-8"><title>Hypotheses and guaranteed no-fold models</title><style>body{font:18px system-ui;max-width:1450px;margin:40px auto;padding:0 20px}img{max-width:100%}section{margin:45px 0}</style><h1>Hypotheses and guaranteed no-fold models</h1><p><a href="../README.md">Explanation and proofs</a> · <a href="RESULTS.md">Measured tables</a></p>']
    titles=['01_geometry_and_actor_preference','02_distribution_switches','03_near_far_gradient_balance','04_hard_no_fold_guarantee','05_objective_comparison']
    if (OUT/'06_rare_tail_codes.png').exists():titles.append('06_rare_tail_codes')
    for title in titles:
        html.append(f'<section><h2>{title[3:].replace("_"," ")}</h2><img src="{title}.png"></section>')
    (OUT/'index.html').write_text('\n'.join(html))
    files=[HERE/'PROTOCOL.md',HERE/'README.md',HERE/'guaranteed_models.py',HERE/'actor_geometry.py',HERE/'encoder_paths.py',HERE/'tail_codes.py',HERE/'report.py']
    files += [old.ORIGINAL/'sources'/f'icm-{s}.pt' for s in range(130,140)]
    old.trials.atomic_json(OUT/'source-hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('Verified 100 continuous monotonicity certificates, 30 actor evaluations and 10 trace/switch cases.',flush=True)


if __name__=='__main__':main()

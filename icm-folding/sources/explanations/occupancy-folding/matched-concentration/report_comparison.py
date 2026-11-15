"""Independent audit and graphs for the concentration-matched comparison."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from run_comparison import HERE,OUT,old,dataset,target_measure,jobs,key,LEVELS,TAILS,BINS,SPLIT,EPS,metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS=dict(native='Native-shaped tail',diffuse='Diffuse tail',one='One far bump',two='Two far bumps')
COLORS=dict(native='#6e6e6e',diffuse='#8758a3',one='#d48a18',two='#147b79')


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def plot_design():
    ds=dataset(131);centers=np.exp((np.arange(BINS)+.5)*np.log(16)/BINS)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for tail in TAILS:
        r=target_measure(ds,.875,tail)
        axes[0].plot(centers[:SPLIT],r[:SPLIT],c=COLORS[tail],lw=2,label=LABELS[tail])
        y=r.copy();y[y==0]=np.nan
        axes[1].plot(centers[SPLIT:],y[SPLIT:],c=COLORS[tail],lw=2,label=LABELS[tail])
    axes[0].set(title='Exactly the same near-object profile',xlim=(1,2),xlabel='Physical distance',ylabel='Probability per log-distance bin')
    axes[0].text(.97,.95,'87.5% of all probability\nbelow distance 2',ha='right',va='top',transform=axes[0].transAxes,fontsize=11)
    axes[1].set(title='Change only the remaining tail shape',xlim=(2,16),xlabel='Physical distance',ylabel='Probability per log-distance bin',yscale='log')
    axes[1].axvspan(float(ds['d'].max()),16,color='gray',alpha=.12)
    axes[1].legend(fontsize=8)
    r=target_measure(ds,.875,'one');mu=r@ds['mean'];var=r@ds['second']-mu**2
    fig.suptitle('The decisive comparison: same 87.5% near mass, same stored support, same 2% background',fontsize=13)
    fig.text(.5,-.01,f'One/two-bump cases also match the full mean and variance of log-distance (mean {mu:.5f}, SD {np.sqrt(var):.5f}).\nSource 131 shown. No new observations are added; gray marks distances beyond its stored range.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.9));save(fig,'01_matched_distributions')


def load_runs():
    expected={key(*job) for job in jobs()}
    paths={p.stem:p for p in (OUT/'runs').glob('*.pt')}
    assert set(paths)==expected,(len(paths),sorted(expected-set(paths)))
    return [old.trials.load(paths[key(*job)]) for job in jobs()]


def audit(runs):
    manifest=json.loads((OUT/'manifest.json').read_text())
    for filename,digest in manifest['sha256'].items():
        p=Path(filename)
        if p==HERE/'PROTOCOL.md':p=HERE/'INITIAL_PROTOCOL.md'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,str(p)
    checks=[];distribution_checks=[]
    for seed in range(130,140):
        ds=dataset(seed);group=[r for r in runs if r['seed']==seed]
        indices={r['index_sha256'] for r in group};assert len(indices)==1
        initial=group[0]['states'][0]
        assert all(all(torch.equal(v,r['states'][0][n]) for n,v in initial.items()) for r in group)
        replay=[r for r in group if r['estimator']=='replay']
        assert len({r['near_draw_sha256'] for r in replay})==1
        for a in LEVELS:
            targets=[target_measure(ds,a,t) for t in TAILS]
            for r in targets:
                assert abs(r[:SPLIT].sum()-a)<1e-12
                assert np.array_equal(r>0,ds['p']>0)
                np.testing.assert_array_equal(r[:SPLIT],targets[0][:SPLIT])
            # Independently integrate over actual endpoints, not bin summaries.
            point_prob=[]
            for r in targets[-2:]:
                values=r[ds['bins'].numpy()]/ds['counts'][ds['bins']].double().numpy()
                assert abs(values.sum()-1)<1e-12
                point_prob.append(values)
            q=ds['q'].double().numpy()
            difference=point_prob[0]-point_prob[1]
            assert abs(difference@q)<1e-9 and abs(difference@(q*q))<1e-9
            distribution_checks.append(dict(seed=seed,level=a,near_mass=True,near_profile_identical=True,
                same_support=True,mean_log_error=float(difference@q),second_log_error=float(difference@(q*q))))
        for r in group:
            model=old.ScaledEncoder(4);model.load_state_dict(r['encoder'])
            q=torch.linspace(0,np.log(16),10001,requires_grad=True)
            y=model(old.trials.behavior.sensor(q.exp(),ds['c']))
            slope=torch.autograd.grad(y.sum(),q)[0]
            evaluated=metrics(q.detach().numpy(),y.detach().numpy(),slope.detach().numpy())
            for field in ('roots','folded'):assert evaluated[field]==r['dense'][field]
            for field in ('backward_fraction','r2','std'):assert evaluated[field]==r['dense'][field]
            # Reconstruct all retained curve checkpoints, including initial/final.
            for step,state in r['states'].items():
                if step%100!=0:continue
                model.load_state_dict(state)
                qq=torch.linspace(0,np.log(16),1025,requires_grad=True)
                yy=model(old.trials.behavior.sensor(qq.exp(),ds['c']))
                ss=torch.autograd.grad(yy.sum(),qq)[0]
                saved=next(f for f in r['frames'] if f['step']==step)
                np.testing.assert_array_equal(yy.detach().numpy(),saved['y'])
                np.testing.assert_array_equal(ss.detach().numpy(),saved['slope'])
            if r['estimator']=='native_paired':assert r['native_reproduction_max_error']==0
            if r['estimator']=='replay':assert abs(r['observed_mass'][:SPLIT].sum()-.875)<1e-12
            checks.append(dict(key=key(r['seed'],r['estimator'],r['level'],r['tail']),
                final_dense_metrics_exact=True,retained_curves_exact=True,
                coarse_dense_fold_agreement=bool(r['frames'][-1]['roots'])==bool(r['dense']['roots'])))
    old.trials.atomic_json(OUT/'verification.json',dict(model_checks=checks,distribution_checks=distribution_checks,
        matching_indices_and_initializations=True,replay_near_draws_identical=True,native_reproductions=10))


def summarize(runs):
    rows=[]
    for r in runs:
        m=r['dense']
        rows.append(dict(seed=r['seed'],estimator=r['estimator'],level=r['level'],tail=r['tail'],
            **m,observed_near_mass=float(r['observed_mass'][:SPLIT].sum()),
            target_near_mass=float(r['target_mass'][:SPLIT].sum()),
            distribution_tv=float(.5*np.abs(r['observed_mass']-r['target_mass']).sum()),
            mean_ess=r['mean_ess'],min_ess=r['min_ess'],support=r['support'],
            ever_observed_fold=any(f['folded'] for f in r['frames'] if f['step']>0),
            fraction_checkpoints_folded=float(np.mean([f['folded'] for f in r['frames'] if f['step']>0])),
            first_observed_root_step=next((f['step'] for f in r['frames'] if f['step']>0 and f['folded']),None),
            last_observed_root_step=next((f['step'] for f in reversed(r['frames']) if f['step']>0 and f['folded']),None)))
    summaries=[]
    for estimator,levels,tails in (('native_paired',(None,),(None,)),('native_floor',(None,),(None,)),
         ('native_replay',(None,),(None,)),('weighted',LEVELS,TAILS),('replay',(.875,),TAILS)):
        for a in levels:
            for tail in tails:
                group=[r for r in rows if r['estimator']==estimator and r['level']==a and r['tail']==tail]
                assert len(group)==10
                summaries.append(dict(estimator=estimator,level=a,tail=tail,n=10,
                    folded=sum(r['folded'] for r in group),in_support_folded=sum(bool(r['in_support_roots']) for r in group),
                    ever_observed_folded=sum(r['ever_observed_fold'] for r in group),
                    transient_only=sum(r['ever_observed_fold'] and not r['folded'] for r in group),
                    mean_r2=float(np.mean([r['r2'] for r in group])),
                    mean_backward_fraction=float(np.mean([r['backward_fraction'] for r in group])),
                    mean_std=float(np.mean([r['std'] for r in group])),
                    min_std=float(min(r['std'] for r in group)),
                    observed_near_mass_range=[min(r['observed_near_mass'] for r in group),max(r['observed_near_mass'] for r in group)],
                    mean_ess=float(np.mean([r['mean_ess'] for r in group]))))
    paired=[]
    for estimator,levels in (('weighted',LEVELS),('replay',(.875,))):
        for a in levels:
            one={r['seed']:r for r in rows if r['estimator']==estimator and r['level']==a and r['tail']=='one'}
            two={r['seed']:r for r in rows if r['estimator']==estimator and r['level']==a and r['tail']=='two'}
            pairs=[(one[s]['folded'],two[s]['folded']) for s in range(130,140)]
            ratios=[two[s]['roots'][0]/one[s]['roots'][0] for s in range(130,140)
                    if len(one[s]['roots'])==len(two[s]['roots'])==1
                    and len(one[s]['in_support_roots'])==len(two[s]['in_support_roots'])==1]
            paired.append(dict(estimator=estimator,level=a,both_folded=sum(x and y for x,y in pairs),
                only_one_bump_folded=sum(x and not y for x,y in pairs),only_two_bumps_folded=sum(y and not x for x,y in pairs),
                neither_folded=sum(not x and not y for x,y in pairs),single_turn_comparisons=len(ratios),
                median_turn_distance_ratio=None if not ratios else float(np.median(ratios))))
    old.trials.atomic_json(OUT/'summary.json',dict(summary=summaries,paired_one_vs_two=paired,runs=rows))
    return rows,summaries,paired


def plot_outcomes(runs,rows,summaries):
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,estimator,title in zip(axes,('weighted','replay'),('Same paired batches; weighted SIGReg averages','Stratified replay: exactly 87.5% near in every batch')):
        group=[next(s for s in summaries if s['estimator']==estimator and s['level']==.875 and s['tail']==t) for t in TAILS]
        ax.bar(range(4),[s['folded'] for s in group],color=[COLORS[t] for t in TAILS],width=.65)
        for i,s in enumerate(group):ax.text(i,s['folded']+.2,f"{s['folded']}/10",ha='center')
        ax.set(xticks=range(4),xticklabels=['Native tail','Diffuse tail','One far bump','Two far bumps'],ylim=(0,11.5),title=title,ylabel='Final folded runs / 10')
        ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('Does changing tail shape remove folds when near-object concentration stays fixed?\nWeights impose the designed near-heavy measure; they do not balance it to uniform.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.85));save(fig,'02_decisive_outcomes')
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for tail in TAILS:
        group=[next(s for s in summaries if s['estimator']=='weighted' and s['level']==a and s['tail']==tail) for a in LEVELS]
        x=np.array(LEVELS)*100
        axes[0].plot(x,[s['folded'] for s in group],'-o',c=COLORS[tail],label=LABELS[tail])
        axes[1].plot(x,[s['ever_observed_folded'] for s in group],'-o',c=COLORS[tail])
        axes[2].plot(x,[100*s['mean_backward_fraction'] for s in group],'-o',c=COLORS[tail])
    axes[0].set(ylabel='Folded runs / 10',ylim=(-.4,10.8),title='Final fold count')
    axes[1].set(ylabel='Runs with any sampled turn / 10',ylim=(-.4,10.8),title='Including transient turns')
    axes[2].set(ylabel='Mean backward part of curve (%)',ylim=(-1,51),title='Final fold severity')
    for ax in axes:
        ax.set(xlabel='Target probability below distance 2 (%)',xticks=np.array(LEVELS)*100,
               xticklabels=[f'{100*a:g}' for a in LEVELS])
        ax.tick_params(axis='x',labelrotation=30)
    axes[0].legend(fontsize=8)
    fig.suptitle('Vary concentration while retaining the same near/tail components and background',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'03_concentration_dose_response')
    fig,ax=plt.subplots(figsize=(13,5.5))
    columns=[(e,t) for e in ('weighted','replay') for t in TAILS]
    colors=np.zeros((10,8));text={}
    for i,s in enumerate(range(130,140)):
        for j,(estimator,tail) in enumerate(columns):
            r=next(r for r in rows if r['seed']==s and r['estimator']==estimator and r['level']==.875 and r['tail']==tail)
            colors[i,j]=2 if r['in_support_roots'] else 1 if r['roots'] else 0
            text[i,j]='—' if not r['roots'] else ', '.join(f'{x:.1f}'+('*' if x in r['extrapolation_roots'] else '') for x in r['roots'])
    from matplotlib.colors import ListedColormap
    ax.imshow(colors,cmap=ListedColormap(['#d7eee6','#f9e5ba','#f2c5c6']),vmin=0,vmax=2,aspect='auto')
    for (i,j),value in text.items():ax.text(j,i,value,ha='center',va='center',fontsize=8)
    ax.set(yticks=range(10),yticklabels=[f'Seed {s}' for s in range(130,140)],xticks=range(8),
        xticklabels=[('Weighted' if e=='weighted' else 'Replay')+'\n'+{'native':'native tail','diffuse':'diffuse','one':'one bump','two':'two bumps'}[t] for e,t in columns])
    ax.axvline(3.5,c='white',lw=3)
    ax.set_title('Every source at 87.5% near concentration · cells show turning distances',pad=14)
    fig.text(.5,-.01,'Green: no detected turn. Red: at least one turn within the stored distance range. Amber: extrapolation-only turn.\n* A turn outside that source’s observed distance range. All turns count in the main totals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,'04_all_sources')
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex='row')
    for col,seed in enumerate((130,131,136)):
        for tail in TAILS:
            r=next(r for r in runs if r['seed']==seed and r['estimator']=='weighted' and r['level']==.875 and r['tail']==tail)
            f=r['frames'][-1];q=f['q'];y=f['y'];sign=1 if np.cov(q,y)[0,1]>=0 else -1
            axes[0,col].plot(np.exp(q),sign*(y-y.mean())/y.std(),c=COLORS[tail],label=LABELS[tail])
            xx=[];yy=[]
            for frame in r['frames']:xx.extend([frame['step']]*len(frame['roots']));yy.extend(frame['roots'])
            axes[1,col].scatter(xx,yy,c=COLORS[tail],s=7,alpha=.7)
        ds=dataset(seed)
        for ax in (axes[0,col],axes[1,col]):
            if ax is axes[0,col]:ax.axvspan(float(ds['d'].max()),16,color='gray',alpha=.1)
        axes[0,col].set(title=f'Seed {seed}: final learned shape',xlabel='Physical distance',ylabel='Learned code (display scale)')
        axes[1,col].set(xlim=(0,4000),ylim=(1,16),xlabel='Training updates',ylabel='Turning distance')
        axes[1,col].axhspan(float(ds['d'].max()),16,color='gray',alpha=.1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same 87.5% near concentration · shape and turn histories\nPrespecified diagnostic sources; turn histories sampled every 100 updates',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'05_curves_and_turn_histories')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    controls=[next(s for s in summaries if s['estimator']==e) for e in ('native_paired','native_floor','native_replay')]
    axes[0].bar(['Original paired','Add only 2% background','Native stratified replay'],[s['folded'] for s in controls],color=['#c13c45','#8758a3','#345e91'])
    for i,s in enumerate(controls):axes[0].text(i,s['folded']+.2,f"{s['folded']}/10",ha='center')
    axes[0].set(title='Bridge controls',ylabel='Folded runs / 10',ylim=(0,11));axes[0].tick_params(axis='x',labelrotation=12)
    for estimator,color in (('weighted','#d48a18'),('replay','#147b79')):
        group=[s for s in summaries if s['estimator']==estimator and s['level']==.875]
        for i,t in enumerate(TAILS):
            s=next(s for s in group if s['tail']==t);lo,hi=s['observed_near_mass_range']
            axes[1].plot([i,i],[100*lo,100*hi],c=color,lw=5,alpha=.75,label=estimator if i==0 else None)
            if estimator=='replay':axes[1].scatter([i],[100*lo],c=color,s=25)
    axes[1].axhline(87.5,c='black',ls='--',lw=1)
    axes[1].set(xticks=range(4),xticklabels=['Native tail','Diffuse','One bump','Two bumps'],ylabel='Range of source-average near mass (%)',title='Finite-batch distribution check')
    axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'06_bridge_and_estimator_checks')
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex='row',sharey='row')
    for col,a in enumerate(LEVELS):
        r=next(r for r in runs if r['seed']==131 and r['estimator']=='weighted' and r['level']==a and r['tail']=='one')
        f=r['frames'][-1];q=f['q'];y=f['y'];sign=1 if np.cov(q,y)[0,1]>=0 else -1
        axes[0,col].plot(np.exp(q),sign*(y-y.mean())/y.std(),c=COLORS['one'],lw=2)
        axes[0,col].set(title=f'{100*a:g}% below distance 2',xlabel='Physical distance')
        axes[0,col].axvspan(r['support'][1],16,color='gray',alpha=.1)
        xx=[];yy=[]
        for frame in r['frames']:xx.extend([frame['step']]*len(frame['roots']));yy.extend(frame['roots'])
        axes[1,col].scatter(xx,yy,c=COLORS['one'],s=12)
        axes[1,col].set(xlim=(0,4000),ylim=(.95,16),xlabel='Training updates')
        if not xx:axes[1,col].text(.5,.85,'No sampled turn',ha='center',transform=axes[1,col].transAxes)
        else:axes[1,col].text(.97,.95,f'Turn range: {min(yy):.2f}–{max(yy):.2f}',ha='right',va='top',transform=axes[1,col].transAxes,fontsize=9)
    axes[0,0].set_ylabel('Final learned code (display scale)');axes[1,0].set_ylabel('Turning distance')
    fig.suptitle('Source 131: concentration need not have a monotone effect\nSame one-bump tail family, same initial model and alignment batches; only concentration changes',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'08_one_source_concentration_histories')


def write_reports(rows,summaries,paired):
    lines=['# Concentration-matched results','',
        'Ten previously inspected ICM sources. All primary targets retain the same observed-point support and 2% uniform-over-occupied-log-bins background. One/two-bump cases match full first two log-distance moments. All counts include extrapolation turns on [1,16].', '',
        '| Estimator | Near mass | Tail | Final folded | Within stored range | Ever observed folded | Mean R² | Mean backward travel (%) | Mean weight ESS |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        a='native' if s['level'] is None else f"{100*s['level']:g}%"
        lines.append(f"| {s['estimator']} | {a} | {s['tail'] or 'native'} | {s['folded']}/10 | {s['in_support_folded']}/10 | {s['ever_observed_folded']}/10 | {s['mean_r2']:.4f} | {100*s['mean_backward_fraction']:.4f} | {s['mean_ess']:.1f} |")
    lines += ['', '## Paired one-bump versus two-bump comparisons', '',
        '| Estimator | Near mass | Both folded | Only one-bump folded | Only two-bump folded | Neither folded | Single-turn cases | Median turn ratio: two/one |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for p in paired:
        ratio='—' if p['median_turn_distance_ratio'] is None else f"{p['median_turn_distance_ratio']:.4f}"
        lines.append(f"| {p['estimator']} | {100*p['level']:g}% | {p['both_folded']} | {p['only_one_bump_folded']} | {p['only_two_bumps_folded']} | {p['neither_folded']} | {p['single_turn_comparisons']} | {ratio} |")
    lines += ['', '[Graph gallery](index.html) · [Explanation and limitations](../README.md) · [All numerical metrics](summary.json) · [Verification](verification.json)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    names=['01_matched_distributions','02_decisive_outcomes','03_concentration_dose_response','04_all_sources','05_curves_and_turn_histories','06_bridge_and_estimator_checks','08_one_source_concentration_histories']
    if (OUT/'07_local_slope_force.png').exists():names.append('07_local_slope_force')
    html=['<!doctype html><meta charset="utf-8"><title>Concentration-matched folding comparison</title><style>body{font:18px system-ui;max-width:1450px;margin:40px auto;padding:0 20px}img{max-width:100%}section{margin:45px 0}</style><h1>Concentration-matched folding comparison</h1><p><a href="../README.md">Explanation</a> · <a href="RESULTS.md">Numerical tables</a></p>']
    for name in names:html.append(f'<section><h2>{name[3:].replace("_"," ")}</h2><img src="{name}.png"></section>')
    (OUT/'index.html').write_text('\n'.join(html))


def main():
    torch.set_num_threads(1);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    runs=load_runs();audit(runs);rows,summaries,paired=summarize(runs)
    plot_design();plot_outcomes(runs,rows,summaries);write_reports(rows,summaries,paired)
    old.trials.atomic_json(OUT/'report-hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'report_comparison.py',HERE/'force_audit.py',HERE/'PROTOCOL.md',HERE/'INITIAL_PROTOCOL.md',HERE/'README.md')})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

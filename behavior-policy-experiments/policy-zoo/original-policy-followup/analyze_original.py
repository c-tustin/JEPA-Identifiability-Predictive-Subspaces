"""Original-policy identifiability, mechanism intervention and planning results."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/kempner-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import original as o
from analyze_matched import sign_test, holm

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
LABELS = {'icm':'ICM','random':'Random','count':'Count','ou_actions_lp_w0.1':'Low-pass OU actions','rnd':'RND','ou':'Ideal OU pairs'}
COLORS = {'icm':'#d95f02','random':'#377eb8','count':'#4daf4a','ou_actions_lp_w0.1':'#984ea3','rnd':'#e7298a','ou':'#333333'}


def bootstrap(delta):
    delta=np.asarray(delta)
    rng=np.random.default_rng(20261106)
    return np.quantile(delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1),[.025,.975]).tolist()


def corrected(rows):
    for row, adjusted in zip(rows,holm([r['p'] for r in rows])):
        row['holm_p']=float(adjusted)
    return rows


def intervention_stats(original, interventions):
    lookup={(r['policy'],r['seed']):r for r in original if r['protocol']=='pure'}
    comparisons=[]
    for name in o.INTERVENTION_POLICIES:
        pairs=[(lookup[r['policy'],r['seed']],r['metrics']) for r in interventions if r['policy']==name]
        if not pairs:continue
        for endpoint in ['affine_log_r2','fold_status']:
            if endpoint=='affine_log_r2':
                delta=np.array([b['affine_log_r2']-a['affine_log_r2'] for a,b in pairs])
            else:
                delta=np.array([int(a['folded'])-int(b['folded']) for a,b in pairs])
            wins,losses,p=sign_test(delta)
            comparisons.append({'policy':name,'endpoint':endpoint,'n':len(pairs),
                                'mean_gain':float(delta.mean()),'wins':wins,'losses':losses,
                                'p':p,'ci95':bootstrap(delta)})
    return corrected(comparisons)


def planning_stats(rows):
    lookup={(r['policy'],r['seed']):r for r in rows}
    shared, native=[],[]
    for name in o.INTERVENTION_POLICIES:
        selected=[r for r in rows if r['policy']==name and ('ou',r['seed']) in lookup]
        if selected:
            delta=np.array([lookup['ou',r['seed']]['planning']['shared']['goal_error']-r['planning']['shared']['goal_error'] for r in selected])
            wins,losses,p=sign_test(delta)
            shared.append({'policy':name,'n':len(delta),'mean_gain':float(delta.mean()),'wins':wins,'losses':losses,'p':p,'ci95':bootstrap(delta)})
        selected=[r for r in rows if r['policy']==name]
        if selected:
            delta=np.array([r['planning']['shared']['goal_error']-r['planning']['native']['goal_error'] for r in selected])
            wins,losses,p=sign_test(delta)
            native.append({'policy':name,'n':len(delta),'mean_gain':float(delta.mean()),'wins':wins,'losses':losses,'p':p,'ci95':bootstrap(delta)})
    return {'shared_vs_ou':corrected(shared),'native_vs_shared':corrected(native)}


def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def figures(original,interventions,planning):
    pure=[r for r in original if r['protocol']=='pure']
    names=sorted({r['policy'] for r in pure},key=lambda n:np.mean([r['historical_r2'] for r in pure if r['policy']==n]))
    fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True)
    for index,name in enumerate(names):
        group=[r for r in pure if r['policy']==name]
        color=COLORS.get(name,'#888888')
        for ax,key in [(axes[0],'historical_r2'),(axes[1],'absolute_rank_correlation'),(axes[2],'alias_distance_gap')]:
            values=[r[key] for r in group]
            ax.scatter(values,index+np.linspace(-.15,.15,len(values)),s=13,color=color,alpha=.5)
            ax.plot([np.mean(values)]*2,[index-.26,index+.26],color=color,lw=3)
    axes[0].set_yticks(range(len(names)),[LABELS.get(n,n) for n in names])
    axes[0].set(xlabel='Affine log-distance R² ↑',title='Nearly linear recovery')
    axes[1].set(xlabel='Absolute rank correlation ↑',title='Preserving distance order')
    axes[2].set(xlabel='Equal-code distance separation',title='Numerical alias witnesses')
    for ax in axes:ax.grid(axis='x',alpha=.2)
    save(fig,'original_identifiability')
    lookup={(r['policy'],r['seed']):r for r in pure}
    curves=np.load(OUT/'original_curves.npz')
    distance=curves['distance']
    for name in o.INTERVENTION_POLICIES:
        selected=sorted([r for r in interventions if r['policy']==name],key=lambda r:r['seed'])
        if not selected:continue
        fig,axes=plt.subplots(4,5,figsize=(15,10),sharex=True,sharey=True)
        for ax,row in zip(axes.flat,selected):
            seed=row['seed']
            raw=lookup[name,seed]
            new=np.load(OUT/f'no_alignment-{name}_seed{seed}.npz')['curve']
            old=curves[f'pure-{name}-{seed}']
            ax.plot(distance,old,color='#d95f02',label='Original loss')
            ax.plot(distance,new,color='#7570b3',ls='--',label='Alignment removed')
            ax.set(title=f"Seed {seed}: fold {int(raw['folded'])} → {int(row['metrics']['folded'])}",xscale='log')
            ax.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'{LABELS[name]}: exact same observations and SIGReg; remove alignment\nCurves centered/scaled for display; all seeds retained',y=1.02)
        for ax in axes[-1]:ax.set_xlabel('Physical distance')
        save(fig,'curves_'+name)
    if interventions:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for index,name in enumerate(o.INTERVENTION_POLICIES):
            selected=[r for r in interventions if r['policy']==name]
            for row in selected:
                a=lookup[name,row['seed']]['affine_log_r2'];b=row['metrics']['affine_log_r2']
                axes[0].plot([index-.18,index+.18],[a,b],color=COLORS[name],alpha=.45,marker='o',ms=3)
            if selected:
                rawfold=np.mean([lookup[name,r['seed']]['folded'] for r in selected])
                newfold=np.mean([r['metrics']['folded'] for r in selected])
                axes[1].plot([index-.18,index+.18],[rawfold,newfold],color=COLORS[name],marker='o')
        for ax in axes:
            ax.set_xticks(range(5),[LABELS[n] for n in o.INTERVENTION_POLICIES],rotation=20);ax.grid(axis='y',alpha=.2)
        axes[0].set(title='Left: original; right: alignment removed',ylabel='Affine log-distance R²')
        axes[1].set(title='Does removing alignment repair folds?',ylabel='Fraction of runs folded',ylim=(-.03,1.03))
        save(fig,'alignment_removal')
    if planning:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for index,name in enumerate(o.PLANNING_POLICIES):
            group=sorted([r for r in planning if r['policy']==name],key=lambda r:r['seed'])
            for ax,kind in [(axes[0],'shared'),(axes[1],'native')]:
                values=[r['planning'][kind]['goal_error'] for r in group if kind in r['planning']]
                if values:
                    ax.scatter(index+np.linspace(-.12,.12,len(values)),values,color=COLORS[name],alpha=.6,s=20)
                    ax.plot([index-.2,index+.2],[np.mean(values)]*2,color=COLORS[name],lw=3)
            axes[2].scatter([lookup[name,r['seed']]['affine_log_r2'] for r in group],
                            [r['planning']['shared']['goal_error'] for r in group],color=COLORS[name],label=LABELS[name],s=25)
        for ax in axes[:2]:
            ax.set_xticks(range(6),[LABELS[n] for n in o.PLANNING_POLICIES],rotation=35,ha='right');ax.set_ylabel('Actual goal error ↓')
        axes[0].set_title('Original encoders, shared predictor data')
        axes[1].set_title('Original physical-policy predictor data')
        axes[2].set(xlabel='Affine log-distance R²',ylabel='Shared-data actual goal error',title='Recovery versus reaching')
        axes[2].legend(fontsize=7)
        save(fig,'original_planning')


def main():
    original=json.loads((OUT/'original_geometry.json').read_text())
    interventions=[json.loads(p.read_text()) for p in sorted(OUT.glob('no_alignment-*_seed*.json'))]
    planning=[json.loads(p.read_text()) for p in sorted(OUT.glob('planning-*_seed*.json'))]
    istats=intervention_stats(original,interventions)
    pstats=planning_stats(planning)
    o.save_json(OUT/'statistics.json',{'alignment_removal':istats,'planning':pstats})
    figures(original,interventions,planning)
    lookup={(r['protocol'],r['policy'],r['seed']):r for r in original}
    lines=['# Original policies: identifiability and planning','',
           '**The original zoo policies differ strongly.** The previous matched-coverage test used altered stationary policies; its null result did not establish that original ICM, random, count or OU are interchangeable.','',
           f"Re-evaluated **{len(original)} saved original encoders**. Completed **{len(interventions)}/100 alignment-removal fits** and **{sum(len(r['planning']) for r in planning)}/110 original-encoder planning evaluations**. [Protocol](../PROTOCOL.md) · [Hypothesis and exact loss symmetry](../MECHANISM.md).",'',
           '## Original-policy identifiability','',
           'Same 20 paired zoo seeds, 8–27, and common evaluation range. Affine R² is the original saved metric. Rank and equal-code witnesses are new evaluations of those unchanged encoders. Handover is the old policy→OU curriculum, not training on that policy alone.','',
           '| Original policy | Pure R² ↑ | Pure folds | Rank correlation ↑ | Mean alias separation | Handover R² ↑ | Handover folds |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    pure=[r for r in original if r['protocol']=='pure']
    names=sorted({r['policy'] for r in pure},key=lambda n:-np.mean([r['historical_r2'] for r in pure if r['policy']==n]))
    for name in names:
        group=[r for r in pure if r['policy']==name]
        hand=[r for r in original if r['protocol']=='handover' and r['policy']==name]
        mean=lambda key:np.mean([r[key] for r in group])
        hand_r2=f"{np.mean([r['historical_r2'] for r in hand]):.4f}" if hand else '—'
        handfold=f"{sum(r['folded'] for r in hand)}/{len(hand)}" if hand else '—'
        lines += [f"| {LABELS.get(name,name)} | {mean('historical_r2'):.4f} | {sum(r['folded'] for r in group)}/{len(group)} | {mean('absolute_rank_correlation'):.4f} | {mean('alias_distance_gap'):.3f} | {hand_r2} | {handfold} |"]
    lines += ['', 'A warped monotone map can retain information without linear recovery. Equal-code witnesses provide evidence of actual aliasing; they are numerically refined on the network and are not an exhaustive maximum-gap search or formal floating-point impossibility certificate. All new fold checks use the same 1,025-point grid; historical R² uses the prior evaluation.','',
              '![Original zoo identifiability](original_identifiability.png)','',
              '## New hypothesis test: is alignment necessary for the folds?','',
              'Remove alignment while retaining every original training endpoint, historical batch plan, initialization, optimizer schedule and unchanged SIGReg. This tests necessity of the explicit alignment term; it does not identify a unique cause of folding. Paired endpoints still co-occur in finite SIGReg batches, so pair-law effects are not all removed. [Interpretation clarification](../CLARIFICATIONS.md).','',
              '| Actual data source | Seeds | Original R² | Alignment removed R² | Original folds retained | Repaired | Newly folded |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in o.INTERVENTION_POLICIES:
        selected=[r for r in interventions if r['policy']==name]
        if not selected:continue
        pairs=[(lookup['pure',name,r['seed']],r['metrics']) for r in selected]
        retained=sum(a['folded'] and b['folded'] for a,b in pairs)
        oldfold=sum(a['folded'] for a,b in pairs)
        repaired=sum(a['folded'] and not b['folded'] for a,b in pairs)
        newfold=sum(not a['folded'] and b['folded'] for a,b in pairs)
        lines += [f"| {LABELS[name]} | {len(pairs)} | {np.mean([a['affine_log_r2'] for a,b in pairs]):.4f} | {np.mean([b['affine_log_r2'] for a,b in pairs]):.4f} | {retained}/{oldfold} | {repaired} | {newfold} |"]
    lines += ['', '![Paired alignment-removal results](alignment_removal.png)','',
              '| Source | Endpoint | Mean improvement [unadjusted 95% interval] | Positive / negative seeds | Holm p |',
              '| --- | --- | ---: | ---: | ---: |']
    for r in istats:
        lines += [f"| {LABELS[r['policy']]} | {r['endpoint']} | {r['mean_gain']:+.4g} [{r['ci95'][0]:+.4g}, {r['ci95'][1]:+.4g}] | {r['wins']} / {r['losses']} | {r['holm_p']:.4g} |"]
    lines += ['', 'Positive improvement means higher recovery or fewer folds after removing alignment. One Holm family covers all ten tests; fold tests use discordant pairs only. All seeds are included. These are exploratory fixed-source interventions, not population identifiability guarantees.','',
              'All-trial curves: [ICM](curves_icm.png) · [Random](curves_random.png) · [Count](curves_count.png) · [Low-pass OU actions](curves_ou_actions_lp_w0.1.png) · [RND](curves_rnd.png).','',
              '## Actual goal-reaching with original encoders','',
              'Ten fixed seeds, 8–17; original encoders unchanged. Identical 64 reachable goals per seed. Shared and native predictors have equal data and training budgets. Ideal OU pairs have no physical action labels, so no native-data score is fabricated.','',
              '| Original encoder | Seeds | Shared-data error ↓ | Native-data error ↓ | Shared success ↑ |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name in o.PLANNING_POLICIES:
        group=[r for r in planning if r['policy']==name]
        if not group:continue
        native=[r['planning']['native']['goal_error'] for r in group if 'native' in r['planning']]
        native_text=f'{np.mean(native):.4f}' if native else '—'
        lines += [f"| {LABELS[name]} | {len(group)} | {np.mean([r['planning']['shared']['goal_error'] for r in group]):.4f} | {native_text} | {np.mean([r['planning']['shared']['success'] for r in group]):.3f} |"]
    lines += ['', '![Original-policy goal-reaching](original_planning.png)','',
              'Full paired planning tests and confidence intervals are in [statistics.json](statistics.json). Shared-vs-OU and native-vs-shared are separate Holm families of five. Native data can change both occupancy and state-dependent action coverage; it is not an occupancy-only intervention.','',
              '## Scope','',
              '- The original-policy ranking retains real visitation differences. It is not a matched-distribution causal comparison.',
              '- The alignment-removal test keeps the original endpoint stream fixed but changes one objective term. It tests whether explicit alignment is necessary, without modifying SIGReg; residual finite-batch endpoint correlations remain.',
              '- Fold counts, affine recovery, preserved order and actual reaching are separate outcomes. No finite grid or high R² certifies the paper’s theorem.',
              '- The two-goal alias bound applies to exactly equal codes. Reported witnesses have a numerical residual; actual planning is tested independently.',
              '- This dataset cohort has been inspected previously. New interventions follow a written protocol, but conclusions remain exploratory.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(f'Report: {len(original)} original encoders, {len(interventions)} interventions, {len(planning)} planning arms.')


if __name__=='__main__':main()

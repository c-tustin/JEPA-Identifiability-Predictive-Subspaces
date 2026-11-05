"""Analyze changed collection age for the original unchanged policies."""
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kempner-cache")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import original as o
from analyze_original import bootstrap, corrected, save, LABELS, COLORS
from analyze_matched import sign_test

HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
NAMES=('icm','random','count')


def main():
    original=json.loads((OUT/'original_geometry.json').read_text())
    lookup={(r['policy'],r['seed']):r for r in original if r['protocol']=='pure'}
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('early-*_seed*.json'))]
    original_planning={(r['policy'],r['seed']):r for r in [json.loads(p.read_text()) for p in OUT.glob('planning-*_seed*.json')]}
    tests=[];planning_tests=[]
    for name in NAMES:
        selected=[r for r in rows if r['policy']==name]
        if not selected:continue
        for endpoint in ['affine_log_r2','fold_status']:
            delta=np.array([r['metrics']['affine_log_r2']-lookup[name,r['seed']]['affine_log_r2'] if endpoint=='affine_log_r2'
                            else int(lookup[name,r['seed']]['folded'])-int(r['metrics']['folded']) for r in selected])
            wins,losses,p=sign_test(delta)
            tests.append({'policy':name,'endpoint':endpoint,'n':len(delta),'mean_gain':float(delta.mean()),
                          'wins':wins,'losses':losses,'p':p,'ci95':bootstrap(delta)})
        selected=[r for r in selected if r['planning']]
        if selected:
            delta=np.array([original_planning[name,r['seed']]['planning']['shared']['goal_error']-r['planning']['shared']['goal_error'] for r in selected])
            wins,losses,p=sign_test(delta)
            planning_tests.append({'policy':name,'n':len(delta),'mean_gain':float(delta.mean()),
                                   'wins':wins,'losses':losses,'p':p,'ci95':bootstrap(delta)})
    corrected(tests);corrected(planning_tests)
    o.save_json(OUT/'early_statistics.json',{'identifiability':tests,'planning':planning_tests})
    old_curves=np.load(OUT/'original_curves.npz');distance=old_curves['distance']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,NAMES):
        selected=[r for r in rows if r['policy']==name]
        if not selected:continue
        for phase,color in [('original','#d95f02'),('early','#377eb8')]:
            probability=np.mean([r[phase+'_exposure']['log_bin_probability'] for r in selected],axis=0)
            centers=np.exp((np.arange(32)+.5)*np.log(16)/32)
            ax.plot(centers,probability,color=color,label='Late (steps 128–191)' if phase=='original' else 'Early (steps 0–63)')
        ax.set(title=LABELS[name],xlabel='Physical distance',ylabel='Pooled endpoint mass per log bin',xscale='log')
        ax.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,'early_coverage')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for index,name in enumerate(NAMES):
        selected=[r for r in rows if r['policy']==name]
        if not selected:continue
        for r in selected:
            a=lookup[name,r['seed']];b=r['metrics']
            axes[0].plot([index-.2,index+.2],[a['affine_log_r2'],b['affine_log_r2']],color=COLORS[name],alpha=.5,marker='o',ms=3)
        axes[1].plot([index-.2,index+.2],[np.mean([lookup[name,r['seed']]['folded'] for r in selected]),np.mean([r['metrics']['folded'] for r in selected])],color=COLORS[name],marker='o')
        for r in selected:
            axes[2].plot([r['original_exposure']['mass_below_2'],r['early_exposure']['mass_below_2']],
                         [lookup[name,r['seed']]['affine_log_r2'],r['metrics']['affine_log_r2']],color=COLORS[name],alpha=.4)
    for ax in axes[:2]:ax.set_xticks(range(3),[LABELS[n] for n in NAMES]);ax.grid(axis='y',alpha=.2)
    axes[0].set(title='Each seed: late → early collection',ylabel='Affine log-distance R²')
    axes[1].set(title='Same frozen action rule',ylabel='Fraction of runs folded',ylim=(-.03,1.03))
    axes[2].set(xlabel='Fraction of endpoints below distance 2',ylabel='Affine log-distance R²',title='Sampling changes alongside recovery')
    save(fig,'early_identifiability')
    selected=sorted([r for r in rows if r['policy']=='icm'],key=lambda r:r['seed'])
    fig,axes=plt.subplots(4,5,figsize=(15,10),sharex=True,sharey=True)
    for ax,r in zip(axes.flat,selected):
        seed=r['seed'];a=lookup['icm',seed]
        curve=np.load(OUT/f'early-icm_seed{seed}.npz')['curve']
        ax.plot(distance,old_curves[f'pure-icm-{seed}'],color='#d95f02',label='Late collection')
        ax.plot(distance,curve,color='#377eb8',label='Early collection')
        ax.set(title=f"Seed {seed}: fold {int(a['folded'])} → {int(r['metrics']['folded'])}",xscale='log')
        ax.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Original ICM action rule: change only when transitions are collected\nUnchanged full encoder objective; independently normalized curves for display',y=1.02)
    save(fig,'early_icm_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,['shared','native']):
        for index,name in enumerate(NAMES):
            selected=[r for r in rows if r['policy']==name and r['planning']]
            for r in selected:
                old=original_planning[name,r['seed']]['planning'][kind]['goal_error']
                new=r['planning'][kind]['goal_error']
                ax.plot([index-.2,index+.2],[old,new],color=COLORS[name],alpha=.5,marker='o',ms=3)
        ax.set_xticks(range(3),[LABELS[n] for n in NAMES]);ax.set_ylabel('Actual goal-reaching error ↓')
        ax.set_title(kind.title()+' predictor data: late → early collection');ax.grid(axis='y',alpha=.2)
    save(fig,'early_planning')
    lines=['# Same ICM policy, different collection age','',
           f"Completed {len(rows)}/60 encoder fits. [Fixed follow-up protocol](../EARLY_COLLECTION_PROTOCOL.md).",'',
           'The original frozen action rules are unchanged. Late data are steps 128–191; early data are steps 0–63 after the same uniform physical-distance resets. Both phases supply 16,384 real transitions. This intentionally changes the observed distribution and empirical pair law, not the encoder loss or SIGReg. Early data are transient, not stationary.','',
           '| Policy | Seeds | Late R² | Early R² | Late folds | Early folds | Endpoint mass below 2: late → early |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        selected=[r for r in rows if r['policy']==name]
        if selected:
            lines += [f"| {LABELS[name]} | {len(selected)} | {np.mean([lookup[name,r['seed']]['affine_log_r2'] for r in selected]):.4f} | {np.mean([r['metrics']['affine_log_r2'] for r in selected]):.4f} | {sum(lookup[name,r['seed']]['folded'] for r in selected)}/{len(selected)} | {sum(r['metrics']['folded'] for r in selected)}/{len(selected)} | {np.mean([r['original_exposure']['mass_below_2'] for r in selected]):.3f} → {np.mean([r['early_exposure']['mass_below_2'] for r in selected]):.3f} |"]
    lines += ['', '![Original policies, changed collection age](early_identifiability.png)','',
              '![How sampling changes](early_coverage.png)','',
              '![Every ICM trial](early_icm_curves.png)','',
              '| Policy | Endpoint | Mean improvement [unadjusted 95% interval] | Positive / negative seeds | Holm p |',
              '| --- | --- | ---: | ---: | ---: |']
    for r in tests:
        lines += [f"| {LABELS[r['policy']]} | {r['endpoint']} | {r['mean_gain']:+.4g} [{r['ci95'][0]:+.4g}, {r['ci95'][1]:+.4g}] | {r['wins']} / {r['losses']} | {r['holm_p']:.4g} |"]
    lines += ['', 'Recovery/fold tests share one Holm family of six. Sign tests assess seed-wise direction, not zero mean. Fold tests use discordant pairs. All 20 seeds are included.','',
              '## Does the change help actual goal-reaching?','',
              '| Policy | Seeds | Late/shared error | Early/shared error | Late/native error | Early/native error |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        selected=[r for r in rows if r['policy']==name and r['planning']]
        if selected:
            values=[np.mean([original_planning[name,r['seed']]['planning'][kind]['goal_error'] for r in selected]) for kind in ['shared','native']]
            new=[np.mean([r['planning'][kind]['goal_error'] for r in selected]) for kind in ['shared','native']]
            lines += [f"| {LABELS[name]} | {len(selected)} | {values[0]:.4f} | {new[0]:.4f} | {values[1]:.4f} | {new[1]:.4f} |"]
    lines += ['', '![Actual reaching](early_planning.png)','',
              'Shared-data comparison changes the frozen encoder while holding predictor data and goals fixed. Native comparison changes both encoder and predictor data. Every goal has zero physical oracle error.','',
              '| Policy | Shared-data mean error reduction [95% interval] | Positive / negative seeds | Holm p |',
              '| --- | ---: | ---: | ---: |']
    for r in planning_tests:
        lines += [f"| {LABELS[r['policy']]} | {r['mean_gain']:+.4g} [{r['ci95'][0]:+.4g}, {r['ci95'][1]:+.4g}] | {r['wins']} / {r['losses']} | {r['holm_p']:.4g} |"]
    lines += ['', 'Planning has a separate Holm family of three. Goal cases are not independent seeds. These ten-seed task results do not establish a universal policy ranking.','',
              '## Conservative sensitivity check','',
              'The predeclared within-study corrections above support the ICM recovery, fold and shared-planning improvements. A stricter post hoc Holm correction across all 29 newly reported tests gives recovery p = 0.0011 and both fold repair and planning p = 0.0508. Thus the task/fold significance is sensitive to the correction scope. [All-test correction](GLOBAL_CORRECTION.json). Fresh actor/seed replication should precede stronger general claims.','',
              '## Interpretation limits','',
              'Together with the alignment-removal test, this checks whether the ICM fold pattern is sensitive to its sampled observations while retaining its actor. Earlier collection also changes action/state frequencies, boundary encounters and pair connections; it does not isolate a single histogram statistic. Curiosity reward learning was already frozen. This test concerns the data supplied to the JEPA encoder, not a claim that ICM’s internal curiosity model itself folds.', '']
    (OUT/'EARLY_COLLECTION_RESULTS.md').write_text('\n'.join(lines))
    print('Early collection fits analyzed:',len(rows))


if __name__=='__main__':main()

"""Cohort results, raw geometry and source/exposure comparisons."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kempner-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/kempner-cache')
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import common_gc as study
import gc_statistics as stats

OUT=study.OUT
LABELS={'icm':'ICM','random':'Random','count':'Count','rnd':'RND','disagreement':'Disagreement',
        'ou_actions_lp_w0.1':'Low-pass OU actions','ou_actions_hp_w0.3':'High-pass OU actions'}


def rows():
    return [json.loads(p.read_text()) for p in sorted(OUT.glob('*_seed*.json')) if 'experiment' in json.loads(p.read_text())]


def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def factor_figures(records):
    names=[f'factor_A{a}_R{r}' for a in ('early','late') for r in ('early','late')]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for index,name in enumerate(names):
        group=[r for r in records if r['condition']==name]
        if not group:continue
        color='#377eb8' if name.endswith('Rearly') else '#d95f02'
        for ax,getter in [(axes[0],lambda r:r['metrics']['affine_log_r2']),
                          (axes[1],lambda r:int(r['metrics']['folded'])),
                          (axes[2],lambda r:r['planning']['shared']['goal_error'])]:
            values=[getter(r) for r in group]
            ax.scatter(index+np.linspace(-.12,.12,len(group)),values,color=color,alpha=.6,s=20)
            ax.plot([index-.2,index+.2],[np.mean(values)]*2,color=color,lw=3)
    labels=['Align early\nSIGReg early','Align early\nSIGReg late','Align late\nSIGReg early','Align late\nSIGReg late']
    for ax in axes:
        ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.2)
    axes[0].set(title='Distance recovery',ylabel='Affine log-distance R² ↑')
    axes[1].set(title='Fold status for every seed',ylabel='Folded',yticks=[0,1])
    axes[2].set(title='Same predictor data and goals',ylabel='Actual terminal distance error ↓')
    fig.suptitle('Change which data each loss sees; A = alignment, R = regularizer',y=1.03)
    save(fig,'factorial_recovery_and_planning')

    # Fixed seed chosen in the protocol; raw code/sign is anchored to the parent.
    seed=200
    if not all((OUT/f'{name}_seed{seed}.npz').exists() for name in names):return
    for quantity in ('curves','slopes'):
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        for ax,name in zip(axes.flat,names):
            data=np.load(OUT/f'{name}_seed{seed}.npz')
            distance=data['frame_distance'];frames=data['frame_curves']
            sign=1 if np.cov(np.log(distance),frames[0])[0,1]>=0 else -1
            for update,color in [(0,'#888888'),(100,'#56b4e9'),(500,'#0072b2'),(1000,'#cc79a7'),(4000,'#d55e00')]:
                index=int(np.flatnonzero(data['frame_steps']==update)[0])
                ax.plot(distance,sign*data['frame_'+quantity][index],color=color,label=f'{update} updates',lw=1.3)
            if quantity=='slopes':ax.axhline(0,color='black',lw=.7)
            phase='early' if name.endswith('Rearly') else 'late'
            pool=study.pooled(study.factor_data(seed,0 if phase=='early' else 128))
            source_distance=(1/pool[:,1]).numpy()
            low,high=np.quantile(source_distance,[.05,.95])
            ax.axvspan(low,high,color='#999999',alpha=.12,label='90% of R observations')
            ax.set(title='Alignment: '+name.split('_')[1][1:]+'; SIGReg: '+name.split('_')[2][1:],xscale='log',xlabel='Physical distance (log scale)',
                   ylabel='Raw code, fixed initial sign' if quantity=='curves' else 'Code slope per log-distance')
            ax.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'Seed 200: {quantity}; no per-frame centering/scaling',y=1.02)
        save(fig,'factorial_raw_'+quantity)

    diagnostic_path=OUT/'MARGINAL_DIAGNOSTICS.json'
    quantile_path=OUT/'monotone_references.npz'
    if diagnostic_path.exists() and quantile_path.exists():
        diagnostics=json.loads(diagnostic_path.read_text())['rows'];quantiles=np.load(quantile_path)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for phase,color in [('early','#377eb8'),('late','#d95f02')]:
            name=f'factor_Aearly_R{phase}'
            data=np.load(OUT/f'{name}_seed200.npz')
            sign=1 if np.cov(np.log(data['frame_distance']),data['frame_curves'][0])[0,1]>=0 else -1
            axes[0].plot(quantiles['distance'],quantiles[name+'_seed200'],color=color,ls='--',label=f'Monotone reference, R {phase}')
            axes[0].plot(data['frame_distance'],sign*data['frame_curves'][-1],color=color,label=f'Network, R {phase}')
            group=[r for r in diagnostics if r['condition']==name]
            for index,row in enumerate(group):
                axes[1].plot([0 if phase=='early' else 1, .3 if phase=='early' else 1.3],
                             [row['before']['expected_total'],row['after']['expected_total']],color=color,alpha=.4)
        axes[0].set(xscale='log',xlabel='Physical distance',ylabel='Code value',title='Gaussianization can change spacing without folding')
        axes[0].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);axes[0].legend(fontsize=7)
        axes[1].set(xticks=[.15,1.15],xticklabels=['R early: before → after','R late: before → after'],
                    ylabel='Expected implemented objective',title='Fitting the sampled loss can harm recovery',yscale='log')
        save(fig,'monotone_reference_and_objective')


def zoo_figures(records):
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for index,policy in enumerate(study.ZOO_POLICIES):
        for source,offset,color in [('native',-.15,'#d95f02'),('reset',.15,'#377eb8')]:
            group=[r for r in records if r['condition']==f'zoo_{policy}_{source}']
            if not group:continue
            for ax,getter in [(axes[0],lambda r:r['metrics']['affine_log_r2']),
                              (axes[1],lambda r:r['planning']['shared']['goal_error']),
                              (axes[2],lambda r:r['planning']['native']['goal_error'])]:
                values=[getter(r) for r in group]
                ax.scatter(index+offset+np.linspace(-.06,.06,len(group)),values,color=color,s=13,alpha=.5)
                ax.plot([index+offset-.08,index+offset+.08],[np.mean(values)]*2,color=color,lw=3)
    for ax in axes:
        ax.set_xticks(range(7),[LABELS[p] for p in study.ZOO_POLICIES],rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set(title='Same regularizer exposure',ylabel='Affine log-distance R² ↑')
    axes[1].set(title='Shared predictor data',ylabel='Actual goal error ↓')
    axes[2].set(title='Policy/source predictor data',ylabel='Actual goal error ↓')
    fig.suptitle('Original policy rules; orange = native alignment, blue = common-reset alignment',y=1.03)
    save(fig,'original_zoo_common_exposure')


def birth_figures():
    path=OUT/'BIRTH_REPLAY_SUMMARY.json'
    if not path.exists():return
    summaries=json.loads(path.read_text())['rows']
    births=[r for r in summaries if r['first_fold_update'] is not None]
    if not births:return
    fig,axes=plt.subplots(len(births),2,figsize=(13,3*len(births)),squeeze=False)
    for row_axes,row in zip(axes,births):
        saved=torch.load(OUT/'birth_replays'/f"{row['condition']}_seed{row['seed']}.pt",map_location='cpu')
        attr=saved['attribution'];before=saved['before'];c=study.original.config()
        model=study.behavior.Encoder();model.load_state_dict(before)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(saved['optimizer_before'])
        optimizer.param_groups[0]['lr']=study.controls.learning_rate(saved['update'],c.jepa_steps)
        for name,p in model.named_parameters():p.grad=attr['gradient_alignment'][name]+attr['gradient_regularizer'][name]
        optimizer.step()
        q=torch.linspace(0,np.log(16),1025,requires_grad=True)
        def evaluate(state=None):
            if state is not None:model.load_state_dict(state)
            y=model(study.behavior.sensor(q.exp(),c));s=torch.autograd.grad(y.sum(),q)[0]
            return y.detach().numpy(),s.detach().numpy()
        actual_y,actual_slope=evaluate()
        before_y,before_slope=evaluate(before)
        sign=1 if np.cov(q.detach().numpy(),before_y)[0,1]>=0 else -1
        location=int(np.argmin(sign*actual_slope));distance=float(q[location].exp())
        values=[sign*before_slope[location],sign*actual_slope[location]]
        names=['Before','Actual update'];directions=attr['directions']
        for label,include in [('SIGReg contribution',['regularizer']),('Omit SIGReg',['carry','alignment'])]:
            _,slope=evaluate({n:before[n]+sum(directions[k][n] for k in include) for n in before})
            names.append(label);values.append(sign*slope[location])
        row_axes[0].bar(range(4),values,color=['#888888','black','#d95f02','#377eb8'])
        row_axes[0].axhline(0,color='black',lw=.6);row_axes[0].set_xticks(range(4),names,rotation=15)
        row_axes[0].set(title=f"Seed {row['seed']}, {row['condition']}, update {row['first_fold_update']}\nSlope at d={distance:.3f}",ylabel='Oriented slope per log-distance')
        for key,color in [('carry','#999999'),('alignment','#377eb8'),('regularizer','#d95f02')]:
            d=attr['after']['distance']
            row_axes[1].plot(d,sign*attr['components'][key]['delta_slope_jvp'],color=color,label=key)
        row_axes[1].plot(attr['after']['distance'],sign*(attr['after']['slope']-attr['slope_before']),color='black',ls='--',label='Actual slope change')
        row_axes[1].set(xscale='log',xlabel='Physical distance',ylabel='One-update slope change',title='Shared weights transmit the update across distances')
        row_axes[1].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);row_axes[1].legend(fontsize=7)
    fig.suptitle('Every first-fold event in the preselected five-seed/four-cell replay cohort\nComponent interventions use the same combined Adam denominator; no separate optimizers',y=1.01)
    save(fig,'first_fold_component_effects')


def region_figures():
    path=OUT/'GRADIENT_REGIONS.json'
    if not path.exists():return
    records=json.loads(path.read_text())['rows']
    arrays=np.load(OUT/'gradient_region_impulses.npz')
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    regions=['near','middle','far'];colors=['#d95f02','#009e73','#7570b3']
    lookup={(r['condition'],r['seed'],r['update']):r for r in records}
    for index,phase in enumerate(['early','late']):
        row=lookup[f'factor_Aearly_R{phase}',200,0]
        bottom=0.
        for region,color in zip(regions,colors):
            count=row['regions'][region]['sample_count']/1024
            axes[0].bar(index,count,bottom=bottom,color=color,label=region if index==0 else None)
            bottom+=count
    axes[0].set(xticks=[0,1],xticklabels=['Early observations','Late observations'],
                ylabel='Fraction of the actual regularizer batch',title='Seed 200: which examples drive the update?')
    axes[0].legend(fontsize=8)
    total=0
    for region,color in zip(regions,colors):
        impulse=arrays[f'factor_Aearly_Rlate_seed200_update0_{region}']
        total=total+impulse
        axes[1].plot(arrays['distance'],impulse,color=color,label=region+' source examples')
    axes[1].plot(arrays['distance'],total,color='black',ls='--',label='Sum of source groups')
    axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set(xscale='log',xlabel='Where the slope is evaluated',ylabel='Predicted slope change from current SIGReg impulse',
                title='Near examples can flatten distant slopes')
    axes[1].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16']);axes[1].legend(fontsize=7)
    for seed in range(200,205):
        values=[lookup[f'factor_Aearly_R{phase}',seed,0]['regions']['near']['mean_slope_impulse_at_far'] for phase in ['early','late']]
        axes[2].plot([0,1],values,'o-',alpha=.7,label=str(seed))
    axes[2].axhline(0,color='gray',lw=.7)
    axes[2].set(xticks=[0,1],xticklabels=['Early observations','Late observations'],
                ylabel='Near-example impulse to mean slope at d ≥ 4',title='All five preselected actors, before first update')
    axes[2].legend(fontsize=7)
    fig.suptitle('Source bins: near <2; middle 2–4; far ≥4. Full-batch chain-rule coefficients are retained.\nCurrent impulse only; past Adam history is not allocated to source regions.',y=1.05)
    save(fig,'near_observations_global_slope_effect')


def write_report(records,inference):
    lines=['# Geometry, exposure and original behavior policies','',
           f"Completed {len(records)}/360 encoders and {sum(len(r['planning']) for r in records)}/640 planning evaluations. [Protocol](../PROTOCOL.md) · [Zoo recap](../ZOO_RECAP.md) · [Mathematics](../MATH.md).",'',
           '## Which loss source changes geometry?','',
           'All factorial cells start from the same early encoder/Adam state within each seed. A is the alignment-pair source; R is independently sampled pooled regularizer observations. The SIGReg formula/weight is unchanged.','',
           '| A source | R source | Seeds | Recovery R² | Folds | Shared-data goal error |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for a in ('early','late'):
        for r in ('early','late'):
            group=[row for row in records if row['condition']==f'factor_A{a}_R{r}']
            if group:
                lines += [f"| {a} | {r} | {len(group)} | {np.mean([x['metrics']['affine_log_r2'] for x in group]):.4f} | {sum(x['metrics']['folded'] for x in group)}/{len(group)} | {np.mean([x['planning']['shared']['goal_error'] for x in group]):.4f} |"]
    lines += ['', 'Early R does not guarantee no folds: the independent-observation sampling protocol can differ from the original coupled-batch controls. These differences are included in the planned tests.','',
              '![Factorial recovery and planning](factorial_recovery_and_planning.png)','',
              '![Raw code movement](factorial_raw_curves.png)','',
              '![Slope movement](factorial_raw_slopes.png)','',
              '![Monotone reference](monotone_reference_and_objective.png)','',
              'The monotone Gaussian-quantile reference shows that distribution matching can change spacing without requiring a fold. Finite-network optimization and parameter sharing must explain the actual reversal; the quantile reference is not asserted to be the full-loss optimizer.','',
              '## Original zoo policies with common exposure','',
              'The same reset-state observations and regularizer input multiset are supplied to every policy/source condition within a seed. Original actors and colored-action streams are retained. Successors, actions and stationarity are not forced to match. Native alignment keeps the original hash-verified trajectory pairs. The broad R pool supplies real observations beyond native gaps; this is not pure reweighting.','',
              '| Policy | A source | Seeds | Recovery R² | Folds | Shared-data error | Native-data error |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for policy in study.ZOO_POLICIES:
        for source in ('native','reset'):
            group=[r for r in records if r['condition']==f'zoo_{policy}_{source}']
            if group:
                lines += [f"| {LABELS[policy]} | {source} | {len(group)} | {np.mean([r['metrics']['affine_log_r2'] for r in group]):.4f} | {sum(r['metrics']['folded'] for r in group)}/{len(group)} | {np.mean([r['planning']['shared']['goal_error'] for r in group]):.4f} | {np.mean([r['planning']['native']['goal_error'] for r in group]):.4f} |"]
    lines += ['', '![Original zoo with fixed exposure](original_zoo_common_exposure.png)','',
              '## Local explanation of a fold','',
              '![Source observations and global slope changes](near_observations_global_slope_effect.png)','',
              'The regional diagnostic keeps the full batch’s SIGReg coefficients and splits their parameter gradient by source distance. It does not apply separate losses to bins. These are current-gradient impulses, not all past regularizer momentum. All five preselected first-update cases are shown, including actors that never develop a fold.','',
              'The attribution splits carried momentum and post-switch alignment/regularizer moment histories with the same actual Adam denominator. The component-only update is a one-step intervention, not a claim about separately trained optimizers. Every preselected case, including no-fold controls, is retained.','']
    birth=OUT/'BIRTH_REPLAY_SUMMARY.json'
    if birth.exists():
        replay=json.loads(birth.read_text())['rows'];events=[r for r in replay if r['first_fold_update'] is not None]
        lines += [f"All {len(replay)} preselected fits reproduce final model and optimizer states exactly. There are {len(events)} first-fold events. The R contribution alone creates the detected fold in {sum(r['birth_component_fold_flags']['regularizer_only'] for r in events)}/{len(events)} events; omitting R prevents it in {sum(not r['birth_component_fold_flags']['without_regularizer'] for r in events)}/{len(events)}. This local cohort is not a new population significance test.",'',
                  '![All first-fold events](first_fold_component_effects.png)','']
    lines += ['## Planned statistical comparisons','',
              'One Holm family covers all 54 tests. Positive simple-source effects mean a benefit from the earlier source or from common-reset alignment. Interactions are signed differences of effects and are not automatically beneficial. Intervals concern unadjusted paired means; sign tests concern seed-wise direction. All cases/seeds are retained.','',
              f"Complete factor seed blocks: {len(inference['factor_complete_seeds'])}/20; zoo blocks: {len(inference['zoo_complete_seeds'])}/20.",'',
              '| Test | Seeds | Mean gain [95% interval] | Positive / negative | Holm p |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in inference['tests']:
        if not row['n']:continue
        lo,hi=row['ci95']
        lines += [f"| {row['test']} | {row['n']} | {row['mean_gain']:+.4g} [{lo:+.4g}, {hi:+.4g}] | {row['wins']} / {row['losses']} | {row['holm_p']:.4g} |"]
    lines += ['', '[Definitions and every seed effect](INFERENCE.json).','',
              '## Requested test: do matched policies differ?','',
              'The requested [equivalence analysis](EQUIVALENCE_SUMMARY.md) uses paired TOST and requires every policy pair to pass. At exploratory tolerances ±0.001 R² and ±0.005 physical goal error, recovery and shared-data planning pass in both matched studies; policy-specific-data planning does not. [Full tolerance sensitivity](MATCHED_EQUIVALENCE.md). [Complementary difference tests](MATCHED_OMNIBUS.md) detect no systematic policy effect, but nonsignificance alone is not equivalence. These post hoc analyses do not replace the 54 planned comparisons.','',
              '## Limits','',
              '- Factorial sampling differs from the original paired finite-batch regularizer. Matching-source comparisons with old coupled controls quantify this protocol change; source-swap effects are conditional on the new sampler.',
              '- Common exposure adds broad real observations; it does not prove that late data can be repaired solely by resampling its existing support.',
              '- Fixed-start controls retain actor weights/action rules but change data collection. They are neither stationary trajectory comparisons nor exact matches of both endpoint marginals.',
              '- Shared predictor data better isolates representation effects for this predictor/planner. Native-data differences also include action/state support. The inherited shared source omits outward reflected actions at grid endpoints.',
              '- Numerical alias witnesses, dense-grid folds and affine recovery are separate empirical diagnostics. No population identifiability or global-convergence theorem is claimed.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def main():
    torch.set_num_threads(1)
    records=rows();inference=stats.comparisons(records)
    study.save_json(OUT/'INFERENCE.json',inference)
    factor_figures(records);zoo_figures(records);birth_figures();region_figures()
    write_report(records,inference)
    print('Analysis:',len(records),'encoders; factor seeds',len(inference['factor_complete_seeds']),'zoo seeds',len(inference['zoo_complete_seeds']))


if __name__=='__main__':main()

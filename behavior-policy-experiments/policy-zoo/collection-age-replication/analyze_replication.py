"""Figures and concise conclusions for the prespecified fresh-seed study."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kempner-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/kempner-cache')

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as study
import replication_stats as stats

OUT = study.OUT
COLORS = {'icm': '#d95f02', 'random': '#377eb8', 'count': '#4daf4a', 'ou': '#333333'}
LABELS = {'icm': 'ICM', 'random': 'Random', 'count': 'Count', 'ou': 'Ideal OU'}
TEST_LABELS = {
    'early_vs_late_recovery': 'Early vs late ICM: recovery',
    'early_vs_late_folds': 'Early vs late ICM: fewer folds',
    'early_vs_late_planning': 'Early vs late ICM: reaching',
    'early_vs_random_recovery': 'Early ICM vs early random: recovery',
    'early_vs_random_planning': 'Early ICM vs early random: reaching',
    'early_vs_count_recovery': 'Early ICM vs early count: recovery',
    'early_vs_count_planning': 'Early ICM vs early count: reaching',
    'early_vs_ou_recovery': 'Early ICM vs ideal OU: recovery',
    'early_vs_ou_planning': 'Early ICM vs ideal OU: reaching',
    'early_vs_initial_recovery': 'Early ICM vs untouched encoder: recovery',
    'age_trend_recovery': 'ICM recovery decreases with collection age',
    'age_trend_planning': 'ICM goal error increases with collection age',
    'rescue_vs_random_recovery': 'ICM early/late recovery gain minus random gain',
    'rescue_vs_random_planning': 'ICM early/late reaching gain minus random gain',
    'rescue_vs_count_recovery': 'ICM early/late recovery gain minus count gain',
    'rescue_vs_count_planning': 'ICM early/late reaching gain minus count gain',
    'continuation_recovery': 'Early→early vs early→late: recovery',
    'continuation_folds': 'Early→early vs early→late: fewer folds',
    'continuation_planning': 'Early→early vs early→late: reaching',
}


def load_rows():
    rows = []
    for path in sorted(OUT.glob('*_seed*.json')):
        row = json.loads(path.read_text())
        if 'condition' in row:
            rows.append(row)
    return rows


def save(fig, name):
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=165, bbox_inches='tight')
    plt.close(fig)


def mean_interval(values):
    return float(np.mean(values)), stats.bootstrap(values)


def line(ax, x, groups, color, label):
    means, lower, upper = [], [], []
    for values in groups:
        mean, (lo, hi) = mean_interval(values)
        means.append(mean)
        lower.append(lo)
        upper.append(hi)
    ax.plot(x, means, 'o-', color=color, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=.13)


def age_figures(rows):
    lookup = {(r['condition'], r['seed']): r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for policy in study.POLICIES:
        seeds = [s for s in study.SEEDS if all((study.label(policy,a),s) in lookup for a in study.AGES)]
        if not seeds:
            continue
        conditions = [[lookup[study.label(policy,age),s] for s in seeds] for age in study.AGES]
        specifications = [
            (axes[0,0], lambda r:r['metrics']['affine_log_r2']),
            (axes[0,1], lambda r:int(r['metrics']['folded'])),
            (axes[0,2], lambda r:r['planning']['shared']['goal_error']),
            (axes[1,0], lambda r:r['exposure']['mass_below_2']),
            (axes[1,1], lambda r:r['exposure']['normalized_log_bin_entropy']),
            (axes[1,2], lambda r:r['planning']['native']['goal_error']),
        ]
        for ax, getter in specifications:
            line(ax, study.AGES, [[getter(r) for r in group] for group in conditions], COLORS[policy], f'{LABELS[policy]} (n={len(seeds)})')
    titles = ['Linear recovery', 'Detected folds', 'Shared predictor data',
              'Concentration near the object', 'Spread across log-distance bins', 'Policy-native predictor data']
    ylabels = ['Affine log-distance R² ↑', 'Fraction folded', 'Actual goal error ↓',
               'Endpoint mass below distance 2', 'Normalized bin entropy', 'Actual goal error ↓']
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set(title=title, xlabel='Discarded steps after reset', ylabel=ylabel)
        ax.set_xticks(study.AGES)
        ax.grid(alpha=.2)
    axes[0,1].set_ylim(-.03,1.03)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fresh actors, same data/training budget; bands are unadjusted seed-bootstrap intervals', y=1.02)
    save(fig, 'collection_age_effects')

    # All 20 ICM seeds are shown in two fixed, readable galleries, not chosen by success.
    for block, seeds in enumerate((study.SEEDS[:10], study.SEEDS[10:]), 1):
        if not any((study.label('icm',a),s) in lookup for s in seeds for a in study.AGES):
            continue
        fig, axes = plt.subplots(10, 5, figsize=(15, 23), sharex=True, sharey=True)
        for i, seed in enumerate(seeds):
            for j, age in enumerate(study.AGES):
                ax = axes[i,j]
                name = study.label('icm', age)
                row = lookup.get((name,seed))
                if row is None:
                    ax.set_visible(False)
                    continue
                arrays = np.load(OUT / f'{name}_seed{seed}.npz')
                d, y = arrays['distance'], arrays['curve']
                q = np.log(d)
                q = (q-q.mean())/q.std()
                ax.plot(d,q,':',color='gray',lw=1)
                ax.plot(d,y,color=COLORS['icm'],lw=1.4)
                ax.set(xscale='log', title=f"Seed {seed}, age {age}\nR²={row['metrics']['affine_log_r2']:.3f}; fold={int(row['metrics']['folded'])}")
                ax.set_xticks([1,2,4,8,16], labels=['1','2','4','8','16'])
                ax.grid(alpha=.15)
        fig.suptitle('Each row: one frozen ICM actor and encoder initialization; columns change collection age\nOnly affine display normalization; gray = affine log-distance', y=1.01)
        for ax in axes[-1]:
            ax.set_xlabel('Physical distance (log scale)')
        for ax in axes[:,0]:
            ax.set_ylabel('Normalized code')
        save(fig, f'icm_age_gallery_{block}')


def continuation_figures(rows):
    lookup = {(r['condition'],r['seed']):r for r in rows}
    names = ('icm_early_to_early', 'icm_early_to_late')
    seeds = [s for s in study.SEEDS if all((name,s) in lookup for name in names)]
    if not seeds:
        return
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for name, color, label in [(names[0], '#377eb8', 'Keep early data'), (names[1], '#d95f02', 'Switch to late data')]:
        group = [lookup[name,s] for s in seeds]
        updates = [h['step'] for h in group[0]['history']]
        for ax, getter in [(axes[0], lambda h:h['affine_log_r2']), (axes[1], lambda h:int(h['folded']))]:
            values = [[getter(r['history'][i]) for r in group] for i in range(len(updates))]
            line(ax, updates, values, color, label)
        positions = 0 if name == names[0] else 1
        errors = [r['planning']['shared']['goal_error'] for r in group]
        axes[2].scatter(positions+np.linspace(-.1,.1,len(errors)),errors,color=color,alpha=.6)
        axes[2].plot([positions-.2,positions+.2],[np.mean(errors)]*2,color=color,lw=3)
    axes[0].set(xlabel='Additional training updates',ylabel='Affine log-distance R²',title='Start from the same early-trained encoder')
    axes[1].set(xlabel='Additional training updates',ylabel='Fraction folded',title='Does switching data introduce folds?',ylim=(-.03,1.03))
    axes[2].set(xticks=[0,1],xticklabels=['Keep early','Switch late'],ylabel='Shared-data actual goal error',title='Same predictor data and physical goals')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, 'continuation_effects')

    fig, axes = plt.subplots(4,5,figsize=(15,10),sharex=True,sharey=True)
    for ax,seed in zip(axes.flat,study.SEEDS):
        if seed not in seeds:
            ax.set_visible(False)
            continue
        for name,color,label in [('icm_age000','#999999','Before continuation'),
                                 (names[0],'#377eb8','Keep early'),(names[1],'#d95f02','Switch late')]:
            arrays = np.load(OUT/f'{name}_seed{seed}.npz')
            ax.plot(arrays['distance'],arrays['curve'],color=color,label=label,lw=1.2)
        a,b=lookup[names[0],seed],lookup[names[1],seed]
        ax.set(xscale='log',title=f"Seed {seed}: folds {int(a['metrics']['folded'])}/{int(b['metrics']['folded'])}")
        ax.set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('Physical distance (log scale)')
    for ax in axes[:,0]:
        ax.set_ylabel('Normalized code')
    fig.suptitle('Both continuations start with identical encoder weights and Adam state\nCurves independently normalized only for display; no selected successful subset',y=1.02)
    save(fig,'continuation_gallery')


def judgment(test):
    if test['holm_p'] >= .05:
        return 'not established'
    if test['mean_oriented_effect'] > 0 and test['wins'] > test['losses']:
        return 'consistent gain in planned direction'
    if test['mean_oriented_effect'] < 0 and test['losses'] > test['wins']:
        return 'consistent effect in opposite direction'
    return 'direction/mean differ; inspect seed effects'


def write_report(rows, inference):
    lookup = {(r['condition'],r['seed']):r for r in rows}
    complete = not inference['pending_seeds']
    tests = {r['test']:r for r in inference['tests']}
    lines = ['# Does early ICM help, and does later data create folds?', '',
             f"**Status:** {len(rows)}/360 fitted encoder conditions; {sum(len(r['planning']) for r in rows)}/700 predictor evaluations. Fresh seeds 200–219. [Fixed protocol](../PROTOCOL.md).", '',
             '**Early means trajectory steps after a reset, with an already trained and frozen behavior actor.** It does not mean early ICM policy-training epochs. Separate-window fits test data usefulness; the continuation experiment tests changing data for the same encoder.', '']
    if complete:
        lines += ['## Answers on the fresh cohort', '']
        for name in ['early_vs_late_recovery','early_vs_late_folds','early_vs_late_planning',
                     'early_vs_random_recovery','early_vs_random_planning','early_vs_count_recovery',
                     'early_vs_count_planning','early_vs_ou_recovery','early_vs_ou_planning',
                     'early_vs_initial_recovery','continuation_recovery','continuation_folds','continuation_planning']:
            row = tests[name]
            lines += [f"- **{TEST_LABELS[name]}:** {judgment(row)}; mean oriented effect {row['mean_oriented_effect']:+.4g}, wins/losses {row['wins']}/{row['losses']}, Holm p={row['holm_p']:.4g}."]
        lines += ['']
    else:
        lines += ['**Primary inference pending:** all required ages, baselines and continuation arms must finish within each seed before it enters the 19-test family. Partial descriptive means below are not final conclusions.', '']
    lines += ['## Same-budget policy and age comparison', '',
              '| Policy | Age | Seeds | Affine log-distance R² ↑ | Folds | Rank correlation ↑ | Shared goal error ↓ | Native goal error ↓ | Near-object mass |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for policy in (*study.POLICIES, 'ou'):
        for age in (study.AGES if policy != 'ou' else [-1]):
            name=study.label(policy,age)
            group=[r for r in rows if r['condition']==name]
            if not group:
                continue
            mean=lambda getter:float(np.mean([getter(r) for r in group]))
            native = f"{mean(lambda r:r['planning']['native']['goal_error']):.4f}" if policy != 'ou' else '—'
            mass = f"{mean(lambda r:r['exposure']['mass_below_2']):.3f}" if policy != 'ou' else '—'
            lines += [f"| {LABELS[policy]} | {age if age>=0 else '—'} | {len(group)} | {mean(lambda r:r['metrics']['affine_log_r2']):.4f} | {sum(r['metrics']['folded'] for r in group)}/{len(group)} | {mean(lambda r:r['metrics']['absolute_rank_correlation']):.4f} | {mean(lambda r:r['planning']['shared']['goal_error']):.4f} | {native} | {mass} |"]
    initial=[lookup['icm_age000',s]['initial_metrics'] for s in study.SEEDS if ('icm_age000',s) in lookup]
    if initial:
        lines += ['',f"Untouched encoders: mean recovery R² {np.mean([r['affine_log_r2'] for r in initial]):.4f}; folds {sum(r['folded'] for r in initial)}/{len(initial)}. The initial architecture can already provide substantial recovery; learning gains must be assessed relative to it."]
    lines += ['', 'Near-object mass is pooled endpoint fraction below physical distance 2. Early data are transient and different policies retain their real occupancy differences. No stationary matched-marginal claim is made. Same 16,384 pairs, 4,000 representation updates and fixed SIGReg for every main condition.', '',
              '![Collection age, recovery, folds and actual reaching](collection_age_effects.png)', '',
              '[ICM curves, seeds 200–209](icm_age_gallery_1.png) · [Seeds 210–219](icm_age_gallery_2.png). Every fresh seed is displayed; these galleries are not selected by success.', '',
              '## Does later data introduce folds in the same encoder?', '',
              'Both branches start from each seed’s age-0 ICM encoder and the exact same saved Adam state. Both restart the learning-rate clock and receive 4,000 more updates. Only the continuation dataset changes. No initial-success subset is used.', '',
              '| Continuation | Seeds | Final R² | Final folds | New folds among all seeds | Shared goal error | Native goal error |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,title in [('icm_early_to_early','Keep early data'),('icm_early_to_late','Switch to late data')]:
        group=[r for r in rows if r['condition']==name]
        if group:
            new=sum(not r['initial_metrics']['folded'] and r['metrics']['folded'] for r in group)
            lines += [f"| {title} | {len(group)} | {np.mean([r['metrics']['affine_log_r2'] for r in group]):.4f} | {sum(r['metrics']['folded'] for r in group)}/{len(group)} | {new}/{len(group)} | {np.mean([r['planning']['shared']['goal_error'] for r in group]):.4f} | {np.mean([r['planning']['native']['goal_error'] for r in group]):.4f} |"]
    lines += ['', 'New-fold counts are descriptive; primary paired tests retain all 20 seeds. History records locate fold onset only to 500-update intervals.', '',
              '![Continuation histories and reaching](continuation_effects.png)', '',
              '[Every continuation representation](continuation_gallery.png).', '',
              '## One family of 19 planned tests', '',
              'Positive effects mean the direction written in the test label. For reaching they mean lower actual physical goal error, not lower latent predictor error. Sign tests concern consistent seed-wise direction; intervals concern mean effects and are unadjusted. Fold tests use discordant pairs. All tests share one Holm correction.', '',
              '| Test | Seeds | Mean effect [95% interval] | Positive / negative | Holm p |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in inference['tests']:
        lo,hi=row['mean_ci95']
        lines += [f"| {TEST_LABELS[row['test']]} | {row['n']} | {row['mean_oriented_effect']:+.4g} [{lo:+.4g}, {hi:+.4g}] | {row['wins']} / {row['losses']} | {row['holm_p']:.4g} |"]
    lines += ['', '[All seed effects and exact p-values](INFERENCE.json).', '',
              '## What these experiments can establish', '',
              '- A fresh early-versus-late improvement supports a collection-phase rescue. It does not establish that early ICM beats random, count or OU; those have separate planned comparisons.',
              '- Late-window fits from scratch are not evidence that a single encoder deteriorated over time. The equal-budget continuation tests address that stronger question.',
              '- Collection age changes occupancy, actions, reflections and empirical transition statistics together. Concentration is measured, not separately randomized as the sole cause.',
              '- Common predictor data and goals hold task evaluation fixed. Native data additionally changes state/action exposure. Every physical goal has zero oracle error. The inherited common source omits outward reflected actions at the two endpoints; scores remain conditional on this predictor/planner and its training coverage.',
              '- Affine recovery, order preservation and loss of information are distinct. Absence of a grid fold is not a proof of injectivity; a high R² is not a population identification theorem.',
              '- The ideal OU reference satisfies a different data law from the bounded physical policies. This is an empirical same-budget comparison in the existing distance-only setup.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def main():
    rows=load_rows()
    inference=stats.comparisons(rows,study.SEEDS,study.AGES)
    study.save_json(OUT/'INFERENCE.json',inference)
    age_figures(rows)
    continuation_figures(rows)
    write_report(rows,inference)
    print('Analyzed encoder conditions:',len(rows),'complete seed blocks:',len(inference['complete_seeds']))


if __name__=='__main__':
    main()

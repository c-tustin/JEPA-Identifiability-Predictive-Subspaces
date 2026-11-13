"""Explain saved experiments; no model training or scientific-file mutations.

Run from anywhere with Python, NumPy and Matplotlib. Inputs are the saved
60-seed cohort JSON results and source manifests; intervals are saved TOST
intervals, not recalculated or selected for significance.
"""
import json
import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/kempner-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Support both the original desktop workspace and the experiment folder on main.
SOURCE_ROOT = ROOT / 'predictive-closure-workspace' if (ROOT / 'predictive-closure-workspace').is_dir() else ROOT
OUT = SOURCE_ROOT / 'behavior-policy-experiments/policy-zoo/confirmatory-matched-exposure/output'
read = lambda path: json.loads(path.read_text())
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
blue, orange = '#176B8F', '#D76A29'

inference = read(OUT / 'INFERENCE.json')
fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
labels = {'random': 'Random', 'count': 'Count', 'rnd': 'RND', 'disagreement': 'Disagreement',
          'ou_actions_lp_w0.1': 'OU action controller: LP', 'ou_actions_hp_w0.3': 'OU action controller: HP'}
for ax, key, title, zoom in zip(axes,
        ['affine_log_r2', 'affine_log_r2', 'shared_goal_error'],
        ['Recovery R²: full tolerance', 'Recovery R²: interval zoom', 'Shared-data physical goal error'],
        [False, True, False]):
    result = inference['primary_endpoints'][key]
    rows = [p for p in result['pairs'] if p['policy_a'] == 'icm']
    margin = result['margin']
    display_scale = 1e6 if zoom else 1
    ax.axvspan(-margin * display_scale, margin * display_scale, color='#DCEFE2')
    ax.axvline(0, color='gray', lw=1)
    for i, row in enumerate(rows):
        mean = row['mean_difference'] * display_scale
        lo, hi = np.array(row['mean_ci90']) * display_scale
        ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color=blue, capsize=4)
    ax.set_yticks(range(len(rows)), [labels[r['policy_b']] for r in rows])
    ax.invert_yaxis(); ax.set_xlim(-margin*1.12, margin*1.12)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    if zoom:
        ax.set_xlim(-8, 8)
    ax.set_title(title + ('\nBounds ±0.001 are outside this zoom' if zoom else f'\nEquivalence bounds: ±{margin:g}'))
    ax.set_xlabel('Mean difference and 90% CI (×10⁻⁶ R²)' if zoom else 'ICM minus comparison; mean and 90% CI')
fig.suptitle('TOST: are differences small enough?  |  60 paired seeds\nGreen = acceptable difference; displayed contrasts are 6 of the 21 tested pairs', fontsize=13)
fig.savefig(HERE / 'tost_explained.png', dpi=180)
plt.close(fig)

native = [read(OUT / f'icm_native_exposure_seed{s}.json') for s in range(6000, 6060)]
common = [read(OUT / f'icm_common_exposure_seed{s}.json') for s in range(6000, 6060)]
hist = np.array([read(OUT / f'source_manifests/icm_seed{s}.json')['native_exposure']['log_bin_probability'] for s in range(6000,6060)])
assert hist.shape == (60, 32) and np.allclose(hist.sum(1), 1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), layout='constrained')
centers = np.exp((np.arange(32)+.5)*np.log(16)/32)
axes[0].plot(centers, hist.mean(0)*100, color=orange, lw=2, label='Native ICM pooled endpoints')
axes[0].axhline(100/32, color=blue, lw=2, label='Additional common single views')
axes[0].set_xscale('log', base=2); axes[0].set_xticks([1,2,4,8,16], ['1','2','4','8','16'])
axes[0].set(xlabel='Distance (equal log-distance bins)', ylabel='Observations per bin (%)', title='What SIGReg sees: input exposure')
axes[0].legend(fontsize=8)
for ax, metric, title in [(axes[1], 'r2', 'Linear recovery improves'), (axes[2], 'goal', 'Actual reaching improves')]:
    extract = lambda r: r['metrics']['affine_log_r2'] if metric == 'r2' else r['planning']['shared']['goal_error']
    a, b = np.array(list(map(extract,native))), np.array(list(map(extract,common)))
    for x,y in zip(a,b): ax.plot([0,1], [x,y], color='gray', alpha=.22, lw=.7)
    ax.scatter(np.zeros(60),a,s=9,color=orange,alpha=.65)
    ax.scatter(np.ones(60),b,s=9,color=blue,alpha=.65)
    ax.plot([0,1], [a.mean(),b.mean()], 'o-', color='black', lw=2, label='Cohort mean')
    ax.set_xticks([0,1], ['Native\nSIGReg pool','Broad common\nSIGReg pool'])
    ax.set_title(title)
    ax.set_ylabel('Held-out log-distance R²' if metric == 'r2' else 'Physical goal error (distance units)')
    ax.text(.5,.96,f'Means: {a.mean():.4f} → {b.mean():.4f}',ha='center',va='top',transform=ax.transAxes)
    if metric == 'r2':
        # Reveal genuine seed variation that overlaps on the full 0–1 scale.
        inset = ax.inset_axes([.46, .11, .49, .40])
        inset.scatter(np.arange(1,61), b, s=8, color=blue)
        inset.set_ylim(.992, 1.000)
        inset.ticklabel_format(axis='y', style='plain', useOffset=False)
        inset.set_title('Broad pool: zoom by seed', fontsize=8)
        inset.set_xlabel('Seed index', fontsize=8)
        inset.tick_params(labelsize=7)
    ax.margins(x=.18)
fig.suptitle('Change only SIGReg observations  |  Same ICM alignment pairs, initial weights and training budget\n60 seeds; additional common views come from simulator resets, not ICM exploration', fontsize=13)
fig.savefig(HERE / 'sigreg_exposure_and_outcomes.png', dpi=180)
plt.close(fig)
summary = {'inputs': str(OUT.relative_to(SOURCE_ROOT)), 'seeds': 60, 'native_folds': sum(r['metrics']['folded'] for r in native),
           'common_folds': sum(r['metrics']['folded'] for r in common),
           'native_endpoint_mass_below_2': float(hist[:, :8].sum(1).mean()),
           'all_recovery_improved': all(b['metrics']['affine_log_r2'] > a['metrics']['affine_log_r2'] for a,b in zip(native, common)),
           'all_goal_error_improved': all(b['planning']['shared']['goal_error'] < a['planning']['shared']['goal_error'] for a,b in zip(native,common))}
(HERE / 'graph_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))

# Rebuild the offline bundle from an explicit file list, so removed figures
# cannot persist in either the HTML gallery or ZIP archive.
bundle = HERE / 'shareable-graphs'
bundle.mkdir(exist_ok=True)
for ext in ('png', 'pdf'):
    (bundle / f'slope_evolution_seed131.{ext}').unlink(missing_ok=True)
mechanism = SOURCE_ROOT / 'behavior-policy-experiments/qualitative-study/fold-location/training-mechanism/output'
history = [r for r in read(mechanism / 'root-history.json')
           if r['seed'] == 131 and r['gain'] == 4 and r['condition'] == 'ordinary']
points = [(r['step'], d) for r in history for d in r['turns_before']]
final = next(r for r in read(mechanism / 'root-events.json')
             if r['seed'] == 131 and r['gain'] == 4 and r['condition'] == 'ordinary')
points += [(4000, d) for d in final['turns_at_selected_steps']['4000']]
birth = read(mechanism / 'birth-event.json')
first = birth['after']['turning_distances'][0]
fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
ax.plot([p[0] for p in points], [p[1] for p in points], color='black', lw=1.5)
ax.scatter(birth['update_after'], first, color=orange, s=50, label=f'First detected in-range turn: {first:.2f}')
ax.axhspan(2, 4, color='.8', alpha=.3, label='Input-scale region 2–4')
ax.set(xlabel='Encoder updates completed', ylabel='Physical turning distance', ylim=(1, 16.5),
       title='First detected fold and inward migration\nICM seed 131 · input gain 4')
ax.grid(alpha=.15)
ax.legend(frameon=False, fontsize=9)
for ext in ('png', 'pdf'):
    fig.savefig(bundle / f'birth_and_inward_migration.{ext}', dpi=180)
plt.close(fig)
for name in ('tost_explained.png', 'sigreg_exposure_and_outcomes.png'):
    shutil.copy2(HERE / name, bundle / name)
for ext in ('png', 'pdf'):
    shutil.copy2(mechanism / f'figures/leftward_curve_and_slope.{ext}', bundle / f'leftward_curve_and_slope.{ext}')
titles = {'leftward_curve_and_slope': 'Folded representation at different training times',
          'birth_and_inward_migration': 'First detected fold and inward migration',
          'tost_explained': 'TOST policy comparisons',
          'sigreg_exposure_and_outcomes': 'SIGReg observations and outcomes'}
html = ['<!doctype html><html><head><meta charset="utf-8"><title>ICM folds, TOST and SIGReg graphs</title><style>body{font:17px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}img{max-width:100%}section{margin:50px 0}</style></head><body><h1>ICM folds, TOST and SIGReg</h1><p>Open this file in any browser. All images are local and work offline.</p>']
files = ['index.html']
for key, title in titles.items():
    html.append(f'<section><h2>{title}</h2><p><a href="{key}.png">Open full PNG</a>')
    files.append(f'{key}.png')
    if (bundle / f'{key}.pdf').exists():
        html.append(f' · <a href="{key}.pdf">PDF</a>')
        files.append(f'{key}.pdf')
    html.append(f'</p><img src="{key}.png" alt="{title}"></section>')
html.append('</body></html>')
(bundle / 'index.html').write_text(''.join(html))
with zipfile.ZipFile(HERE / 'shareable-graphs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(bundle / name, arcname='shareable-graphs/' + name)
print('Updated offline gallery and ZIP: 4 figures; no training heatmaps.')

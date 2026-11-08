"""Figures from the complete fixed cohort; no partial-cohort plots or selection."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kempner-matplotlib')
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import run_confirmation as run
import analyze_confirmation as analysis

LABELS = {'icm': 'ICM', 'random': 'Random', 'count': 'Count', 'rnd': 'RND',
          'disagreement': 'Disagree', 'ou_actions_lp_w0.1': 'OU-LP', 'ou_actions_hp_w0.3': 'OU-HP'}


def save(fig, name):
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(run.OUT / f'{name}.{extension}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def equivalence_figure(inference):
    fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    labels = []
    for ax, (key, result) in zip(axes, inference['primary_endpoints'].items()):
        labels = []
        for index, pair in enumerate(result['pairs']):
            lo, hi = np.array(pair['mean_ci90']) / result['margin']
            mean = pair['mean_difference'] / result['margin']
            color = '#16856c' if lo > -1 and hi < 1 else '#d65a21'
            ax.plot([lo, hi], [index, index], color=color, lw=2)
            ax.plot(mean, index, 'o', color=color, ms=3)
            labels.append(LABELS[pair['policy_a']] + ' − ' + LABELS[pair['policy_b']])
        ax.axvspan(-1, 1, alpha=.07, color='green')
        for bound in (-1, 1):
            ax.axvline(bound, color='gray', ls='--', lw=1)
        ax.axvline(0, color='black', lw=.6)
        ax.set_xlim(-1.15, 1.15)
        titles = {'affine_log_r2':'Whole-range recovery', 'near_r2':'Near-object recovery', 'far_r2':'Far-object recovery', 'shared_goal_error':'Physical goal error'}
        ax.set(title=titles[key], xlabel='Difference / fixed tolerance')
        ax.grid(axis='x', alpha=.15)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('All 21 policy pairs, 60 fresh seeds: pointwise 90% paired-t intervals\nDashed lines are the fixed tolerances; confirmatory decision uses the four-claim Holm family.')
    save(fig, 'fresh_equivalence')


def mechanism_figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    cells = ('native_exposure', 'common_exposure', 'matched')
    for ax, key in zip(axes[:2], ('affine_log_r2', 'shared_goal_error')):
        for seed in run.SEEDS:
            values = [analysis.endpoint(rows['icm', cell, seed], key) for cell in cells]
            ax.plot(range(3), values, color='gray', alpha=.2, lw=.6)
        means = [np.mean([analysis.endpoint(rows['icm', cell, seed], key) for seed in run.SEEDS]) for cell in cells]
        ax.plot(range(3), means, '-o', color='#d65a21', lw=2)
        ax.set_xticks(range(3), ['ICM pairs\nICM views', 'ICM pairs\nbroad views', 'Reset pairs\nbroad views'])
        ax.set(title='Log-distance recovery (R²)' if key == 'affine_log_r2' else 'Physical goal-reaching error')
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(axis='x', labelsize=9)
    counts = [sum(rows['icm', cell, seed]['metrics']['folded'] for seed in run.SEEDS) for cell in cells]
    axes[2].bar(range(3), counts, color=['#d65a21', '#16856c', '#397db5'])
    axes[2].set_xticks(range(3), ['ICM pairs\nICM views', 'ICM pairs\nbroad views', 'Reset pairs\nbroad views'])
    axes[2].set(title='Folded runs (out of 60)', ylim=(0, 63))
    axes[2].tick_params(axis='x', labelsize=9)
    for index, count in enumerate(counts):
        axes[2].text(index, count + 1, str(count), ha='center')
    fig.suptitle('ICM: change which observations SIGReg sees\nGray lines: 60 paired runs; orange: mean. First two conditions use identical transition pairs.')
    save(fig, 'icm_exposure_control')


def representation_figure():
    torch.set_num_threads(1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    distance = np.geomspace(1, 16, 1025)
    q = np.log(distance)
    c = run.original.config()
    cells = ('native_exposure', 'common_exposure', 'matched')
    colors = ('#d65a21', '#16856c', '#397db5')
    for ax, seed in zip(axes.flat, run.SEEDS[:6]):
        initial, _ = run.original.reliability.make_plan(c, seed, seed, 'icm')
        with torch.no_grad():
            x = run.behavior.sensor(torch.tensor(distance, dtype=torch.float32), c)
            y0 = initial(x).numpy()
        sign = 1 if np.cov(q, y0)[0, 1] >= 0 else -1
        for cell, color in zip(cells, colors):
            state = torch.load(run.OUT / f'encoder_icm_{cell}_seed{seed}.pt', map_location='cpu')
            model = run.behavior.Encoder()
            model.load_state_dict(state['encoder'])
            with torch.no_grad():
                y = sign * model(x).numpy()
            labels = {'native_exposure':'ICM pairs + ICM observations', 'common_exposure':'ICM pairs + broad observations', 'matched':'Reset pairs + broad observations'}
            ax.plot(distance, y, color=color, label=labels[cell])
        ax.set(xscale='log', title=f'Seed {seed}', xlabel='Physical distance', ylabel='Learned output')
        ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('First six seed IDs, fixed before inspecting their curves\nRaw outputs with one initial sign convention per seed; no per-curve unfolding or normalization.')
    save(fig, 'first_six_icm_curves')


def main():
    rows, missing = analysis.collect()
    assert not missing, 'No partial-cohort plots.'
    inference = json.loads((run.OUT / 'INFERENCE.json').read_text())
    assert inference['signature'] == run.signature()
    equivalence_figure(inference)
    mechanism_figure(rows)
    representation_figure()
    (run.OUT / 'FIGURES.md').write_text('# Fresh-cohort figures\n\n![All policy pairs](fresh_equivalence.png)\n\n![ICM exposure controls](icm_exposure_control.png)\n\n![First six ICM curves](first_six_icm_curves.png)\n\nThe blue and green broad-exposure curves nearly overlap. A decreasing monotone curve is a valid sign reversal, not a fold.\n\n[Results](RESULTS.md)\n')


if __name__ == '__main__':
    main()

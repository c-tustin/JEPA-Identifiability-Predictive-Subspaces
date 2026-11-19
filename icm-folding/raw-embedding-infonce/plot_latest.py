"""Latest raw/own-code InfoNCE only; no alternative-objective dependencies."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/icm-gradient-cache')
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as original

TEAL = '#087f8c'
OUT = original.OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=170, facecolor='white')
    plt.close(fig)


def main():
    original.torch.set_num_threads(1)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    records, geometries, checks = {}, {}, []
    previous = json.loads((OUT / 'verification.json').read_text())
    hashes = {r['seed']: r['sha256'] for r in previous['runs']}
    for seed in range(130, 145):
        path = OUT / 'runs' / f'{seed}-n4000.pt'
        assert digest(path) == hashes[seed]
        r = original.base.trials.load(path)
        c = original.base.B.Config(**r['config'])
        model = original.base.B.Encoder()
        model.load_state_dict(r['model'])
        g = original.base.geometry(model, c, 20001)
        assert g['folded'] == r['geometry']['folded']
        np.testing.assert_allclose(g['std'], r['geometry']['std'], rtol=.005)
        # geometry() differentiates with respect to q=log(distance).
        # Chain rule: dz/dd = (dz/dq)/distance, without display normalization.
        distance = np.exp(g['q'])
        slope = g['slope'] / distance
        numerical = np.gradient(g['y'], distance, edge_order=2)
        np.testing.assert_allclose(slope, numerical, rtol=.003, atol=2e-6)
        g['distance_slope'] = slope
        records[seed], geometries[seed] = r, g
        checks.append({'seed': seed, 'checkpoint_sha256': hashes[seed],
                       'points': len(distance), 'folded': g['folded'],
                       'distance_derivative_checked': True})

    fig, axes = plt.subplots(3, 5, figsize=(16, 9), layout='constrained')
    for seed, ax in zip(range(130, 145), axes.flat):
        g = geometries[seed]
        ax.plot(np.exp(g['q']), g['y'], color=TEAL, lw=2)
        ax.set(title=f'Seed {seed} · no detected fold' if not g['folded'] else f'Seed {seed} · detected fold',
               xlabel='Physical distance', ylabel='Learned code z\n(arbitrary units)', xlim=(1, 16))
        ax.grid(alpha=.2)
    fig.suptitle('Latest InfoNCE: learned representations of physical distance\n'
                 'Raw observation / own code · final update 4,000 · original code units, no sign flips', fontsize=15)
    save(fig, 'latest_distance_representations')

    g = geometries[131]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout='constrained')
    axes[0].plot(np.exp(g['q']), g['y'], color=TEAL, lw=2.5)
    axes[0].set(title='Physical distance → learned representation',
                ylabel='Learned code z (arbitrary units)')
    axes[1].plot(np.exp(g['q']), g['distance_slope'], color=TEAL, lw=2.5)
    axes[1].axhline(0, color='#777777', ls=':', lw=1)
    axes[1].set(title='Slope of the distance representation',
                ylabel='dz / d(physical distance)')
    for ax in axes:
        ax.set(xlabel='Physical distance (evaluation only)', xlim=(1, 16))
        ax.grid(alpha=.2)
    fig.suptitle('Latest raw / own-code InfoNCE only · seed 131 · update 4,000\n'
                 'Positive sampled slope throughout; flattening means compressed distance resolution', fontsize=13)
    save(fig, 'latest_seed131_distance')

    ranked = sorted(records, key=lambda s: (records[s]['prediction']['native_mse'], s))
    selected = [ranked[0], ranked[len(ranked)//2], ranked[-1]]
    roles = ['Lowest', 'Median', 'Highest']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    steps = (0, 100, 500, 2000, 4000)
    colors = ('#aeb5bd', '#92cbd0', '#4ea9b3', '#167d8a', '#064b54')
    for ax, seed, role in zip(axes, selected, roles):
        frames = {f['step']: f for f in records[seed]['frames']}
        for step, color in zip(steps, colors):
            frame = frames[step]
            ax.plot(np.exp(frame['q']), frame['y'], color=color, lw=2, label=f'{step:,}')
        ax.set(title=f'Seed {seed} · {role.lower()} final prediction error',
               xlabel='Physical distance (evaluation only)',
               ylabel='Learned code z (arbitrary units)', xlim=(1, 16))
        ax.grid(alpha=.2)
    axes[0].legend(title='Encoder updates', fontsize=9)
    fig.suptitle('Latest InfoNCE: learning a representation of physical distance\n'
                 'Raw observation / own code only · original code units, no sign flips', fontsize=14)
    save(fig, 'latest_distance_learning')

    original.base.trials.atomic_json(OUT / 'latest_plots_verification.json', {
        'date': '2026-11-19', 'method': 'raw-own-code-infonce',
        'plot_code_sha256': digest(original.HERE / 'plot_latest.py'),
        'training_unchanged': True, 'display_normalization': False,
        'selection': 'min, median, max saved final held-out decoder MSE',
        'selected_seeds': selected, 'runs': checks,
        'guarantee': 'Empirical finite-grid checks only; no injectivity certificate.'})
    print('Verified and plotted all 15 latest InfoNCE encoders; examples:', selected)


if __name__ == '__main__':
    main()

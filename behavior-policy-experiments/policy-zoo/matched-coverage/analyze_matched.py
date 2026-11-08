"""Concise results, paired inference, and figures for the fixed comparison."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/kempner-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matched as exp

HERE = Path(__file__).resolve().parent
COLORS = dict(zip(exp.NAMES, ['#444444', '#d95f02', '#1b9e77', '#7570b3']))


def sign_test(difference):
    difference = np.asarray(difference)
    wins = int(np.sum(difference > 1e-10))
    losses = int(np.sum(difference < -1e-10))
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    return wins, losses, p


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues))
    largest = 0.
    for rank, i in enumerate(order):
        largest = max(largest, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1., largest)
    return adjusted


def inference(rows):
    lookup = {(r['name'], r['seed']): r for r in rows}
    comparisons = []
    for name in exp.NAMES[1:]:
        seeds = sorted(s for n, s in lookup if n == name and ('reference', s) in lookup)
        for endpoint in ['log_distance_r2', 'shared_goal_error', 'native_goal_error']:
            def value(r):
                if endpoint == 'log_distance_r2':
                    return r['representation'][endpoint]
                kind = endpoint.split('_')[0]
                return -r['planning'][kind]['goal_error']
            delta = np.array([value(lookup[name, s]) - value(lookup['reference', s]) for s in seeds])
            if not len(delta):
                continue
            wins, losses, p = sign_test(delta)
            generator = np.random.default_rng(20261105)
            bootstrap = delta[generator.integers(0, len(delta), (20000, len(delta)))].mean(1)
            comparisons.append({'name': name, 'endpoint': endpoint, 'n': len(seeds),
                                'mean_gain': float(delta.mean()), 'wins': wins, 'losses': losses,
                                'p': p, 'ci95': np.quantile(bootstrap, [.025, .975]).tolist()})
    for family in ['primary', 'secondary']:
        selected = [r for r in comparisons if (r['endpoint'] == 'native_goal_error') == (family == 'secondary')]
        for row, adjusted in zip(selected, holm([r['p'] for r in selected])):
            row['family'], row['holm_p'] = family, float(adjusted)
    return comparisons


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / (name + '.png'), dpi=170)
    fig.savefig(output / (name + '.pdf'))
    plt.close(fig)


def plots(design, rows, output):
    d, pi = design['distance'], design['pi']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(d, pi, color='black')
    axes[0, 0].set(title='Same distance probabilities for every policy', ylabel='Stationary probability')
    for name in exp.NAMES:
        kernel = design['kernels'][name]
        eigenvalues, features = exp.slow_features(kernel, pi, d)
        axes[0, 1].plot((d[1:] + d[:-1])/2, np.diag(design['flows'][name], 1) / design['size'],
                        color=COLORS[name], label=exp.LABELS[name])
        axes[1, 0].plot(d, features[:, 0], color=COLORS[name], label=exp.LABELS[name])
        axes[1, 1].plot([1, 2, 3], 1 - eigenvalues, 'o-', color=COLORS[name])
    axes[0, 1].set(title='Different probabilities of crossing each edge', ylabel='Stationary one-way crossing probability')
    axes[1, 0].set(title='Slowest varying feature (unit variance)', ylabel='Feature value')
    axes[1, 1].set(title='How quickly the first three features change', xlabel='Feature number', ylabel='1 − eigenvalue (smaller = slower)', xticks=[1, 2, 3])
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xscale('log'); ax.set_xlabel('Physical distance'); ax.set_xticks([1, 2, 4, 8, 16], labels=['1','2','4','8','16'])
    axes[0, 1].legend(fontsize=8)
    save(fig, output, 'coverage_and_slow_features')
    spectral = {}
    for name in exp.NAMES:
        eigenvalues, features = exp.slow_features(design['kernels'][name], pi, d)
        spectral[name + '_eigenvalues'] = eigenvalues
        spectral[name + '_features'] = features
    np.savez_compressed(output / 'slow_features.npz', distance=d, pi=pi, **spectral)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, ax in enumerate(axes):
        for name in exp.NAMES:
            ax.plot(d, spectral[name + '_features'][:, index], color=COLORS[name], label=exp.LABELS[name])
        ax.set(title=f'Slow feature {index + 1}', xlabel='Physical distance', xscale='log')
        ax.set_xticks([1, 2, 4, 8, 16], labels=['1','2','4','8','16']); ax.grid(alpha=.2)
    axes[0].set_ylabel('Feature value (unit variance)')
    axes[-1].legend(fontsize=7)
    save(fig, output, 'first_three_slow_features')
    if not rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, name in zip(axes, exp.NAMES):
        for row in rows:
            if row['name'] != name:
                continue
            path = output / f"{name}_seed{row['seed']}.npz"
            curve = np.load(path)['representation']
            ax.plot(d, curve, color=COLORS[name], alpha=.5, linewidth=1)
        _, features = exp.slow_features(design['kernels'][name], pi, d)
        ax.plot(d, features[:, 0], '--', color='black', label='Slowest feature')
        logd = np.log(d); logd = (logd - pi @ logd)/np.sqrt(pi @ (logd - pi @ logd)**2)
        ax.plot(d, logd, ':', color='gray', label='Affine log-distance')
        ax.set(title=exp.LABELS[name], xlabel='Physical distance', xscale='log')
        ax.set_xticks([1, 2, 4, 8, 16], labels=['1','2','4','8','16']); ax.grid(alpha=.2)
    axes[0].set_ylabel('Normalized encoder output')
    axes[-1].legend(fontsize=8)
    save(fig, output, 'learned_representations')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, name in enumerate(exp.NAMES):
        group = sorted([r for r in rows if r['name'] == name], key=lambda r:r['seed'])
        if not group:
            continue
        for ax, values in [(axes[0], [r['representation']['log_distance_r2'] for r in group]),
                           (axes[1], [r['planning']['shared']['goal_error'] for r in group]),
                           (axes[2], [r['planning']['native']['goal_error'] for r in group])]:
            jitter = np.linspace(-.12, .12, len(values))
            ax.scatter(index + jitter, values, color=COLORS[name], alpha=.65, s=22)
            ax.plot([index-.2, index+.2], [np.mean(values)]*2, color=COLORS[name], linewidth=3)
    labels = ['Reference', 'Near slow', 'Far slow', 'ICM-shaped']
    for ax in axes:
        ax.set_xticks(range(4), labels, rotation=20); ax.grid(axis='y', alpha=.2)
    axes[0].set(title='Distance recovery', ylabel='Affine log-distance R² ↑')
    axes[1].set(title='Shared predictor data', ylabel='Actual goal error ↓')
    axes[2].set(title='Policy-native predictor data', ylabel='Actual goal error ↓')
    save(fig, output, 'identifiability_and_planning')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in exp.NAMES:
        group = [r for r in rows if r['name'] == name]
        axes[0].scatter([r['representation']['log_distance_r2'] for r in group],
                        [r['planning']['shared']['goal_error'] for r in group], color=COLORS[name], label=exp.LABELS[name])
        for kind, offset in [('shared', -.12), ('native', .12)]:
            if group:
                means = [np.mean([r['planning'][kind][region+'_error'] for r in group]) for region in ['near', 'middle', 'far']]
                axes[1].plot(np.arange(3) + offset, means, 'o-' if kind == 'shared' else 's--', color=COLORS[name], alpha=.8)
    axes[0].set(xlabel='Affine log-distance R²', ylabel='Shared-data actual goal error', title='Does recovery predict reaching?')
    axes[0].legend(fontsize=7)
    axes[1].set(xticks=[0,1,2], xticklabels=['Start d < 4', '4 ≤ d < 9', 'd ≥ 9'], ylabel='Actual goal error', title='Solid: shared data; dashed: native data')
    save(fig, output, 'planning_by_region')
    lookup = {(r['name'], r['seed']): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, name in enumerate(exp.NAMES[1:]):
        seeds = sorted(s for n, s in lookup if n == name and ('reference', s) in lookup)
        functions = [lambda r:r['representation']['log_distance_r2'],
                     lambda r:-r['planning']['shared']['goal_error'],
                     lambda r:-r['planning']['native']['goal_error']]
        for ax, function in zip(axes, functions):
            delta = [function(lookup[name, seed])-function(lookup['reference', seed]) for seed in seeds]
            ax.scatter(index + np.linspace(-.12, .12, len(delta)), delta, color=COLORS[name], alpha=.7)
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks([0,1,2], ['Near slow', 'Far slow', 'ICM-shaped'], rotation=20)
        ax.grid(axis='y', alpha=.2)
    axes[0].set(title='Paired recovery gains', ylabel='R² minus reference')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[1].set(title='Shared predictor data', ylabel='Reference goal error minus policy error')
    axes[2].set(title='Native predictor data', ylabel='Reference goal error minus policy error')
    save(fig, output, 'paired_differences')


def report(design, rows, output):
    comparisons = inference(rows)
    (output / 'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    lines = ['# Same coverage, different movement: results', '',
             '**Question:** With the same visited-distance distribution, do movement rules change distance recovery and planning?', '',
             f"Completed {len(rows)} / 40 encoder fits, each with shared and native predictor evaluations. See [fixed protocol](../PROTOCOL.md).", '',
             'Both training endpoints have exactly the same 65,536-state counts across policies. The stationary law, global action counts and mean physical movement also match. This isolates transition rules at a fixed marginal, including their effects on training noise. SIGReg is unchanged.', '',
             '**First result:** all four policies recover log-distance similarly (mean R² ≈ 0.99484), with no detected folds. Shared-data planning errors are nearly identical. Native-data errors vary more, but no comparison is significant after the specified correction. This does not establish equivalence or explain the original ICM–OU gap.', '',
             '| Policy | Seeds | Log-distance R² ↑ | Dense-grid folds | Shared-data goal error ↓ | Native-data goal error ↓ | Shared success ↑ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in exp.NAMES:
        group = [r for r in rows if r['name'] == name]
        if group:
            mean = lambda fn: np.mean([fn(r) for r in group])
            lines += [f"| {exp.LABELS[name]} | {len(group)} | {mean(lambda r:r['representation']['log_distance_r2']):.4f} | {sum(r['representation']['folded'] for r in group)}/{len(group)} | {mean(lambda r:r['planning']['shared']['goal_error']):.4f} | {mean(lambda r:r['planning']['native']['goal_error']):.4f} | {mean(lambda r:r['planning']['shared']['success']):.3f} |"]
    lines += ['', 'Goal error is actual terminal physical-distance error after 12 actions, not latent prediction error. Success means within 0.25. Every test goal has zero oracle error. Near/middle/far results use starting distance; trajectories can cross regions.', '',
              '![Identifiability and planning](identifiability_and_planning.png)', '',
              '## Comparisons with the reference', '',
              'Positive gain means higher R² or lower goal error. Tests use seed-wise signs, not goal cases. Bootstrap mean-difference intervals are unadjusted. Primary Holm family: six tests (three policies × recovery/shared planning). Native planning: separate secondary family.', '',
              '| Policy | Endpoint | Mean gain [95% interval] | Wins / seeds | Holm p |',
              '| --- | --- | ---: | ---: | ---: |']
    for r in comparisons:
        lines += [f"| {exp.LABELS[r['name']]} | {r['endpoint']} | {r['mean_gain']:+.3g} [{r['ci95'][0]:+.3g}, {r['ci95'][1]:+.3g}] | {r['wins']}/{r['n']} | {r['holm_p']:.4f} |"]
    lines += ['', '![Paired differences, showing the small recovery scale](paired_differences.png)', '',
              '## What the slow features mean', '',
              'A slow feature changes little under a particular policy while varying across distances. We calculate it directly from that policy’s exact transition matrix; no encoder training is needed. The dashed feature is the best alignment coordinate under centering/unit-variance constraints. Because our unchanged training also uses SIGReg, it is a mechanistic benchmark, not an exact prediction of the trained network.', '',
              '![Same coverage and different slow features](coverage_and_slow_features.png)', '',
              '![First three slow features](first_three_slow_features.png)', '',
              '![All learned curves](learned_representations.png)', '',
              '![Planning by starting region](planning_by_region.png)', '',
              '## Limits', '',
              '- These four stationary, memoryless local policies are controlled follow-ups, not the original ICM/OU comparison. The ICM-shaped rule uses one frozen old actor and corrects its occupancy; it does not retain original ICM dynamics.',
              '- A nearest-neighbor reversible chain has a monotone first nonconstant feature. This experiment can reveal stretching/compression and planning effects; it does not imply a spectral explanation for original ICM folds.',
              '- Ten seeds quantify training variation conditional on fixed kernels, one actor and this finite lattice. They do not test variation across learned behavior policies or all continuous-world starts.',
              '- Stationary one-step pair datasets remove trajectory collection/mixing efficiency as a confound. They do not claim equally rapid coverage in actual rollouts.',
              '- Good affine recovery and low goal error are distinct outcomes. Shared predictor data better isolates the learned representation; native data additionally tests policy-specific state/action support.',
              '- Dense interpolation between observed lattice states is checked for folds, but the principal recovery and reaching scores are on reachable lattice states.',
              '- Same marginal plus unchanged SIGReg fixes its population marginal objective. Finite paired batches can still change its stochastic estimates; this is not a regularizer ablation.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted(args.output.glob('*_seed*.json'))]
    design = exp.make_design()
    plots(design, rows, args.output)
    report(design, rows, args.output)
    print(f'Wrote figures and report for {len(rows)} fits.')


if __name__ == '__main__':
    main()

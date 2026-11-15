"""Recompute the saved experiment and challenge hard/soft ordering claims.

Run from any directory. Uses existing observations/checkpoints; no retraining of
the 100 historical fits. The adversarial Adam run below is new training.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = (HERE.parent / 'hard-order-guarantees').resolve()
sys.path.insert(0, str(STUDY))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from guaranteed_models import ProjectedSpline, MARGIN, pava, old, OUT
spec = importlib.util.spec_from_file_location('no_fold_source_report', STUDY / 'report.py')
source_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_report)
check_models = source_report.check_models


def main():
    torch.set_num_threads(1)
    dest = HERE / 'output'
    dest.mkdir(exist_ok=True)
    rows, summaries, checks, _ = check_models()
    assert len(checks) == 100
    # Independent pairwise inequalities, including distances not in the data.
    rng = np.random.default_rng(20261115)
    least_slack = float('inf')
    hashes = {}
    for path in sorted((OUT / 'guaranteed').glob('*.pt')):
        saved = old.trials.load(path)
        knots = saved['model']['knots'].numpy()
        if saved['method'] == 'projected':
            z = saved['model']['z'].numpy()
        else:
            raw = saved['model']['raw'].numpy()
            z = float(saved['model']['offset']) + np.r_[0., np.cumsum(
                (MARGIN + np.logaddexp(0, raw)) * np.diff(knots))]
        ss = np.sort(rng.uniform(knots[0], knots[-1], (1000, 2)), axis=1)
        gap = np.interp(ss[:, 1], knots, z) - np.interp(ss[:, 0], knots, z)
        slack = gap - MARGIN * np.diff(ss, axis=1).ravel()
        assert slack.min() >= -1e-12
        least_slack = min(least_slack, float(slack.min()))
        hashes[str(path.relative_to(STUDY))] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Deliberately train toward an alternating, folded knot target. Every Adam
    # proposal is projected; verify the true segment slopes after every update.
    model = ProjectedSpline()
    optimizer = torch.optim.Adam(model.parameters(), lr=.2)
    target = torch.tensor(3 * (-1.) ** np.arange(65), dtype=torch.float64)
    before_fail = after_fail = 0
    worst = None
    lowest_after = float('inf')
    for step in range(500):
        optimizer.zero_grad()
        loss = (model.z - target).square().mean()
        loss.backward()
        optimizer.step()
        proposal = model.z.detach().clone().numpy()
        before = float((np.diff(proposal) / np.diff(model.knots.numpy())).min())
        before_fail += before < 0
        model.project()
        after = model.certificate()['min_slope']
        lowest_after = min(lowest_after, after)
        after_fail += after < MARGIN - 1e-10
        assert after >= MARGIN - 1e-10
        if worst is None or before < worst[0]:
            worst = (before, step + 1, proposal.copy(), model.z.detach().numpy().copy())
    assert before_fail > 0 and after_fail == 0

    # Verify Euclidean projection using primal/dual KKT conditions, independently.
    for _ in range(1000):
        x = rng.normal(size=65)
        v = pava(x)
        dual = np.cumsum(x-v)[:-1]
        assert np.min(np.diff(v)) >= -1e-12
        assert np.min(dual) >= -1e-10
        assert abs(np.sum(v-x)) < 1e-10
        assert np.max(np.abs(dual*np.diff(v))) < 1e-10

    # Every sampled value and derivative looks ordered; between probes it folds.
    grid = np.linspace(0, 1, 65)
    dense = np.linspace(0, 1, 16385)
    amp = .01
    curve = dense + amp*np.sin(128*np.pi*dense)
    derivative = 1 + amp*128*np.pi*np.cos(128*np.pi*dense)
    assert np.min(np.diff(grid + amp*np.sin(128*np.pi*grid))) > 0
    assert np.min(1 + amp*128*np.pi*np.cos(128*np.pi*grid)) > 0
    assert derivative.min() < 0

    # A finite squared hinge penalty has a folded global optimum for this task.
    lam, c, margin = 100., 20., .05
    optimum = margin - c/(2*lam)
    assert optimum < 0
    loss = lambda z: c*z + lam*np.maximum(0, margin-z)**2
    assert loss(optimum) < loss(0.) and loss(optimum) < loss(margin)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    knots = model.knots.numpy()
    axes[0].plot(knots, worst[2], color='#b94348', label='Adam proposal')
    axes[0].plot(knots, worst[3], color='#147b79', lw=2, label='After projection')
    axes[0].set(xlabel='Apparent height s', ylabel='Code H(s)',
                title=f'Adversarial training: update {worst[1]}')
    axes[0].legend(frameon=False)
    axes[1].plot(dense, curve, color='#345e91', lw=1)
    axes[1].scatter(grid, grid, s=6, color='#b94348', zorder=3)
    axes[1].set(xlim=(.2, .3), ylim=(.19, .31), xlabel='Ordered coordinate s',
                ylabel='Code H(s)', title='Ordered probes can miss folds')
    fig.tight_layout()
    fig.savefig(dest/'training-stress.png', dpi=180)
    fig.savefig(dest/'training-stress.pdf')
    plt.close(fig)
    result = dict(
        historical_fits_rechecked=len(checks), summaries=summaries,
        random_ordered_pairs_checked=100000, minimum_pairwise_slack=least_slack,
        adversarial_training=dict(updates=500, negative_slope_proposals=before_fail,
            post_projection_margin_violations=after_fail, minimum_post_projection_slope=lowest_after,
            example_update=worst[1], example_proposal_minimum_slope=worst[0]),
        projection_KKT_cases=1000,
        finite_probe_counterexample=dict(minimum_true_derivative=float(derivative.min()),
            minimum_probe_derivative=float((1+amp*128*np.pi*np.cos(128*np.pi*grid)).min())),
        soft_penalty_counterexample=dict(weight=lam, task_coefficient=c, margin=margin,
            optimal_slope=optimum), checkpoint_sha256=hashes)
    (dest/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['% Generated by verify.py; fold counts are descriptive historical results.',
        r'\begin{tabular}{llrrr}', r'\toprule',
        r'Occupancy & Method & Folds / fits & Objective & $R^2$ \\', r'\midrule']
    for r in summaries:
        level = 'Native' if r['level'] is None else f"{100*r['level']:g}\\%"
        method = dict(free='Free MLP', positive='Positive slopes', projected='Projection')[r['method']]
        lines.append(f"{level} & {method} & {r['folded']} / {r['n']} & {r['batch_objective_mean']:.3f} & {r['mean_r2']:.3f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (dest/'results-table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoint_sha256', 'summaries')}, indent=2))


if __name__ == '__main__':
    main()

"""Secondary frozen-initial-encoder planning baseline; no primary-test changes."""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

import run_confirmation as run
import analyze_confirmation as analysis

OUT = run.OUT / 'untrained_baseline'


def fit(seed):
    torch.set_num_threads(1)
    run.check_lock()
    path = OUT / f'seed{seed}.json'
    if path.exists():
        assert json.loads(path.read_text())['signature'] == run.signature()
        return
    model, _ = run.original.reliability.make_plan(run.original.config(), seed, seed, 'icm')
    model.eval()
    metrics, curve = run.original.geometry(model)
    common = run.previous.prior.matched.shared_predictor_data(run.previous.prior.matched.make_design(size=16384), seed)
    checkpoint = OUT / f'predictor_seed{seed}.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['signature'] == run.signature()
        predictor = run.previous.prior.matched.planning.Predictor()
        predictor.load_state_dict(saved['predictor'])
        mean, scale = saved['mean'], saved['scale']
    else:
        predictor, mean, scale = run.previous.prior.matched.planning.train_predictor(model, common, common, seed, 2000)
        run.save(checkpoint, {'signature': run.signature(), 'predictor': predictor.state_dict(), 'mean': mean, 'scale': scale})
    planning, paths = run.previous.prior.matched.planning_evaluation(model, predictor, mean, scale, seed)
    np.savez_compressed(OUT / f'seed{seed}.npz', **curve, **paths)
    run.save_json(path, {'signature': run.signature(), 'seed': seed, 'encoder_updates': 0,
                         'predictor_updates': 2000, 'metrics': metrics, 'planning': planning})
    print(json.dumps({'baseline_completed_seed': seed}), flush=True)


def verify(seed):
    torch.set_num_threads(1)
    row = json.loads((OUT / f'seed{seed}.json').read_text())
    assert row['signature'] == run.signature() and row['encoder_updates'] == 0
    model, _ = run.original.reliability.make_plan(run.original.config(), seed, seed, 'icm')
    metrics, curve = run.original.geometry(model)
    assert metrics == row['metrics']
    saved = np.load(OUT / f'seed{seed}.npz')
    for key, value in curve.items():
        np.testing.assert_array_equal(value, saved[key])
    checkpoint = torch.load(OUT / f'predictor_seed{seed}.pt', map_location='cpu')
    predictor = run.previous.prior.matched.planning.Predictor()
    predictor.load_state_dict(checkpoint['predictor'])
    planning, paths = run.previous.prior.matched.planning_evaluation(model, predictor, checkpoint['mean'], checkpoint['scale'], seed)
    assert planning == row['planning']
    for key, value in paths.items():
        np.testing.assert_array_equal(value, saved[key])
    for policy in run.POLICIES:
        condition = json.loads((run.OUT / f'{policy}_matched_seed{seed}.json').read_text())
        assert metrics == condition['initial_metrics']
    return True


def report():
    rows, missing = analysis.collect()
    assert not missing, 'No partial-cohort comparison.'
    baseline = [json.loads((OUT / f'seed{seed}.json').read_text()) for seed in run.SEEDS]
    assert all(r['signature'] == run.signature() for r in baseline)
    rng = np.random.default_rng(20261107)
    draws = rng.integers(0, 60, (20000, 60))
    effects = {}
    for policy in run.POLICIES:
        recovery = np.array([rows[policy, 'matched', seed]['metrics']['affine_log_r2'] - b['metrics']['affine_log_r2'] for seed, b in zip(run.SEEDS, baseline)])
        reaching = np.array([b['planning']['goal_error'] - rows[policy, 'matched', seed]['planning']['shared']['goal_error'] for seed, b in zip(run.SEEDS, baseline)])
        effects[policy] = {}
        for name, delta in [('recovery_gain', recovery), ('goal_error_reduction', reaching)]:
            effects[policy][name] = {'mean': float(delta.mean()), 'bootstrap_ci95_unadjusted': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
    result = {'signature': run.signature(), 'secondary_post_start_addendum': True,
              'seeds': list(run.SEEDS), 'encoder_updates': 0, 'predictors': 60,
              'mean_initial_recovery': float(np.mean([r['metrics']['affine_log_r2'] for r in baseline])),
              'initial_fold_count': sum(r['metrics']['folded'] for r in baseline),
              'mean_initial_encoder_goal_error': float(np.mean([r['planning']['goal_error'] for r in baseline])),
              'trained_vs_initial_effects': effects, 'confirmatory_p_values_added': False}
    run.save_json(OUT / 'SUMMARY.json', result)
    lines = ['# Does training help beyond the starting geometry?', '',
             '**Secondary analysis added after training began; no additional confirmatory claims.** The exact starting encoder is frozen, while its predictor gets the same data/budget/goals as the trained-encoder shared-data arm.', '',
             f"Untrained encoder: mean recovery **{result['mean_initial_recovery']:.4f}**, **{result['initial_fold_count']}/60** folds, mean physical goal error **{result['mean_initial_encoder_goal_error']:.4f}**.", '',
             '| Trained policy | Recovery gain over initialization | Goal-error reduction over initialization |',
             '| --- | ---: | ---: |']
    for policy, effect in effects.items():
        cells = []
        for key in ('recovery_gain', 'goal_error_reduction'):
            value = effect[key]
            lo, hi = value['bootstrap_ci95_unadjusted']
            cells.append(f"{value['mean']:.5f} [{lo:.5f}, {hi:.5f}]")
        lines.append(f"| {policy} | {cells[0]} | {cells[1]} |")
    lines += ['', 'Brackets are unadjusted descriptive bootstrap 95% intervals. Positive changes favor training. This comparison distinguishes acquiring useful geometry from preserving a useful initial map; it does not change the primary equivalence conclusion.', '',
              '[Addendum](../../BASELINE_ADDENDUM.md) · [Exact results](SUMMARY.json) · [Verification](VERIFICATION.json)', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--wait', action='store_true', help='Wait for the main verified pipeline before comparison.')
    args = parser.parse_args()
    run.check_lock()
    if args.train:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            list(pool.imap_unordered(fit, run.SEEDS))
    if args.wait:
        completed = run.OUT / 'PIPELINE_COMPLETE.json'
        while not completed.exists():
            time.sleep(30)
        assert json.loads(completed.read_text())['signature'] == run.signature()
    if args.analyze or args.verify:
        _, missing = analysis.collect()
        assert not missing
    if args.analyze:
        report()
    if args.verify:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            checked = list(pool.imap_unordered(verify, run.SEEDS))
        assert len(checked) == 60 and all(checked)
        run.save_json(OUT / 'VERIFICATION.json', {'signature': run.signature(), 'complete': True,
                      'initial_geometries_replayed': 60, 'physical_planning_replays': 60,
                      'initial_geometries_matched_to_trained_cells': 420, 'errors': 0})
        print('All 60 baseline geometries and physical rollouts verified.')


if __name__ == '__main__':
    main()

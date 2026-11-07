"""Recreate all sources, recovery curves and physical rollouts before publication."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import stats
import torch

import run_confirmation as run
import analyze_confirmation as analysis


def same_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=0)


def source_check(task):
    policy, seed = task
    torch.set_num_threads(1)
    c = run.original.config()
    source = torch.load(run.OUT / 'sources' / f'{policy}_seed{seed}.pt', map_location='cpu')
    manifest = json.loads((run.OUT / 'source_manifests' / f'{policy}_seed{seed}.json').read_text())
    assert source['signature'] == manifest['signature'] == run.signature()
    actor = None
    actor_hash = None
    if policy in run.LEARNED:
        actor_file = torch.load(run.OUT / 'actors' / f'{policy}_seed{seed}.pt', map_location='cpu')
        assert actor_file['signature'] == run.signature()
        actor = run.original.learned.load(actor_file['actor'])
        actor_hash = run.previous.prior.tensor_hash(actor.state_dict())
        assert actor_hash == manifest['actor_hash']
        native = run.original.learned.collect(c, policy, c.train_episodes, 10000 + seed, actor)
    elif policy == 'random':
        native = run.behavior.collect(c, 'random', c.train_episodes, 10000 + seed)
    else:
        native, _ = run.original.run_zoo.source(c, policy, seed, run.OUT)
    for key in native:
        torch.testing.assert_close(native[key], source['native'][key], atol=0, rtol=0)
    distance, pool = run.previous.reset_pool(seed)
    torch.testing.assert_close(pool, source['pool'], atol=0, rtol=0)
    import hashlib
    assert hashlib.sha256(pool.numpy().tobytes()).hexdigest() == manifest['common_pool_hash']
    torch.testing.assert_close(pool, source['reset']['x'], atol=0, rtol=0)
    torch.manual_seed(5300000 + seed)
    if policy in ('ou_actions_lp_w0.1', 'ou_actions_hp_w0.3'):
        action = native['action']
    elif actor is not None:
        with torch.no_grad():
            action = torch.cat([actor(pool[t*c.train_episodes:(t+1)*c.train_episodes])[0].sample() for t in range(c.horizon)])
    else:
        action = torch.cat([torch.randint(0, 3, (c.train_episodes,)) for _ in range(c.horizon)])
    torch.testing.assert_close(action, source['reset']['action'], atol=0, rtol=0)
    successor = run.behavior.move(distance, action, c)
    torch.testing.assert_close(run.behavior.sensor(successor, c), source['reset']['xp'], atol=0, rtol=0)
    for kind in ('native', 'reset'):
        assert run.previous.source_hash(source[kind]) == manifest[kind + '_hash']
    if actor is not None:
        assert actor_hash == run.previous.prior.tensor_hash(actor.state_dict())
    assert manifest['actor_unchanged']
    return {'policy': policy, 'seed': seed, 'pool_hash': manifest['common_pool_hash'], 'actor_hash': actor_hash}


def result_check(task):
    policy, cell, seed = task
    torch.set_num_threads(1)
    name = run.cell_name(policy, cell)
    row = json.loads((run.OUT / f'{name}_seed{seed}.json').read_text())
    assert row['signature'] == run.signature()
    source = torch.load(run.OUT / 'sources' / f'{policy}_seed{seed}.pt', map_location='cpu')
    data = source['reset' if cell == 'matched' else 'native']
    pool = run.previous.pooled(source['native']) if cell == 'native_exposure' else source['pool']
    import hashlib
    assert row['source_hash'] == run.previous.source_hash(data)
    assert row['regularizer_pool_hash'] == hashlib.sha256(pool.numpy().tobytes()).hexdigest()
    initial, _ = run.original.reliability.make_plan(run.original.config(), seed, seed, 'icm')
    initial_metrics, _ = run.original.geometry(initial)
    same_metrics(initial_metrics, row['initial_metrics'])
    state = torch.load(run.OUT / f'encoder_{name}_seed{seed}.pt', map_location='cpu')
    assert state['signature'] == run.signature() and state['step'] == 4000
    assert all(int(s['step']) == 4000 for s in state['optimizer']['state'].values())
    model = run.behavior.Encoder()
    model.load_state_dict(state['encoder'])
    model.eval()
    geometry, curves = run.original.geometry(model)
    same_metrics(geometry, row['metrics'])
    saved = np.load(run.OUT / f'{name}_seed{seed}.npz')
    for key, value in curves.items():
        np.testing.assert_array_equal(value, saved[key])
    for kind in ('shared', 'native'):
        predictor_state = torch.load(run.OUT / f'predictor_{name}_{kind}_seed{seed}.pt', map_location='cpu')
        assert predictor_state['signature'] == run.signature()
        predictor = run.previous.prior.matched.planning.Predictor()
        predictor.load_state_dict(predictor_state['predictor'])
        metrics, paths = run.previous.prior.matched.planning_evaluation(model, predictor, predictor_state['mean'], predictor_state['scale'], seed)
        same_metrics(metrics, row['planning'][kind])
        assert metrics['oracle_error'] == 0
        for key, value in paths.items():
            np.testing.assert_array_equal(value, saved[kind + '_' + key])
    return 2


def inference_check(rows):
    inference = json.loads((run.OUT / 'INFERENCE.json').read_text())
    assert inference['signature'] == run.signature() and inference['seeds'] == list(run.SEEDS)
    pair_checks = 0
    for key, result in inference['primary_endpoints'].items():
        matrix = np.array([[analysis.endpoint(rows[p, 'matched', seed], key) for p in run.POLICIES] for seed in run.SEEDS])
        checked = []
        for pair in result['pairs']:
            i, j = run.POLICIES.index(pair['policy_a']), run.POLICIES.index(pair['policy_b'])
            d = matrix[:, i] - matrix[:, j]
            if np.std(d, ddof=1) > 0:
                p = max(stats.ttest_1samp(d, -result['margin'], alternative='greater').pvalue,
                        stats.ttest_1samp(d, result['margin'], alternative='less').pvalue)
            else:
                p = 0. if abs(d[0]) < result['margin'] else 1.
            assert np.isclose(p, pair['mean_tost_p'], atol=1e-14, rtol=1e-12)
            checked.append(p)
            pair_checks += 1
        assert np.isclose(max(checked), result['mean_global_p'], atol=1e-14, rtol=1e-12)
    assert pair_checks == 84
    assert inference['primary_joint_p'] == max(r['mean_global_p'] for r in inference['primary_endpoints'].values())
    for key, saved in inference['mechanism'].items():
        d = []
        for seed in run.SEEDS:
            a, b = rows['icm', 'native_exposure', seed], rows['icm', 'common_exposure', seed]
            delta = int(a['metrics']['folded']) - int(b['metrics']['folded']) if key == 'folded' else analysis.endpoint(b, key) - analysis.endpoint(a, key)
            if key == 'shared_goal_error':
                delta *= -1
            d.append(delta)
        np.testing.assert_array_equal(d, saved['seed_benefits'])
        wins, losses = int(np.sum(np.array(d) > 1e-10)), int(np.sum(np.array(d) < -1e-10))
        p = stats.binomtest(wins, wins + losses, .5, alternative='greater').pvalue if wins + losses else 1.
        assert abs(p - saved['p']) < 1e-14
    family = inference['confirmatory_holm_family']
    order = np.argsort([r['p'] for r in family])
    previous = 0.
    for rank, index in enumerate(order):
        previous = min(1., max(previous, (4 - rank) * family[index]['p']))
        assert abs(previous - family[index]['holm_p']) < 1e-14
    return pair_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run.check_lock()
    rows, missing = analysis.collect()
    if missing:
        raise SystemExit(f'Incomplete cohort: {len(missing)} missing. Verification not substituted with a partial cohort.')
    source_tasks = [(p, s) for s in run.SEEDS for p in run.POLICIES]
    result_tasks = [(p, cell, s) for p, cell, s in rows]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        sources = list(pool.imap_unordered(source_check, source_tasks))
        planning = list(pool.imap_unordered(result_check, result_tasks))
    for seed in run.SEEDS:
        assert len({r['pool_hash'] for r in sources if r['seed'] == seed}) == 1
        assert rows['icm', 'native_exposure', seed]['source_hash'] == rows['icm', 'common_exposure', seed]['source_hash']
        assert rows['icm', 'common_exposure', seed]['regularizer_pool_hash'] == rows['icm', 'matched', seed]['regularizer_pool_hash']
    for policy in run.LEARNED:
        assert len({r['actor_hash'] for r in sources if r['policy'] == policy}) == 60
    assert len(rows) == 540 and len(sources) == 420 and sum(planning) == 1080
    inference_checks = inference_check(rows)
    gates = json.loads((run.OUT / 'GATES.json').read_text())
    assert gates['signature'] == run.signature() and gates['matched_encoder_and_planning_bit_exact']
    replay = json.loads((run.OUT / 'FRESH_REPLAY_GATES.json').read_text())
    assert replay['signature'] == run.signature() and all(r['exact'] for r in replay['checks'])
    result = {'signature': run.signature(), 'complete': True, 'fresh_unique_actors': 240,
              'sources_recreated': len(sources), 'encoders_geometry_replayed': len(rows),
              'physical_planning_replays': sum(planning), 'primary_tost_boundary_checks': inference_checks,
              'holm_claims_verified': 4, 'shared_pool_and_icm_pair_controls_verified': True,
              'full_and_resumed_training_gates': True, 'errors': 0}
    run.save_json(run.OUT / 'VERIFICATION.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

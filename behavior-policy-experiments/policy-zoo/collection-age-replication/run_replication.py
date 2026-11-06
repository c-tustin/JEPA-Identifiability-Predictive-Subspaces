"""Fresh frozen actors, paired collection windows, encoders and reaching tests.

Run in the background or foreground. Completed work and encoder/Adam checkpoints
are reused on restart. Training an interrupted behavior actor restarts that actor.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import common as study


def old_actor_gates():
    """Ensure current actor training matches the historical zoo implementation."""
    path = study.OUT / 'ACTOR_REPLAY_GATES.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved['signature'] == study.signature()
        return
    torch.set_num_threads(1)
    results = []
    for policy in ('icm', 'count'):
        actor, _ = study.learned.train(study.original.config(), 13, policy)
        old = torch.load(study.HERE.parent / 'output/policies' / f'{policy}_seed13.pt', map_location='cpu')
        error = max(float((value - old['policy'][key]).abs().max()) for key, value in actor.state_dict().items())
        if error != 0:
            raise RuntimeError(f'Historical actor gate failed: {policy}, error {error}')
        results.append({'policy': policy, 'seed': 13, 'maximum_parameter_error': error})
        print(json.dumps({'stage': 'actor_gate', **results[-1]}), flush=True)
    study.save_json(path, {'signature': study.signature(), 'results': results})


def prepare_source(task):
    policy, seed = task
    torch.set_num_threads(1)
    c = study.original.config()
    path = study.OUT / 'sources' / f'{policy}_seed{seed}.pt'
    if path.exists():
        assert torch.load(path, map_location='cpu')['signature'] == study.signature()
        return
    actor = None
    actor_hash = None
    if policy != 'random':
        actor_path = study.OUT / 'actors' / f'{policy}_seed{seed}.pt'
        if actor_path.exists():
            saved = torch.load(actor_path, map_location='cpu')
            assert saved['signature'] == study.signature()
            actor = study.learned.load(saved['actor'])
        else:
            actor, saved = study.learned.train(c, seed, policy)
            study.atomic_torch(actor_path, {'signature': study.signature(), 'actor': saved})
        actor_hash = study.tensor_hash(actor.state_dict())
    collection = copy.deepcopy(c)
    collection.burnin, collection.horizon = 0, 192
    kind = 'random' if policy == 'random' else 'icm'
    full = study.behavior.collect(collection, kind, c.train_episodes, 10000 + seed, actor)
    # A full rollout and separate burn-in collection must agree on the late data.
    direct = study.behavior.collect(c, kind, c.train_episodes, 10000 + seed, actor)
    late_hash = study.data_hash(study.extract_window(full, 128))
    assert late_hash == study.data_hash(direct)
    if actor is not None:
        assert actor_hash == study.tensor_hash(actor.state_dict())
    windows = {}
    for age in study.AGES:
        data = study.extract_window(full, age)
        assert len(data['x']) == 16384
        windows[str(age)] = {'sha256': study.data_hash(data), 'exposure': study.exposure(data)}
    study.atomic_torch(path, {'signature': study.signature(), 'full': full})
    study.save_json(study.OUT / 'source_manifests' / f'{policy}_seed{seed}.json', {
        'signature': study.signature(), 'policy': policy, 'seed': seed,
        'actor_hash': actor_hash, 'actor_unchanged': True,
        'late_window_matches_direct_collection': True, 'windows': windows,
    })
    print(json.dumps({'stage': 'source', 'policy': policy, 'seed': seed}), flush=True)


def train_condition(task):
    policy, age, seed, continuation = task
    torch.set_num_threads(1)
    condition = continuation or study.label(policy, age)
    path = study.result_path(condition, seed)
    signature = study.signature()
    if path.exists():
        assert json.loads(path.read_text())['signature'] == signature
        return
    c = study.original.config()
    data = study.source(policy, age, seed)
    source_hash = study.data_hash(data)
    parent_hash = None
    if continuation:
        parent_path = study.OUT / f'encoder_icm_age000_seed{seed}.pt'
        parent = torch.load(parent_path, map_location='cpu')
        assert parent['signature'] == signature
        model = study.behavior.Encoder()
        model.load_state_dict(parent['encoder'])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.load_state_dict(parent['optimizer'])
        parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        generator = torch.Generator().manual_seed(3100000 + seed)
        indices = torch.randint(len(data['x']), (c.jepa_steps, c.batch), generator=generator)
        interval = 500
    else:
        model, plan = study.original.reliability.make_plan(c, seed, seed, 'icm')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        indices = plan['indices']
        interval = 1000
    initial_metrics, initial_curve = study.original.geometry(model)
    partial = study.OUT / 'partials' / f'{condition}_seed{seed}.pt'
    start = 0
    history = [{'step': 0, **initial_metrics}]
    if partial.exists():
        saved = torch.load(partial, map_location='cpu')
        assert saved['signature'] == signature and saved['source_hash'] == source_hash
        model.load_state_dict(saved['encoder'])
        optimizer.load_state_dict(saved['optimizer'])
        start, history = saved['step'], saved['history']
    regularizer = study.behavior.SIGReg1D()
    for step in range(start, c.jepa_steps):
        ix = indices[step]
        losses = study.controls.update(model, optimizer, regularizer,
                                       data['x'][ix], data['xp'][ix], c,
                                       study.controls.learning_rate(step, c.jepa_steps))
        if (step + 1) % interval == 0:
            metrics, _ = study.original.geometry(model)
            history.append({'step': step + 1, **losses, **metrics})
            study.atomic_torch(partial, {
                'signature': signature, 'source_hash': source_hash,
                'step': step + 1, 'history': history,
                'encoder': model.state_dict(), 'optimizer': optimizer.state_dict(),
            })
    model.eval()
    metrics, curve = study.original.geometry(model)
    encoder_hash = study.tensor_hash(model.state_dict())
    study.atomic_torch(study.OUT / f'encoder_{condition}_seed{seed}.pt', {
        'signature': signature, 'encoder': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'config': vars(c),
    })
    result = {'signature': signature, 'condition': condition, 'policy': policy,
              'age': age, 'seed': seed, 'continuation': bool(continuation),
              'parent_checkpoint_sha256': parent_hash, 'source_hash': source_hash,
              'encoder_hash': encoder_hash, 'initial_metrics': initial_metrics,
              'metrics': metrics, 'history': history, 'planning': {}}
    if policy != 'ou':
        result['exposure'] = study.exposure(data)
    common = study.matched.shared_predictor_data(study.matched.make_design(size=16384), seed)
    datasets = [('shared', common)]
    if policy != 'ou':
        datasets.append(('native', data))
    arrays = {**curve, 'initial_curve': initial_curve['curve']}
    for kind, predictor_data in datasets:
        predictor, mean, scale = study.matched.planning.train_predictor(model, predictor_data, common, seed, 2000)
        planning_metrics, paths = study.matched.planning_evaluation(model, predictor, mean, scale, seed)
        result['planning'][kind] = planning_metrics
        arrays.update({kind + '_' + key: value for key, value in paths.items()})
        study.atomic_torch(study.OUT / f'predictor_{condition}_{kind}_seed{seed}.pt', {
            'predictor': predictor.state_dict(), 'mean': mean, 'scale': scale,
            'encoder_hash': encoder_hash, 'signature': signature,
        })
    np.savez_compressed(study.OUT / f'{condition}_seed{seed}.npz', **arrays)
    study.save_json(path, result)
    partial.unlink(missing_ok=True)
    print(json.dumps({'stage': 'fit', 'condition': condition, 'seed': seed,
                      'r2': metrics['affine_log_r2'], 'folded': metrics['folded'],
                      'goal_error': result['planning']['shared']['goal_error']}), flush=True)


def pool_map(function, tasks, workers):
    with mp.get_context('spawn').Pool(workers) as pool:
        list(pool.imap_unordered(function, tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    study.OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'signature': study.signature(), 'seeds': study.SEEDS, 'ages': study.AGES,
        'policies': study.POLICIES, 'config': vars(study.original.config()),
        'python': platform.python_version(), 'torch': torch.__version__,
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    }
    manifest_path = study.OUT / 'MANIFEST.json'
    if manifest_path.exists():
        # A later commit may change HEAD without changing the protocol/code.
        assert json.loads(manifest_path.read_text())['signature'] == manifest['signature']
    else:
        study.save_json(manifest_path, manifest)
    old_actor_gates()
    pool_map(prepare_source, [(policy, seed) for seed in study.SEEDS for policy in study.POLICIES], args.workers)
    main_tasks = [(policy, age, seed, None) for seed in study.SEEDS
                  for policy in study.POLICIES for age in study.AGES]
    main_tasks += [('ou', -1, seed, None) for seed in study.SEEDS]
    pool_map(train_condition, main_tasks, args.workers)
    continuations = [("icm", age, seed, name) for seed in study.SEEDS
                     for age, name in [(0, 'icm_early_to_early'), (128, 'icm_early_to_late')]]
    pool_map(train_condition, continuations, args.workers)
    study.save_json(study.OUT / 'TRAINING_COMPLETE.json', {'signature': study.signature(),
                                                         'encoder_conditions': 360,
                                                         'predictor_conditions': 700})


if __name__ == '__main__':
    main()

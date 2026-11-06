"""Prospective fresh-actor replication; checkpoints and fixed-size inference.

Run --gate before --freeze, then --run. No inference on an incomplete cohort.
The protocol and executable dependencies are hash-locked before fresh training.
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'geometry-and-coverage'))
import common_gc as previous

OUT = HERE / 'output'
SEEDS = tuple(range(6000, 6060))
POLICIES = previous.ZOO_POLICIES
LEARNED = ('icm', 'count', 'rnd', 'disagreement')
behavior = previous.behavior
original = previous.original


def save(path, value):
    previous.atomic_torch(path, value)


def save_json(path, value):
    previous.save_json(path, value)


def signature():
    digest = hashlib.sha256()
    for name in ('PROTOCOL.md', 'run_confirmation.py', 'analyze_confirmation.py'):
        digest.update((HERE / name).read_bytes())
    digest.update(previous.signature().encode())
    for name in ('matched_equivalence.py', 'identifiability_equivalence.py'):
        digest.update((previous.HERE / name).read_bytes())
    for module in (original.run_zoo, original.run_zoo.zoo, previous.prior.matched.planning):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def check_lock():
    locked = json.loads((OUT / 'LOCK.json').read_text())
    assert locked['signature'] == signature(), 'Frozen scientific code/protocol changed.'
    assert locked['seeds'] == list(SEEDS) and locked['policies'] == list(POLICIES)
    return locked


def prepare_source(policy, seed, output=OUT):
    """Train a genuinely fresh actor, then collect native and matched-reset pairs."""
    torch.set_num_threads(1)
    path = output / 'sources' / f'{policy}_seed{seed}.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu')
        assert saved['signature'] == signature()
        return saved
    c = original.config()
    actor = None
    actor_hash = None
    if policy in LEARNED:
        actor_path = output / 'actors' / f'{policy}_seed{seed}.pt'
        if actor_path.exists():
            checkpoint = torch.load(actor_path, map_location='cpu')
            assert checkpoint['signature'] == signature()
            actor = original.learned.load(checkpoint['actor'])
        else:
            actor, saved_actor = original.learned.train(c, seed, policy)
            save(actor_path, {'signature': signature(), 'actor': saved_actor})
        actor_hash = previous.prior.tensor_hash(actor.state_dict())
        native = original.learned.collect(c, policy, c.train_episodes, 10000 + seed, actor)
    elif policy == 'random':
        native = behavior.collect(c, 'random', c.train_episodes, 10000 + seed)
    else:
        native, _ = original.run_zoo.source(c, policy, seed, output)
    distance, pool = previous.reset_pool(seed)
    torch.manual_seed(5300000 + seed)
    if policy in ('ou_actions_lp_w0.1', 'ou_actions_hp_w0.3'):
        action = native['action'].clone()
    elif actor is not None:
        actions = []
        with torch.no_grad():
            for t in range(c.horizon):
                views = pool[t * c.train_episodes:(t + 1) * c.train_episodes]
                actions.append(actor(views)[0].sample())
        action = torch.cat(actions)
    else:
        action = torch.cat([torch.randint(0, 3, (c.train_episodes,)) for _ in range(c.horizon)])
    successor = behavior.move(distance, action, c)
    reset = {'x': pool, 'xp': behavior.sensor(successor, c), 'action': action,
             'z': torch.log(distance / 4) / c.sigma, 'zp': torch.log(successor / 4) / c.sigma}
    if actor is not None:
        assert actor_hash == previous.prior.tensor_hash(actor.state_dict())
    saved = {'signature': signature(), 'native': native, 'reset': reset, 'pool': pool}
    save(path, saved)
    manifest = {'signature': signature(), 'policy': policy, 'seed': seed,
                'actor_hash': actor_hash, 'actor_unchanged': True,
                'native_hash': previous.source_hash(native), 'reset_hash': previous.source_hash(reset),
                'common_pool_hash': hashlib.sha256(pool.numpy().tobytes()).hexdigest(),
                'native_exposure': previous.prior.exposure(native),
                'reset_exposure': previous.prior.exposure(reset)}
    save_json(output / 'source_manifests' / f'{policy}_seed{seed}.json', manifest)
    return saved


def cell_name(policy, cell):
    return f'{policy}_{cell}'


def train_cell(policy, cell, seed, source, output=OUT, stop_at=None):
    """Identical optimization to the preceding independently sampled loss study."""
    torch.set_num_threads(1)
    name = cell_name(policy, cell)
    result_path = output / f'{name}_seed{seed}.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['signature'] == signature()
        return result
    c = original.config()
    data = source['reset' if cell == 'matched' else 'native']
    pool = previous.pooled(source['native']) if cell == 'native_exposure' else source['pool']
    model, plan = original.reliability.make_plan(c, seed, seed, 'icm')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    alignment_indices = plan['indices']
    rng = torch.Generator().manual_seed(3400000 + seed)
    regularizer_indices = torch.randint(len(pool), (c.jepa_steps, 2 * c.batch), generator=rng)
    regularizer = behavior.SIGReg1D()
    checkpoint_path = output / f'encoder_{name}_seed{seed}.pt'
    partial = output / 'partials' / f'{name}_seed{seed}.pt'
    initial, _ = original.geometry(model)
    history = [{'step': 0, **initial}]
    start = 0
    if checkpoint_path.exists() or partial.exists():
        state = torch.load(checkpoint_path if checkpoint_path.exists() else partial, map_location='cpu')
        assert state['signature'] == signature()
        model.load_state_dict(state['encoder'])
        optimizer.load_state_dict(state['optimizer'])
        start, history = state['step'], state['history']
    parameters = list(model.parameters())
    for step in range(start, c.jepa_steps):
        optimizer.param_groups[0]['lr'] = previous.controls.learning_rate(step, c.jepa_steps)
        index = alignment_indices[step]
        y = model(torch.cat([data['x'][index], data['xp'][index]]))
        alignment = .9 * (y[:c.batch] - y[c.batch:]).square().mean()
        reg_loss = .1 * regularizer(model(pool[regularizer_indices[step]]))
        alignment_gradient = torch.autograd.grad(alignment, parameters)
        regularizer_gradient = torch.autograd.grad(reg_loss, parameters)
        optimizer.zero_grad(set_to_none=True)
        for p, ga, gr in zip(parameters, alignment_gradient, regularizer_gradient):
            p.grad = ga + gr
        optimizer.step()
        if (step + 1) % 500 == 0:
            metrics, _ = original.geometry(model)
            history.append({'step': step + 1, 'alignment': float(alignment.detach()),
                            'regularizer': float(reg_loss.detach()), **metrics})
            save(partial, {'signature': signature(), 'step': step + 1,
                           'encoder': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history})
        if stop_at is not None and step + 1 == stop_at:
            return None
    metrics, curves = original.geometry(model)
    if not checkpoint_path.exists():
        save(checkpoint_path, {'signature': signature(), 'step': c.jepa_steps,
                              'encoder': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history})
    model.eval()
    shared = previous.prior.matched.shared_predictor_data(previous.prior.matched.make_design(size=16384), seed)
    planning = {}
    arrays = dict(curves)
    for kind, predictor_data in (('shared', shared), ('native', data)):
        predictor_path = output / f'predictor_{name}_{kind}_seed{seed}.pt'
        if predictor_path.exists():
            saved_predictor = torch.load(predictor_path, map_location='cpu')
            assert saved_predictor['signature'] == signature()
            predictor = previous.prior.matched.planning.Predictor()
            predictor.load_state_dict(saved_predictor['predictor'])
            mean, scale = saved_predictor['mean'], saved_predictor['scale']
        else:
            predictor, mean, scale = previous.prior.matched.planning.train_predictor(model, predictor_data, shared, seed, 2000)
            save(predictor_path, {'signature': signature(), 'predictor': predictor.state_dict(), 'mean': mean, 'scale': scale})
        planning[kind], paths = previous.prior.matched.planning_evaluation(model, predictor, mean, scale, seed)
        arrays.update({kind + '_' + key: value for key, value in paths.items()})
    np.savez_compressed(output / f'{name}_seed{seed}.npz', **arrays)
    result = {'signature': signature(), 'seed': seed, 'policy': policy, 'cell': cell,
              'source_hash': previous.source_hash(data),
              'regularizer_pool_hash': hashlib.sha256(pool.numpy().tobytes()).hexdigest(),
              'initial_metrics': initial, 'metrics': metrics, 'history': history, 'planning': planning}
    save_json(result_path, result)
    partial.unlink(missing_ok=True)
    return result


def job(task):
    policy, seed = task
    check_lock()
    source = prepare_source(policy, seed)
    cells = ('matched', 'native_exposure', 'common_exposure') if policy == 'icm' else ('matched',)
    for cell in cells:
        train_cell(policy, cell, seed, source)
        print(json.dumps({'event': 'completed', 'policy': policy, 'cell': cell, 'seed': seed}), flush=True)


def gate():
    """All fresh-cohort paths remain untouched by these historical checks."""
    torch.set_num_threads(1)
    output = OUT / 'gates'
    checks = []
    for policy in ('icm', 'rnd'):
        actor, _ = original.learned.train(original.config(), 13, policy)
        historical = previous.old_actor(policy, 13)
        error = max(float((value - historical.state_dict()[key]).abs().max()) for key, value in actor.state_dict().items())
        assert error == 0, (policy, error)
        checks.append({'actor': policy, 'seed': 13, 'maximum_parameter_error': error})
    source = prepare_source('icm', 13, output)
    result = train_cell('icm', 'matched', 13, source, output)
    expected = json.loads((previous.OUT / 'zoo_icm_reset_seed13.json').read_text())
    assert result['metrics'] == expected['metrics'] and result['planning'] == expected['planning']
    actual = torch.load(output / 'encoder_icm_matched_seed13.pt', map_location='cpu')
    old = torch.load(previous.OUT / 'encoder_zoo_icm_reset_seed13.pt', map_location='cpu')
    for key in actual['encoder']:
        torch.testing.assert_close(actual['encoder'][key], old['encoder'][key], atol=0, rtol=0)
    save_json(OUT / 'GATES.json', {'signature': signature(), 'actor_checks': checks,
                                 'matched_encoder_and_planning_bit_exact': True})
    print('Historical actor, encoder and planning gates passed.', flush=True)


def freeze():
    assert not any((OUT / 'actors').glob('*seed6*.pt')), 'Cohort actors already exist.'
    assert not any(OUT.glob('*seed6*.json')), 'Cohort outcomes already exist.'
    gates = json.loads((OUT / 'GATES.json').read_text())
    assert gates['signature'] == signature()
    # Audit stored filenames, before any cohort files are made.
    import re
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=HERE.parents[2], text=True).splitlines()
    earlier = set()
    for name in tracked:
        earlier.update(int(s) for s in re.findall(r'(?:seed|data|init)(\d+)', name))
    assert not set(SEEDS) & earlier
    lock = {'signature': signature(), 'frozen_utc': datetime.now(timezone.utc).isoformat(),
            'seeds': list(SEEDS), 'policies': list(POLICIES), 'encoders': 540, 'predictors': 1080,
            'margins': {'r2': .001, 'goal_error': .005, 'secondary_fold_risk': .05},
            'git_head_before_cohort': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
            'fresh_seed_filename_audit_passed': True,
            'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'numpy': np.__version__},
            'config': vars(original.config())}
    save_json(OUT / 'LOCK.json', lock)
    print(json.dumps(lock), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.gate:
        gate()
    if args.freeze:
        freeze()
    if args.run:
        check_lock()
        save_json(HERE / 'RUN_STATE.json', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'signature': signature()})
        tasks = [(p, seed) for seed in SEEDS for p in POLICIES]
        with mp.get_context('spawn').Pool(args.workers) as pool:
            list(pool.imap_unordered(job, tasks))
        save_json(OUT / 'TRAINING_COMPLETE.json', {'signature': signature(), 'encoders': 540, 'predictors': 1080})
        print('Fixed cohort complete.', flush=True)


if __name__ == '__main__':
    main()

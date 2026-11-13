"""Regular OU alignment with original, independent-OU or broad SIGReg data."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'confirmatory-matched-exposure'))
import run_confirmation as previous

OUT = HERE / 'output'
SEEDS = tuple(range(6000, 6060))
ARMS = ('original', 'ou_observations', 'broad_observations')
world = previous.previous
original = previous.original


def save(path, value):
    previous.save(path, value)


def save_json(path, value):
    previous.save_json(path, value)


def signature():
    digest = hashlib.sha256(previous.signature().encode())
    for name in ('PROTOCOL.md', 'run_ou.py', 'analyze_ou.py'):
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def check_lock():
    lock = json.loads((OUT / 'LOCK.json').read_text())
    assert lock['signature'] == signature() and lock['seeds'] == list(SEEDS)
    return lock


def source(seed):
    data = original.run_zoo.zoo.collect_pairs(original.config(), .9, seed)
    # This is the same original OU source, not a clipped physical controller.
    expected = previous.behavior.collect(original.config(), 'ou', original.config().train_episodes, 10000 + seed)
    for key in data:
        torch.testing.assert_close(data[key], expected[key], atol=0, rtol=0)
    _, broad = world.reset_pool(seed)
    return data, broad


@torch.no_grad()
def gaussian_metrics(model, seed):
    rng = torch.Generator().manual_seed(9400000 + seed)
    z = torch.randn(16384, generator=rng)
    y = model(previous.behavior.observe(z, original.config().sigma)).numpy().astype(float)
    target = z.numpy().astype(float)
    tail = np.abs(target) > 2
    return {'gaussian_r2': world.prior.matched.affine_r2(target, y),
            'orthogonal_mse': float(min(np.mean((y-target)**2), np.mean((y+target)**2))),
            'raw_mean': float(y.mean()), 'raw_std': float(y.std()),
            'tail_r2': world.prior.matched.affine_r2(target[tail], y[tail]),
            'tail_points': int(tail.sum())}


def train(arm, seed, output=OUT, stop_at=None):
    torch.set_num_threads(1)
    result_path = output / f'{arm}_seed{seed}.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['signature'] == signature()
        return result
    c = original.config()
    data, broad = source(seed)
    pool = world.pooled(data) if arm == 'ou_observations' else broad
    model, plan = original.reliability.make_plan(c, seed, seed, 'icm')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    regularizer = previous.behavior.SIGReg1D()
    rng = torch.Generator().manual_seed(3400000 + seed)
    ir = torch.randint(len(pool), (c.jepa_steps, 2*c.batch), generator=rng)
    initial, _ = original.geometry(model)
    checkpoint = output / f'encoder_{arm}_seed{seed}.pt'
    partial = output / 'partials' / f'{arm}_seed{seed}.pt'
    start, history = 0, []
    if checkpoint.exists() or partial.exists():
        state = torch.load(checkpoint if checkpoint.exists() else partial, map_location='cpu')
        assert state['signature'] == signature()
        model.load_state_dict(state['encoder'])
        optimizer.load_state_dict(state['optimizer'])
        start, history = state['step'], state['history']
    parameters = list(model.parameters())
    for step in range(start, c.jepa_steps):
        lr = world.controls.learning_rate(step, c.jepa_steps)
        ix = plan['indices'][step]
        if arm == 'original':
            losses = world.controls.update(model, optimizer, regularizer, data['x'][ix], data['xp'][ix], c, lr)
        else:
            optimizer.param_groups[0]['lr'] = lr
            y = model(torch.cat([data['x'][ix], data['xp'][ix]]))
            a = .9*(y[:c.batch]-y[c.batch:]).square().mean()
            r = .1*regularizer(model(pool[ir[step]]))
            ga = torch.autograd.grad(a, parameters)
            gr = torch.autograd.grad(r, parameters)
            optimizer.zero_grad(set_to_none=True)
            for p, x, z in zip(parameters, ga, gr):
                p.grad = x+z
            optimizer.step()
            losses = {'alignment_weighted': float(a.detach()), 'sigreg_weighted': float(r.detach())}
        if (step+1) % 500 == 0:
            history.append({'step': step+1, **losses})
            save(partial, {'signature': signature(), 'step': step+1, 'encoder': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'history': history})
        if stop_at is not None and step+1 == stop_at:
            return None
    if not checkpoint.exists():
        save(checkpoint, {'signature': signature(), 'step': c.jepa_steps, 'encoder': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'history': history})
    model.eval()
    metrics, curves = original.geometry(model)
    gaussian = gaussian_metrics(model, seed)
    common = world.prior.matched.shared_predictor_data(world.prior.matched.make_design(size=16384), seed)
    prediction_path = output / f'predictor_{arm}_seed{seed}.pt'
    if prediction_path.exists():
        saved = torch.load(prediction_path, map_location='cpu')
        assert saved['signature'] == signature()
        predictor = world.prior.matched.planning.Predictor()
        predictor.load_state_dict(saved['predictor'])
        mean, scale = saved['mean'], saved['scale']
    else:
        predictor, mean, scale = world.prior.matched.planning.train_predictor(model, common, common, seed, 2000)
        save(prediction_path, {'signature': signature(), 'predictor': predictor.state_dict(), 'mean': mean, 'scale': scale})
    planning, paths = world.prior.matched.planning_evaluation(model, predictor, mean, scale, seed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'{arm}_seed{seed}.npz', **curves, **paths)
    result = {'signature': signature(), 'arm': arm, 'seed': seed, 'source_hash': world.source_hash(data),
              'regularizer_pool_hash': 'coupled_to_alignment' if arm=='original' else hashlib.sha256(pool.numpy().tobytes()).hexdigest(),
              'initial_metrics': initial, 'metrics': metrics, 'gaussian': gaussian, 'planning': planning, 'history': history}
    save_json(result_path, result)
    partial.unlink(missing_ok=True)
    print(json.dumps({'completed': arm, 'seed': seed}), flush=True)
    return result


def job(task):
    check_lock()
    return train(*task)


def gate():
    torch.set_num_threads(1)
    train('original', 13, OUT / 'gates')
    actual = torch.load(OUT / 'gates/encoder_original_seed13.pt', map_location='cpu')
    old = original.load_encoder('ou', 13)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, actual['encoder'][key], atol=0, rtol=0)
    save_json(OUT / 'GATES.json', {'signature': signature(), 'historical_ou_seed13_bit_exact': True})
    print('Historical original OU gate passed exactly.', flush=True)


def freeze():
    assert not list(OUT.glob('*seed6*.json'))
    assert json.loads((OUT / 'GATES.json').read_text())['signature'] == signature()
    save_json(OUT / 'LOCK.json', {'signature': signature(), 'seeds': list(SEEDS), 'arms': list(ARMS),
              'encoder_fits': 180, 'predictors': 180, 'tests_in_holm_family': 16,
              'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'git_head_before_ou_cohort': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'icm_reference_signature': previous.check_lock()['signature'], 'config': vars(original.config())})


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
        save_json(HERE / 'RUN_STATE.json', {'pid':os.getpid(), 'started_utc':datetime.now(timezone.utc).isoformat()})
        with mp.get_context('spawn').Pool(args.workers) as pool:
            list(pool.imap_unordered(job, [(a,s) for s in SEEDS for a in ARMS]))
        save_json(OUT / 'TRAINING_COMPLETE.json', {'signature':signature(),'encoder_fits':180,'planning_evaluations':180})


if __name__ == '__main__':
    main()

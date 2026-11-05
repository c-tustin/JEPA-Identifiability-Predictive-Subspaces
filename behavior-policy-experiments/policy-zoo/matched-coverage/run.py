"""Run paired, resumable fits and actual goal-reaching tests. CPU only."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import matched as exp

HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def signature(steps, predictor_steps):
    digest = hashlib.sha256()
    for path in [HERE / 'matched.py', HERE / 'run.py']:
        digest.update(path.read_bytes())
    digest.update(f'{steps}:{predictor_steps}'.encode())
    return digest.hexdigest()


def run_one(task):
    name, seed, steps, predictor_steps, output = task
    torch.set_num_threads(1)
    output = Path(output)
    label = f'{name}_seed{seed}'
    result_path = output / (label + '.json')
    protocol_hash = signature(steps, predictor_steps)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved['protocol_hash'] != protocol_hash:
            raise RuntimeError('Existing result uses a different protocol')
        return saved
    design = exp.make_design()
    data = exp.source(design, name, seed)
    common = exp.shared_predictor_data(design, seed)
    c = exp.behavior.Config()
    c.jepa_steps = steps
    torch.manual_seed(40000 + seed)
    encoder = exp.behavior.Encoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(1900000 + seed)
    indices = torch.randint(design['size'], (steps, c.batch), generator=generator)
    regularizer = exp.behavior.SIGReg1D()
    partial = output / (label + '.partial.pt')
    start, history = 0, []
    if partial.exists():
        state = torch.load(partial, map_location='cpu')
        if state['protocol_hash'] != protocol_hash:
            raise RuntimeError('Partial checkpoint uses a different protocol')
        encoder.load_state_dict(state['encoder'])
        optimizer.load_state_dict(state['optimizer'])
        start, history = state['step'], state['history']
    for step in range(start, steps):
        ix = indices[step]
        losses = exp.controls.update(encoder, optimizer, regularizer, data['x'][ix], data['xp'][ix],
                                     c, exp.controls.learning_rate(step, steps))
        if (step + 1) % 1000 == 0 or step == steps - 1:
            metrics, _ = exp.representation_metrics(encoder, design, name)
            history.append({'step': step + 1, **losses, **metrics})
            temporary = partial.with_suffix('.tmp')
            torch.save({'protocol_hash': protocol_hash, 'step': step + 1, 'history': history,
                        'encoder': encoder.state_dict(), 'optimizer': optimizer.state_dict()}, temporary)
            temporary.replace(partial)
    encoder.eval()
    representation, curve = exp.representation_metrics(encoder, design, name)
    torch.save({'encoder': encoder.state_dict(), 'protocol_hash': protocol_hash,
                'config': vars(c)}, output / (label + '.pt'))
    result = {'name': name, 'seed': seed, 'protocol_hash': protocol_hash,
              'representation': representation, 'history': history, 'planning': {}}
    paths = {'distance': design['distance'], 'representation': curve}
    for kind, predictor_data in [('shared', common), ('native', data)]:
        predictor, mean, scale = exp.planning.train_predictor(encoder, predictor_data, common, seed, predictor_steps)
        metrics, rollout = exp.planning_evaluation(encoder, predictor, mean, scale, seed)
        # Exact held-out state/action table, uniform over all feasible pairs.
        i = np.repeat(np.arange(61), 3)
        action = np.tile(np.arange(3), 61)
        keep = (i + action - 1 >= 0) & (i + action - 1 <= 60)
        i, action = i[keep], action[keep]
        d = torch.tensor(design['distance'], dtype=torch.float32)
        with torch.no_grad():
            y = exp.planning.encode(encoder, exp.behavior.sensor(d[i], c), mean, scale)
            yp = exp.planning.encode(encoder, exp.behavior.sensor(d[i + action - 1], c), mean, scale)
            metrics['heldout_latent_mse'] = float((predictor(y, torch.tensor(action)) - yp).square().mean())
        result['planning'][kind] = metrics
        paths.update({kind + '_' + key: value for key, value in rollout.items()})
        torch.save({'predictor': predictor.state_dict(), 'mean': mean, 'scale': scale},
                   output / (label + '_' + kind + '_predictor.pt'))
    np.savez_compressed(output / (label + '.npz'), **paths)
    atomic_json(result_path, result)
    partial.unlink(missing_ok=True)
    print(json.dumps({'name': name, 'seed': seed, 'r2': representation['log_distance_r2'],
                      'goal_error': result['planning']['shared']['goal_error']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(100, 110)))
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--predictor-steps', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', type=Path, default=HERE / 'output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design = exp.make_design()
    manifest = {'protocol_hash': signature(args.steps, args.predictor_steps),
                'steps': args.steps, 'predictor_steps': args.predictor_steps,
                'seeds': args.seeds, 'actor_sha256': design['actor_sha256'],
                'states': 61, 'pairs': design['size'], 'total_flux': design['total_flux']}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Use a different output directory for changed settings')
    atomic_json(manifest_path, manifest)
    np.savez_compressed(args.output / 'design.npz', distance=design['distance'], pi=design['pi'],
                        counts=design['counts'], **{name: design['kernels'][name] for name in exp.NAMES})
    tasks = [(name, seed, args.steps, args.predictor_steps, str(args.output))
             for seed in args.seeds for name in exp.NAMES]
    with mp.get_context('spawn').Pool(args.workers) as pool:
        list(pool.imap_unordered(run_one, tasks))


if __name__ == '__main__':
    main()

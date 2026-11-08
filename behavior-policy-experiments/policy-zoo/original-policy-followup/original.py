"""Use original frozen zoo policies and datasets, with source hash verification."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

HERE = Path(__file__).resolve().parent
ZOO = HERE.parent
sys.path.insert(0, str(ZOO))
import run_zoo
from zoo import behavior, controls, reliability, learned
sys.path.insert(0, str(ZOO / 'matched-coverage'))
import matched

INTERVENTION_POLICIES = ('icm', 'random', 'count', 'ou_actions_lp_w0.1', 'rnd')
PLANNING_POLICIES = INTERVENTION_POLICIES + ('ou',)
SEEDS = tuple(range(8, 28))


def config():
    return behavior.Config(**json.loads((ZOO / 'output/config.json').read_text())['config'])


def raw_record(name, seed, protocol='pure'):
    return json.loads((ZOO / 'output/runs' / f'{protocol}-{name}_seed{seed}.json').read_text())


def load_encoder(name, seed, protocol='pure'):
    state = torch.load(ZOO / 'output/checkpoints' / f'encoder_{protocol}-{name}_data{seed}_init{seed}.pt', map_location='cpu')
    model = behavior.Encoder()
    model.load_state_dict(state['encoder'])
    return model.eval()


def original_source(name, seed):
    """Never retrain a behavior actor. Check every regenerated pair pool."""
    c = config()
    if name in ('icm', 'rnd', 'count'):
        data = None
        for directory in run_zoo.EXISTING_CACHES:
            path = directory / 'sources' / f'{name}_seed{seed}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu')
                data = saved['data']
                break
        if data is None:
            path = ZOO / 'output/policies' / f'{name}_seed{seed}.pt'
            saved = torch.load(path, map_location='cpu')
            actor = learned.load(saved)
            data = learned.collect(c, name, c.train_episodes, 10000 + seed, actor)
    else:
        data, _ = run_zoo.source(c, name, seed, ZOO / 'output')
    actual = run_zoo.diagnostics(data, c)['sha256']
    expected = raw_record(name, seed)['source_stats']['sha256']
    if actual != expected:
        raise RuntimeError(f'Original source differs: {name}, seed {seed}')
    return data, actual


@torch.no_grad()
def geometry(model, points=1025):
    """Affine recovery, distance order, and a refined equal-encoding witness."""
    c = config()
    q = np.linspace(0, np.log(16), points)
    distance = np.exp(q)
    y = model(behavior.sensor(torch.tensor(distance, dtype=torch.float32), c)).numpy()
    sign = 1 if np.cov(q, y)[0, 1] >= 0 else -1
    y = sign * (y - y.mean()) / max(1e-8, y.std())
    diff = np.diff(y)
    result = {'affine_log_r2': matched.affine_r2(q, y),
              'absolute_rank_correlation': float(abs(spearmanr(q, y).statistic)),
              'reversed_fraction': float(np.mean(diff < -1e-6)),
              'folded': bool(np.any(diff < -1e-6)),
              'near_r2': matched.affine_r2(q[distance < 4], y[distance < 4]),
              'far_r2': matched.affine_r2(q[distance >= 9], y[distance >= 9])}
    # Find distinct branches sharing one output. Piecewise interpolation only
    # proposes brackets; bisection evaluates the actual double-precision network.
    network = copy.deepcopy(model).double()
    q_tensor = torch.tensor(q, dtype=torch.float64)
    raw = network(behavior.sensor(q_tensor.exp(), c)).numpy()
    witness = None
    widest = 0.
    for level in np.linspace(raw.min(), raw.max(), 259)[1:-1]:
        crossings = np.where((raw[:-1] - level) * (raw[1:] - level) < 0)[0]
        if len(crossings) < 2:
            continue
        left, right = int(crossings[0]), int(crossings[-1])
        gap = distance[right] - distance[left]
        if gap > widest:
            widest = gap
            witness = (left, right, float(level))
    result.update(alias_distance_gap=0., alias_goal_error_bound=0., alias_residual=0.)
    if witness is not None:
        left, right, level = witness
        low = torch.tensor([q[left], q[right]], dtype=torch.float64)
        high = torch.tensor([q[left+1], q[right+1]], dtype=torch.float64)
        low_sign = network(behavior.sensor(low.exp(), c)) - level
        for _ in range(44):
            middle = (low + high) / 2
            mid_sign = network(behavior.sensor(middle.exp(), c)) - level
            same = low_sign * mid_sign > 0
            low = torch.where(same, middle, low)
            low_sign = torch.where(same, mid_sign, low_sign)
            high = torch.where(same, high, middle)
        roots = ((low + high) / 2).exp()
        residual = float((network(behavior.sensor(roots, c)) - level).abs().max())
        gap = float(roots[1] - roots[0])
        result.update(alias_distance_gap=gap, alias_goal_error_bound=gap/2,
                      alias_residual=residual, alias_goals=roots.tolist())
    return result, {'distance': distance, 'curve': y}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def export_original_geometry(output):
    torch.set_num_threads(1)
    rows, curves = [], {}
    for protocol in ('pure', 'handover'):
        names = list(run_zoo.zoo.PHYSICAL) + (['ou'] if protocol == 'pure' else [])
        for name in names:
            for seed in SEEDS:
                saved = raw_record(name, seed, protocol)
                result, curve = geometry(load_encoder(name, seed, protocol))
                row = {'policy': name, 'seed': seed, 'protocol': protocol,
                       'historical_r2': saved['metrics']['common_grid_r2'],
                       'source_stats': saved['source_stats'], **result}
                rows.append(row)
                curves[f'{protocol}-{name}-{seed}'] = curve['curve']
    save_json(output / 'original_geometry.json', rows)
    np.savez_compressed(output / 'original_curves.npz', distance=curve['distance'], **curves)
    print(f'Exported {len(rows)} original encoder comparisons.', flush=True)

"""Stationary pair laws with identical finite marginals, different local movement.

No new world coordinates. Each edge joins distances separated by the existing
0.25 action. Integer symmetric flows make stationarity and finite coverage exact.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import zoo
sys.path.insert(0, str(HERE.parents[1] / 'planning'))
import evaluate_planning as planning

behavior, controls = zoo.behavior, zoo.controls
NAMES = ('reference', 'near_bottleneck', 'far_bottleneck', 'icm_shaped')
LABELS = {'reference': 'Reference local walk', 'near_bottleneck': 'Slow crossing near 2.5',
          'far_bottleneck': 'Slow crossing near 8', 'icm_shaped': 'ICM-shaped, coverage corrected'}


def integer_allocation(weights, total):
    """Largest-remainder allocation, with deterministic index tie breaking."""
    exact = total * np.asarray(weights, dtype=float) / np.sum(weights)
    result = np.floor(exact).astype(np.int64)
    order = np.argsort(-(exact - result), kind='stable')
    result[order[:total - int(result.sum())]] += 1
    return result


def make_design(size=65536):
    c = behavior.Config()
    distance = np.linspace(c.distance_min, c.distance_max, 61)
    # Integrate a log-uniform density over physical Voronoi cells. This is a
    # fixed finite approximation, NOT a claim of continuous log-uniform support.
    edges = np.r_[distance[0], (distance[1:] + distance[:-1]) / 2, distance[-1]]
    counts = integer_allocation(np.diff(np.log(edges)), size)
    pi = counts / size
    midpoint = (distance[1:] + distance[:-1]) / 2
    base = np.minimum(counts[:-1], counts[1:]).astype(float)
    actor_path = HERE.parent / 'output/policies/icm_seed13.pt'
    saved = torch.load(actor_path, map_location='cpu')
    actor = zoo.learned.load(saved)
    with torch.no_grad():
        probabilities = actor(behavior.sensor(torch.tensor(distance, dtype=torch.float32), c))[0].probs.numpy()
    # Preserve only ICM's relative preference for crossing each adjacent edge.
    # Symmetrization and occupancy correction deliberately change the old policy.
    affinity = np.sqrt(probabilities[:-1, 2] * probabilities[1:, 0])
    shapes = {'reference': np.ones(60),
              'near_bottleneck': 1 - .95 * np.exp(-.5 * ((midpoint - 2.5) / .55)**2),
              'far_bottleneck': 1 - .95 * np.exp(-.5 * ((midpoint - 8) / 1.0)**2),
              'icm_shaped': np.maximum(affinity, .02)}
    normalized = {name: base * shape / np.sum(base * shape) for name, shape in shapes.items()}
    # A common flux sum matches action frequencies and expected |delta d|.
    # Keep every row's move probability <= .4, hence all eigenvalues positive.
    worst = max(np.max((np.r_[0, v] + np.r_[v, 0]) / counts) for v in normalized.values())
    total_flux = int(np.floor(.39 / worst))
    kernels, flows = {}, {}
    for name, weights in normalized.items():
        flow = integer_allocation(weights, total_flux)
        joint_counts = np.diag(counts - np.r_[0, flow] - np.r_[flow, 0])
        joint_counts += np.diag(flow, 1) + np.diag(flow, -1)
        if np.any(flow <= 0) or np.any(np.diag(joint_counts) < counts / 2):
            raise ValueError('Need a larger source size or smaller movement budget')
        flows[name] = joint_counts
        kernels[name] = joint_counts / counts[:, None]
    return {'distance': distance, 'counts': counts, 'pi': pi, 'flows': flows,
            'kernels': kernels, 'actor_sha256': hashlib.sha256(actor_path.read_bytes()).hexdigest(),
            'total_flux': total_flux, 'size': size}


def slow_features(kernel, pi, distance):
    """Eigendecompose the reversible operator in the stationary inner product."""
    root = np.sqrt(pi)
    symmetric = root[:, None] * kernel / root[None, :]
    if not np.allclose(symmetric, symmetric.T, atol=1e-14):
        raise ValueError('This calculation requires reversibility')
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, features = eigenvalues[order], vectors[:, order] / root[:, None]
    # The unique constant mode has eigenvalue 1; exclude it, retain the next 3.
    features = features[:, 1:4]
    for column in range(3):
        if np.sum(pi * features[:, column] * np.log(distance)) < 0:
            features[:, column] *= -1
    return eigenvalues[1:4], features


def source(design, name, seed):
    """Enumerate every integer flow once, then shuffle successors within state.

    Source distances AND successor histograms match exactly across policies.
    These are stratified stationary one-step pairs, not a single rollout.
    """
    generator = np.random.default_rng(1600000 + seed)
    starts, ends = [], []
    for i, row in enumerate(design['flows'][name]):
        destinations = np.repeat(np.arange(len(row)), row)
        generator.shuffle(destinations)
        starts.extend([i] * len(destinations))
        ends.extend(destinations)
    i, j = np.array(starts), np.array(ends)
    d = torch.tensor(design['distance'], dtype=torch.float32)
    c = behavior.Config()
    return {'x': behavior.sensor(d[i], c), 'xp': behavior.sensor(d[j], c),
            'action': torch.tensor(j - i + 1, dtype=torch.long),
            'i': i, 'j': j}


def affine_r2(target, embedding, weights=None):
    weights = np.ones(len(target)) / len(target) if weights is None else weights
    a = target - np.sum(weights * target)
    b = embedding - np.sum(weights * embedding)
    return float(np.sum(weights * a * b)**2 /
                 max(1e-24, np.sum(weights * a*a) * np.sum(weights * b*b)))


@torch.no_grad()
def representation_metrics(encoder, design, name):
    distance = design['distance']
    y = encoder(behavior.sensor(torch.tensor(distance, dtype=torch.float32), behavior.Config())).numpy()
    pi = design['pi']
    y = (y - pi @ y) / max(1e-8, np.sqrt(pi @ (y - pi @ y)**2))
    if np.cov(y, np.log(distance))[0, 1] < 0:
        y *= -1
    eigenvalues, features = slow_features(design['kernels'][name], pi, distance)
    dense_distance = np.geomspace(1, 16, 1024)
    dense_y = encoder(behavior.sensor(torch.tensor(dense_distance, dtype=torch.float32), behavior.Config())).numpy()
    if np.cov(dense_y, np.log(dense_distance))[0, 1] < 0:
        dense_y *= -1
    reversed_fraction = float(np.mean(np.diff(dense_y) < -1e-6))
    return {'log_distance_r2': affine_r2(np.log(distance), y, pi),
            'physical_distance_r2': affine_r2(distance, y, pi),
            'uniform_log_grid_r2': affine_r2(np.log(dense_distance), dense_y),
            'slow_feature_r2': affine_r2(features[:, 0], y, pi),
            'folded': reversed_fraction > 0, 'reversed_fraction': reversed_fraction,
            'eigenvalue_1': float(eigenvalues[0]),
            'alignment_unit_variance': float(np.sum(pi[:, None] * design['kernels'][name] * (y[:, None] - y[None, :])**2))}, y


def shared_predictor_data(design, seed):
    """Every feasible action at every state, same source occupancy as native."""
    generator = np.random.default_rng(1700000 + seed)
    i = np.repeat(np.arange(61), design['counts'])
    actions = generator.integers(0, 3, len(i))
    actions[(i == 0) & (actions == 0)] = 1
    actions[(i == 60) & (actions == 2)] = 1
    j = i + actions - 1
    distance = torch.tensor(design['distance'], dtype=torch.float32)
    c = behavior.Config()
    return {'x': behavior.sensor(distance[i], c), 'xp': behavior.sensor(distance[j], c),
            'action': torch.tensor(actions)}


@torch.no_grad()
def planning_evaluation(encoder, predictor, mean, scale, seed):
    """64 full-range reachable goals, identical across encoder/predictor arms."""
    c = behavior.Config()
    generator = np.random.default_rng(1800000 + seed)
    starts = np.rint(np.linspace(0, 60, 64)).astype(int)
    offsets = generator.choice([-8, -4, -1, 1, 4, 8], len(starts))
    target = np.clip(starts + offsets, 0, 60)
    target[target == starts] = np.where(starts[target == starts] == 0, 1, 59)
    distance = torch.tensor(1 + starts * .25, dtype=torch.float32)
    initial = distance.clone()
    goal = torch.tensor(1 + target * .25, dtype=torch.float32)
    goal_y = planning.encode(encoder, behavior.sensor(goal, c), mean, scale)
    bank = planning.candidate_bank(seed)
    path = [distance.numpy().copy()]
    for _ in range(12):
        y = planning.encode(encoder, behavior.sensor(distance, c), mean, scale)
        action = planning.choose_actions(predictor, y, goal_y, bank)
        distance = behavior.move(distance, action, c)
        path.append(distance.numpy().copy())
    error = (distance - goal).abs().numpy()
    # All desired displacements are <= 8 physical steps, horizon=8, budget=12.
    oracle = initial.clone()
    for _ in range(12):
        action = torch.where(goal - oracle > .125, 2, torch.where(oracle - goal > .125, 0, 1)).long()
        oracle = behavior.move(oracle, action, c)
    assert float((oracle - goal).abs().max()) < 1e-6
    metrics = {'goal_error': float(error.mean()), 'success': float(np.mean(error <= .25001)),
               'stay_error': float((initial - goal).abs().mean()), 'oracle_error': 0.0}
    for label, mask in [('near', starts < 12), ('middle', (starts >= 12) & (starts < 32)), ('far', starts >= 32)]:
        metrics[label + '_error'] = float(error[mask].mean())
    return metrics, {'starts': initial.numpy(), 'goals': goal.numpy(), 'errors': error, 'path': np.stack(path)}

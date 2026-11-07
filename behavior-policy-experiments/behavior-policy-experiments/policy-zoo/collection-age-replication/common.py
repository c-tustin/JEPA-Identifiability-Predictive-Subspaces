"""Shared definitions for the prospective collection-age replication."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'original-policy-followup'))
import original
import run_early

behavior = original.behavior
controls = original.controls
learned = original.learned
matched = original.matched
SEEDS = tuple(range(200, 220))
AGES = (0, 16, 32, 64, 128)
POLICIES = ('icm', 'random', 'count')
OUT = HERE / 'output'


def signature():
    digest = hashlib.sha256()
    for name in ('common.py', 'run_replication.py', 'PROTOCOL.md'):
        digest.update((HERE / name).read_bytes())
    # Capture the upstream routines used by actor/encoder training as well.
    for module in (learned, behavior, controls, original, matched, run_early, original.reliability,
                   sys.modules[behavior.Encoder.__module__], sys.modules[behavior.SIGReg1D.__module__]):
        digest.update(Path(module.__file__).read_bytes())
    digest.update((HERE.parent / 'output/config.json').read_bytes())
    return digest.hexdigest()


def label(policy, age):
    return policy if policy == 'ou' else f'{policy}_age{age:03d}'


def result_path(condition, seed):
    return OUT / f'{condition}_seed{seed}.json'


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def save_json(path, value):
    original.save_json(path, value)


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def data_hash(data):
    digest = hashlib.sha256()
    for key in ('x', 'xp', 'action'):
        digest.update(data[key].numpy().tobytes())
    return digest.hexdigest()


def exposure(data):
    stats = run_early.exposure(data)
    distance = 1 / data['x'][:, 1]
    c = original.config()
    proposed = distance + (data['action'].float() - 1) * c.move
    stats['reflection_fraction'] = float(((proposed < 1 - 1e-5) | (proposed > 16 + 1e-5)).float().mean())
    stats['action_probability'] = (data['action'].bincount(minlength=3).double() / len(distance)).tolist()
    stats['endpoint_log_mean'] = float(torch.cat([data['z'], data['zp']]).mean())
    return stats


def extract_window(full, age, episodes=256, horizon=64):
    start, stop = age * episodes, (age + horizon) * episodes
    return {key: value[start:stop].clone() for key, value in full.items()}


def source(policy, age, seed):
    if policy == 'ou':
        c = original.config()
        return behavior.collect(c, 'ou', c.train_episodes, 10000 + seed)
    saved = torch.load(OUT / 'sources' / f'{policy}_seed{seed}.pt', map_location='cpu')
    if saved['signature'] != signature():
        raise RuntimeError('Source settings changed; use a separate output folder')
    return extract_window(saved['full'], age)

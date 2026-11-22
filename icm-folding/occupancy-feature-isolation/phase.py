"""Exploratory phase map (not pre-registered): end state vs input squashing k and near mass m, initialization 131.

End states, classified from the trained code:
  fold      backward fraction > 0.1
  inflated  no fold, |skewness| > 1 (code scale inflated, tail left outside SIGReg's reach)
  stretched no fold, |skewness| <= 1 (monotone code, heavy region stretched)
Reuses the geometry and kernel-test fits; trains only the missing cells into output/phase.
"""
import argparse
import json
import shutil
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import kernel_test as KT
import geometry as G

K, R = G.K, G.R
GEOM = K.S.OUT/'geometry'; KTEST = K.S.OUT/'kernel-test'
G.OUT = OUT = K.S.OUT/'phase'
MAPS = ['linear', 'tanh1', 'tanh1.5x', 'tanh2x', 'tanh2.5x', 'tanh3x', 'tanh4', 'tanh6x', 'tanh8x']
KVAL = {'linear': 0, 'tanh1': 1, 'tanh4': 4, **{f'tanh{k:g}x': k for k in KT.KS}}
LAWS = ['m0.5', 'm0.625', 'm0.75', 'm0.875', 'm0.9375']


def where(law, mp):
    for base in (GEOM, KTEST, OUT):
        if (base/'cells'/f'{law}-{mp}-i131.json').exists(): return base
    return None


def classify(base, law, mp, x, d):
    w = np.load(base/'laws'/f'{law}.npy'); m = G.MappedEncoder(mp)
    m.load_state_dict(R.trials.load(base/'runs'/f'{law}-{mp}-i131.pt')['model'])
    with torch.no_grad(): y = m(x).double().numpy()
    mu = w @ y; sd = np.sqrt(w @ (y - mu)**2); z = (y - mu)/sd
    z = z*np.sign(np.corrcoef(np.log(d), y)[0, 1])
    b = json.loads((base/'cells'/f'{law}-{mp}-i131.json').read_text())['backward_fraction']; sk = float(w @ z**3)
    return dict(law=law, near_mass=float(law[1:]), input_map=mp, k=KVAL[mp], backward=b, skew=sk, sd=float(sd),
                state='fold' if b > .1 else 'inflated' if abs(sk) > 1 else 'stretched')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    for sub in ('runs', 'cells', 'laws'): (OUT/sub).mkdir(parents=True, exist_ok=True)
    for law in LAWS: shutil.copy(GEOM/'laws'/f'{law}.npy', OUT/'laws'/f'{law}.npy')
    if not (OUT/'PREDICTIONS.json').exists():
        R.trials.atomic_json(OUT/'PREDICTIONS.json', dict(note='exploratory; no predictions', code_hash=R.digest(Path(__file__).read_bytes())))
    setup = K.D.setup(); x, d = K.atoms(setup[0])
    if a.shard is not None:
        # Shard over the fixed grid (not over missing cells), so concurrent shards agree on the split.
        grid = [(law, mp, 131) for law in LAWS for mp in MAPS]
        for k, job in enumerate(grid):
            if k % a.shards == a.shard and where(job[0], job[1]) is None: G.fit(*job, setup, x)
        return
    rows = [classify(where(law, mp), law, mp, x, d) for law in LAWS for mp in MAPS if where(law, mp)]
    R.trials.atomic_json(HERE/'output'/'phase.json', rows)
    for law in LAWS:
        print(law, ' '.join(f"{r['input_map']}:{r['state'][:4]}" for r in rows if r['law'] == law))


if __name__ == '__main__':
    main()

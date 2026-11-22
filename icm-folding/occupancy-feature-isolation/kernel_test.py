"""Prospective kernel-locking test (see KERNEL_PROTOCOL.md). Reuses geometry.fit in its own output directory."""
import argparse
import json
import shutil
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from scipy.stats import spearmanr
import torch
import geometry as G
import kernel as KN

K, R = G.K, G.R
PROTOCOL = HERE/'KERNEL_PROTOCOL.md'
KS = (1.5, 2, 2.5, 3, 6, 8)
for _k in KS:
    G.MAPS[f'tanh{_k:g}x'] = (lambda kk: (lambda v: torch.tanh(kk*v)))(_k)
GEOM_LAWS = G.OUT/'laws'
G.OUT = OUT = K.S.OUT/'kernel-test'
CELLS = [(mp, i) for mp in [f'tanh{k:g}x' for k in KS] for i in (131, 133, 135, 137)] + [('tanh4', 135), ('tanh4', 137)]
LAWS = ('m0.75', 'm0.875')


def design():
    (OUT/'laws').mkdir(parents=True, exist_ok=True)
    for law in LAWS: shutil.copy(GEOM_LAWS/f'{law}.npy', OUT/'laws'/f'{law}.npy')
    _, c, *_ = K.D.setup(); rows = []
    for mp, i in CELLS:
        r = KN.decoupling(mp, i, c); L = r['tail (d 4 vs 16)']/r['near cluster (d 1 vs 1.9)']
        rows.append(dict(input_map=mp, init=i, locking=L, decoupling=r, prediction=bool(L > 5)))
    R.trials.atomic_json(OUT/'PREDICTIONS.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()),
                         code_hash=R.digest(Path(__file__).read_bytes()), threshold=5., cells=rows,
                         law_hashes={l: R.digest((OUT/'laws'/f'{l}.npy').read_bytes()) for l in LAWS}))
    for r in rows: print(r['input_map'], r['init'], round(r['locking'], 1), r['prediction'])


def report():
    pred = json.loads((OUT/'PREDICTIONS.json').read_text()); rows = []
    for p in pred['cells']:
        for law in LAWS:
            f = OUT/'cells'/f"{law}-{p['input_map']}-i{p['init']}.json"
            if not f.exists(): continue
            o = json.loads(f.read_text()); b = o['backward_fraction']
            rows.append(dict(law=law, input_map=p['input_map'], init=p['init'], locking=p['locking'], prediction=p['prediction'],
                             backward=b, junction_fold=b > .1, roots=o['roots'], correct=(b > .1) == p['prediction']))
    L = np.array([r['locking'] for r in rows]); f = np.array([r['junction_fold'] for r in rows])
    bins = {'<5': L < 5, '5-20': (L >= 5) & (L <= 20), '>20': L > 20}
    frac = {k: (float(f[s].mean()) if s.any() else None, int(s.sum())) for k, s in bins.items()}
    vals = [v[0] for v in frac.values() if v[0] is not None]
    rho = spearmanr(np.log(L), [r['backward'] for r in rows])
    summary = dict(rows=rows, correct=int(sum(r['correct'] for r in rows)), scored=len(rows), bin_fold_fraction=frac,
                   ordinal_holds=all(a <= b for a, b in zip(vals, vals[1:])), spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
                   predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes()))
    R.trials.atomic_json(OUT/'summary.json', summary)
    for r in sorted(rows, key=lambda r: r['locking']):
        print(f"{r['input_map']:9s} i{r['init']} L {r['locking']:7.1f} {r['law']:7s} pred {r['prediction']!s:5s} bwd {r['backward']:.3f} {'ok' if r['correct'] else 'WRONG'}")
    print({k: v for k, v in summary.items() if k != 'rows'})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--design', action='store_true'); p.add_argument('--report', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    for sub in ('runs', 'cells'): (OUT/sub).mkdir(parents=True, exist_ok=True)
    if a.design: return design()
    if a.report: return report()
    pred = json.loads((OUT/'PREDICTIONS.json').read_text())
    assert pred['protocol_hash'] == R.digest(PROTOCOL.read_bytes())
    setup = K.D.setup(); x, _ = K.atoms(setup[0])
    jobs = [(law, mp, i) for mp, i in CELLS for law in LAWS]
    for k, job in enumerate(jobs):
        if k % a.shards == a.shard: G.fit(*job, setup, x)


if __name__ == '__main__':
    main()

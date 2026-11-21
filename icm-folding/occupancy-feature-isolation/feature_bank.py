"""Reconstruct the exact target endpoint law and audited outcome of every fit.

Every included fit shares encoder initialization 131, source-131 alignment
pairs, the 0.9/0.1 alignment/SIGReg objective and the 4,000-update schedule;
only the endpoint law P differs. Laws are returned as (distance atoms, weights)
from the protocols' own component definitions (target law, not the realized
uniform-plan draw). Robustness fits with other initializers are excluded here.
"""
import glob
import json
from pathlib import Path
import numpy as np
import torch
import fresh_fit as FF
import support_split as SS

HERE = Path(__file__).resolve().parent
OUT = HERE/'output'


def _d(x): return (1/x[:, 1]).double().numpy()


def _law(parts, weights):
    d = np.concatenate([_d(p) for p in parts])
    w = np.concatenate([np.full(len(p), wt/len(p)) for p, wt in zip(parts, weights)])
    return d, w


def _outcome(r):
    return dict(material=bool(r['material_fold']), folded=bool(r['folded']),
                backward=float(r['backward_fraction']), roots=list(r['roots']))


def cells():
    ds = FF.setup()[0]
    split = lambda s, cut: (ds[s]['x'][ds[s]['d'] < cut], ds[s]['x'][ds[s]['d'] >= cut])
    rows = []
    for f in sorted(glob.glob(str(OUT/'dose/cells/*-sigreg.json'))):
        r = json.load(open(f))
        rows.append(dict(study='dose', name=r['name'], law=_law([ds[r['near']]['near'], ds[r['far']]['far']], [r['alpha'], 1 - r['alpha']]), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'cells/*.json'))):
        r = json.load(open(f))
        rows.append(dict(study='stage1', name=r['name'], law=_law([ds[r['near_source']]['near'], ds[r['far_source']]['far']], [r['alpha'], 1 - r['alpha']]), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'fresh/encoders/cells/*-sigreg.json'))):
        r = json.load(open(f))
        rows.append(dict(study='fresh', name=r['name'], law=_law([ds[r['near']]['near'], ds[r['far']]['far']], [r['alpha'], 1 - r['alpha']]), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'fresh-shapes/*/cells/*-sigreg.json'))):
        r = json.load(open(f))
        rows.append(dict(study='fresh-shapes', name=Path(f).parent.parent.name + '/' + r['name'],
                         law=_law([ds[r['near']]['near'], ds[r['far']]['far']], [r['alpha'], 1 - r['alpha']]), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'far-regions/cells/*.json'))):
        r = json.load(open(f)); pair = r['pair']; shape = pair[r['bits'][2]]
        x = ds[shape]['far']; d = 1/x[:, 1]
        parts = [ds[pair[0]]['near'], x[d < 3], x[(d >= 3) & (d < 6)], x[d >= 6]]
        rows.append(dict(study='far-regions', name=r['name'], law=_law(parts, r['region_weights']), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'deep-shape/cells/*.json'))):
        r = json.load(open(f)); a, b = 9104, 9109; regs = {}
        for s in (a, b):
            x = ds[s]['far']; d = 1/x[:, 1]; regs[s] = [x[d < 3], x[(d >= 3) & (d < 6)], x[d >= 6]]
        parts = [ds[a]['near']] + [regs[(a, b)[bit]][k] for k, bit in enumerate(r['bits'])]
        rows.append(dict(study='deep-shape', name=r['name'], law=_law(parts, r['weights']), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'shoulder/cells/*.json'))):
        r = json.load(open(f)); a, b = 9104, 9109; parts = {}
        for s in (a, b):
            x = ds[s]['far']; d = 1/x[:, 1]; z = x[d < 3]; q = 1/z[:, 1]
            parts[s] = [z[q < 2.5], z[q >= 2.5], x[(d >= 3) & (d < 6)], x[d >= 6]]
        bits = r['bits']
        arrays = [ds[a]['near'], parts[(a, b)[bits[1]]][0], parts[(a, b)[bits[2]]][1], parts[a][2], parts[a][3]]
        rows.append(dict(study='shoulder', name=r['name'], law=_law(arrays, r['weights']), **_outcome(r)))
    for f in sorted(glob.glob(str(OUT/'split-support/split*/pair*/cells/*-sigreg.json'))):
        r = json.load(open(f)); cut = float(Path(f).parts[-4][5:])
        n, _ = split(r['near'], cut); _, fa = split(r['far'], cut)
        rows.append(dict(study=f'split{cut:g}', name=Path(f).parts[-3] + '/' + r['name'], law=_law([n, fa], [r['alpha'], 1 - r['alpha']]), **_outcome(r)))
    pool = SS.common_pool(ds); x, bins, counts = pool[0], pool[1], pool[2]
    for f in sorted(glob.glob(str(OUT/'split-support/common/cells/*.json'))):
        r = json.load(open(f)); target = torch.tensor(json.loads(r['target_bin_mass']) if isinstance(r['target_bin_mass'], str) else r['target_bin_mass'], dtype=torch.float64)
        w = (target[bins]/counts[bins]).numpy()
        rows.append(dict(study='common-support', name=r['name'], law=(_d(x), w), **_outcome(r)))
    return rows


if __name__ == '__main__':
    rows = cells()
    from collections import Counter
    print(Counter(r['study'] for r in rows), sum(r['material'] for r in rows), len(rows))

"""Rigid-group SIGReg gradient test (see RIGID_PROTOCOL.md).  --design | --shard K --shards N | --report"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import geometry as G

K, R, B = G.K, G.R, G.B
OUT = K.S.OUT/'rigid'
PROTOCOL = HERE/'RIGID_PROTOCOL.md'
LAWS = ('m0.625', 'm0.75', 'm0.875')
INITS = (131, 133, 135, 137)
GROUPS = ('tail', 'spike', 'both', 'shift', 'scale')
# tail/spike fits were trained by the version of this file in commit 4f407af4 (same behaviour); 'both' is the addendum.
PRED_FILE = {'tail': 'PREDICTIONS.json', 'spike': 'PREDICTIONS.json', 'both': 'PREDICTIONS_BOTH.json', 'shift': 'PREDICTIONS_SPLIT.json', 'scale': 'PREDICTIONS_SPLIT.json'}
# 'both' was trained by the version of this file in commit 444c05a5 (same behaviour); shift/scale are addendum 2.


def rigid(y, mask):
    """Same values as y; SIGReg gradient on y[mask] reduced to a block shift and rescale."""
    if mask.sum() < 2: return y
    g = y[mask]; mu = g.mean(); sd = g.std(unbiased=False); gd, mud, sdd = g.detach(), mu.detach(), sd.detach()
    out = y.clone(); out[mask] = gd + (mu - mud) + (gd - mud)*(sd/sdd - 1); return out


def part(y, mask, keep):
    """Like rigid(), but keep only the shift ('shift') or only the rescale ('scale') part of the gradient."""
    if mask.sum() < 2: return y
    g = y[mask]; mu = g.mean(); sd = g.std(unbiased=False); gd, mud, sdd = g.detach(), mu.detach(), sd.detach()
    out = y.clone(); out[mask] = gd + ((mu - mud) if keep == 'shift' else (gd - mud)*(sd/sdd - 1)); return out


def project(code, near_mask, group):
    if group in ('shift', 'scale'): return part(part(code, near_mask, group), ~near_mask, group)
    if group == 'spike': return rigid(code, near_mask)
    if group == 'tail': return rigid(code, ~near_mask)
    return rigid(rigid(code, near_mask), ~near_mask)


def design():
    (OUT/'laws').mkdir(parents=True, exist_ok=True)
    for law in LAWS: (OUT/'laws'/f'{law}.npy').write_bytes((G.OUT/'laws'/f'{law}.npy').read_bytes())
    y = torch.randn(64, requires_grad=True); m = torch.arange(64) % 3 == 0
    assert torch.equal(rigid(y, m), y)
    R.trials.atomic_json(OUT/'PREDICTIONS.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()), code_hash=R.digest(Path(__file__).read_bytes()),
                         predictions={'tail': 'no fold in >= 10 of 12', 'spike': 'fold in >= 10 of 12'}))
    print('locked')


def fit(law, group, init_seed, setup, x, d):
    ds, c, _, ix, _, _, _, _ = setup
    tag = f'{law}-{group}-i{init_seed}'; path = OUT/'runs'/f'{tag}.pt'
    w = torch.from_numpy(np.load(OUT/'laws'/f'{law}.npy'))
    u = torch.rand((4000, 1024), dtype=torch.float64, generator=torch.Generator().manual_seed(9600131))
    cdf = w.cumsum(0); cdf[-1] = 1.; ids = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(w) - 1)
    init = G.initial(c, init_seed); near = torch.from_numpy(d < 2)
    identity = dict(law=law, group=group, init=init_seed, law_hash=R.digest(w.numpy().tobytes()), initial_hash=R.state_hash(init),
                    alignment_plan_hash=R.digest(ix.numpy().tobytes()), index_hash=R.digest(ids.numpy().tobytes()),
                    code_hash=R.digest(Path(__file__).read_bytes()), predictions_hash=R.digest((OUT/PRED_FILE[group]).read_bytes()))
    if path.exists():
        assert R.trials.load(path)['identity'] == identity; return
    m = B.Encoder(); m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']
    for step in range(4000):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        idx = ids[step]; code = m(x[idx])
        ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
        loss = .9*(ya[:512] - ya[512:]).square().mean() + .1*reg(project(code, near[idx], group))
        assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step()
    saved = dict(identity=identity, model=m.state_dict(), config=vars(c)); R.trials.save_torch(path, saved)
    check = B.Encoder(); check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry'] = R.geometry(check, c, 20001); R.trials.save_torch(path, saved)
    wn = w.numpy()
    with torch.no_grad(): y = check(x).double().numpy()
    mu = wn @ y; sd = np.sqrt(wn @ (y - mu)**2); z = (y - mu)/sd*np.sign(np.corrcoef(np.log(d), y)[0, 1])
    R.trials.atomic_json(OUT/'cells'/f'{tag}.json', dict(**identity, **saved['geometry'], sd=float(sd), skew=float(wn @ z**3)))
    print(tag, saved['geometry']['backward_fraction'], flush=True)


def report():
    rows = []
    for g in GROUPS:
        for law in LAWS:
            for i in INITS:
                p = OUT/'cells'/f'{law}-{g}-i{i}.json'
                if p.exists():
                    o = json.loads(p.read_text()); rows.append(dict(group=g, law=law, init=i, backward=o['backward_fraction'],
                                                                    fold=o['backward_fraction'] > .1, skew=o['skew'], sd=o['sd']))
    verdict = {}
    for g in GROUPS:
        f = [r['fold'] for r in rows if r['group'] == g]
        need = {'tail': lambda f: len(f) - sum(f) >= 10, 'spike': lambda f: sum(f) >= 10, 'both': lambda f: sum(f) >= 10,
                'shift': lambda f: sum(f) >= 7, 'scale': lambda f: sum(f) <= 1}[g]
        verdict[g] = dict(cells=len(f), folds=sum(f), correct=need(f) if len(f) in (8, 12) else None)
    # predictions_hash records both locked files
    R.trials.atomic_json(OUT/'summary.json', dict(rows=rows, verdict=verdict, predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes()),
                         addendum_predictions_hash=R.digest((OUT/'PREDICTIONS_BOTH.json').read_bytes()),
                         addendum2_predictions_hash=R.digest((OUT/'PREDICTIONS_SPLIT.json').read_bytes())))
    for r in rows: print(f"{r['group']:5s} {r['law']:7s} i{r['init']} bwd {r['backward']:.3f} skew {r['skew']:+.2f} sd {r['sd']:.2f}")
    print(verdict)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--design', action='store_true'); p.add_argument('--report', action='store_true')
    p.add_argument('--addendum', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    for sub in ('runs', 'cells'): (OUT/sub).mkdir(parents=True, exist_ok=True)
    if a.design: return design()
    if a.report: return report()
    if a.addendum:
        y = torch.randn(64, requires_grad=True); m = torch.arange(64) % 3 == 0
        assert torch.equal(project(y, m, 'shift'), y) and torch.equal(project(y, m, 'scale'), y)
        R.trials.atomic_json(OUT/'PREDICTIONS_SPLIT.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()), code_hash=R.digest(Path(__file__).read_bytes()),
                             predictions={'shift': 'fold in >= 7 of 8', 'scale': 'fold in <= 1 of 8'}))
        return print('addendum 2 locked')
    assert json.loads((OUT/'PREDICTIONS_SPLIT.json').read_text())['protocol_hash'] == R.digest(PROTOCOL.read_bytes())
    setup = K.D.setup(); x, d = K.atoms(setup[0])
    jobs = [(law, g, i) for g in ('shift', 'scale') for law in ('m0.75', 'm0.875') for i in INITS]
    for k, job in enumerate(jobs):
        if k % a.shards == a.shard: fit(*job, setup, x, d)


if __name__ == '__main__':
    main()

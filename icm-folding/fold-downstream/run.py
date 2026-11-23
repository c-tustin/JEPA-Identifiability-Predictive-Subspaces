"""Does the fold itself hurt downstream use? Folded encoder vs its monotone rearrangement.

See PROTOCOL.md. Usage from the workspace root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/fold-downstream/run.py [--workers 10]
    python3 icm-folding/fold-downstream/run.py --report-only
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/icm-fold-downstream-mpl')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "sources"
LEGACY = ROOT/'predictive-closure-workspace/behavior-policy-experiments/qualitative-study'
CROSSING = HERE.parent/'initialization-crossing/output'
SOURCES = LEGACY/'output/sources'
OUT = HERE/'output'
sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location('downstream_trials', LEGACY/'run_trials.py')
trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trials)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

B, P = trials.behavior, trials.planning
GRID = 20001
LOG16 = float(np.log(16))
REPS = 3
PROTOCOL_SHA = '816b56b924891c9c80863ff9b2f820e732864427bd239531a35a23589d8d30a2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Table(nn.Module):
    """Observation -> d = 1/x1 (exact for this renderer) -> linear interpolation of a code table over log d."""
    def __init__(self, table):
        super().__init__()
        self.table = torch.as_tensor(table, dtype=torch.float64)

    def forward(self, x):
        q = torch.log(1/x[:, 1].double().clamp_min(1e-12)).clamp(0, LOG16)
        pos = q/LOG16*(GRID-1)
        i0 = pos.floor().long().clamp(0, GRID-2)
        f = pos-i0
        return (self.table[i0]*(1-f)+self.table[i0+1]*f).float()


def curves(state, c):
    model = B.Encoder()
    model.load_state_dict(state)
    m = model.double().eval()
    q = torch.linspace(0, LOG16, GRID, dtype=torch.float64)
    with torch.no_grad():
        y = m(B.sensor(q.exp(), c)).numpy()
    s = 1.0 if np.corrcoef(y, q.numpy())[0, 1] >= 0 else -1.0
    rearranged = s*np.sort(s*y)
    # Network vs wrapper check at off-grid points, in float32 as used downstream.
    probe = torch.exp(torch.rand(4096, generator=torch.Generator().manual_seed(7))*LOG16)
    net = model.float()
    with torch.no_grad():
        err = float((net(B.sensor(probe, c)) - Table(y)(B.sensor(probe, c))).abs().max())
    return q.numpy(), y, rearranged, s, err


def near_goals(c, seed, cases=64):
    g = torch.Generator().manual_seed(1600000+seed)
    start = 1+2.5*torch.rand(cases, generator=g)
    k = torch.zeros(cases, dtype=torch.long)
    bad = torch.ones(cases, dtype=torch.bool)
    while bad.any():
        k[bad] = torch.randint(-8, 9, (int(bad.sum()),), generator=g)
        bad = (k == 0) | (start+c.move*k < 1)
    return start, start+c.move*k


def oracle_path(start, goal, c):
    d, path = start.clone(), [start.clone()]
    for _ in range(12):
        a = torch.where(goal-d > c.move/2, 2, torch.where(d-goal > c.move/2, 0, 1)).long()
        d = B.move(d, a, c)
        path.append(d.clone())
    assert float((d-goal).abs().max()) < 1e-4
    return torch.stack(path)


@torch.no_grad()
def plan(encoder, predictor, mean, scale, c, start, goal, bank):
    d = start.clone()
    goal_y = P.encode(encoder, B.sensor(goal, c), mean, scale)
    for _ in range(12):
        y = P.encode(encoder, B.sensor(d, c), mean, scale)
        d = B.move(d, P.choose_actions(predictor, y, goal_y, bank), c)
    e = (d-goal).abs()
    return dict(error=float(e.mean()), success=float((e <= c.move+1e-5).float().mean()))


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, n_out))


def fit(net, xx, target, seed, updates=1500):
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        ix = torch.randint(len(xx), (256,), generator=gen)
        loss = F.mse_loss(net(xx[ix]), target[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net


def next_obs(encoder, train, test, common, seed):
    t = torch.tanh(4*train['xp'])
    mean, std = t.mean(0), t.std(0, unbiased=False).clamp_min(.05)
    with torch.no_grad():
        code = encoder(train['x'])
        center, scale = code.mean(), code.std(unbiased=False).clamp_min(.05)
        feat = lambda d: torch.cat([((encoder(d['x'])-center)/scale)[:, None], F.one_hot(d['action'], 3).float()], 1)
        tgt = lambda d: (torch.tanh(4*d['xp'])-mean)/std
    torch.manual_seed(5100000+seed)
    net = fit(mlp(4, 6), feat(train), tgt(train), 5200000+seed)
    with torch.no_grad():
        return dict(native=float(F.mse_loss(net(feat(test)), tgt(test))),
                    common=float(F.mse_loss(net(feat(common)), tgt(common))))


def readout(encoder, train, test, c, seed):
    with torch.no_grad():
        code = encoder(train['x'])
        center, scale = code.mean(), code.std(unbiased=False).clamp_min(.05)
        feat = lambda x: ((encoder(x)-center)/scale)[:, None]
        logd = lambda x: torch.log(1/x[:, 1].clamp_min(1e-12))[:, None]
    torch.manual_seed(5300000+seed)
    net = fit(mlp(1, 1), feat(train['x']), logd(train['x']), 5400000+seed)
    grid = B.sensor(torch.exp(torch.linspace(0, LOG16, 1024)), c)
    with torch.no_grad():
        return dict(native=float(F.mse_loss(net(feat(test['x'])), logd(test['x']))),
                    grid=float(F.mse_loss(net(feat(grid)), logd(grid))))


def cell(row):
    torch.set_num_threads(1)
    ds, init = row['data_seed'], row['init_seed']
    path = OUT/'cells'/f'data{ds}-init{init}.json'
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    ckpt_path = CROSSING/'runs'/f'data{ds}-init{init}-n4000.pt'
    ckpt = trials.load(ckpt_path)
    c = B.Config(**ckpt['config'])
    q, y, yr, s, wrap_err = curves(ckpt['model'], c)
    src = trials.load(SOURCES/f'icm-{ds}.pt')['data']
    data = {k: src[k] for k in ('x', 'xp', 'action')}
    common = {k: v for k, v in trials.load(SOURCES/f'physical-common-{ds}.pt').items() if k in ('x', 'xp', 'action')}
    episode = torch.arange(len(data['x'])) % c.train_episodes
    mask = episode < int(.8*c.train_episodes)
    train = {k: v[mask] for k, v in data.items()}
    test = {k: v[~mask] for k, v in data.items()}

    moved = np.abs(yr-y) > 1e-3*(y.max()-y.min())
    def moved_share(dist):
        idx = np.clip(np.rint(np.log(dist.numpy().ravel())/LOG16*(GRID-1)).astype(int), 0, GRID-1)
        return float(moved[idx].mean())
    goal_sets = dict(standard=P.goals(c, 10000+ds, 64), near=near_goals(c, ds))
    paths = {k: oracle_path(*v, c) for k, v in goal_sets.items()}
    q_moved = q[moved]
    result = dict(data_seed=ds, init_seed=init, material_fold=row['material_fold'],
                  backward_fraction=row['backward_fraction'], orientation=s, wrapper_max_abs_error=wrap_err,
                  multiset_equal=bool(np.array_equal(np.sort(y), np.sort(yr))),
                  moved_distance_span=[float(np.exp(q_moved.min())), float(np.exp(q_moved.max()))] if moved.any() else None,
                  moved_grid_fraction=float(moved.mean()),
                  moved_native_mass=moved_share(1/data['x'][:, 1].clamp_min(1e-12)),
                  moved_route_share={k: moved_share(v) for k, v in paths.items()},
                  checkpoint_sha256=sha(ckpt_path), protocol_sha256=PROTOCOL_SHA, arms={})
    bank = P.candidate_bank(10000+ds)
    for arm, table in (('original', y), ('rearranged', yr)):
        enc = Table(table)
        reps = []
        for r in range(REPS):
            seed = ds+100000*r
            rep = dict(next_obs=next_obs(enc, train, test, common, seed), readout=readout(enc, train, test, c, seed), planning={})
            for kind, pairs in (('shared', common), ('native', data)):
                pred, mean, scale = P.train_predictor(enc, pairs, common, seed, 1000)
                rep['planning'][kind] = {g: plan(enc, pred, mean, scale, c, *goal_sets[g], bank) for g in goal_sets}
            reps.append(rep)
        result['arms'][arm] = reps
    result['seconds'] = time.time()-t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(f'data{ds}-init{init} done {result["seconds"]:.0f}s', flush=True)
    return result


METRICS = [('next_obs', 'native'), ('next_obs', 'common'), ('readout', 'native'), ('readout', 'grid')] + \
          [('planning', k, g, m) for k in ('shared', 'native') for g in ('standard', 'near') for m in ('error', 'success')]


def get(rep, key):
    for k in key:
        rep = rep[k]
    return rep


def name(key):
    return '/'.join(key)


def report():
    import matplotlib.pyplot as plt
    rows = [json.loads(p.read_text()) for p in sorted((OUT/'cells').glob('*.json'))]
    material = [r for r in rows if r['material_fold']]
    datasets = sorted({r['data_seed'] for r in material})
    enc = {}
    for r in rows:
        enc[(r['data_seed'], r['init_seed'])] = {name(k): {a: float(np.mean([get(rep, k) for rep in r['arms'][a]])) for a in r['arms']} for k in METRICS}
    # Noise yardstick: replicate SD of the original arm, averaged over material encoders.
    noise = {name(k): float(np.mean([np.std([get(rep, k) for rep in r['arms']['original']], ddof=1) for r in material])) for k in METRICS}
    by_ds = {}
    for ds in datasets:
        rs = [r for r in material if r['data_seed'] == ds]
        entry = dict(n=len(rs), moved_native_mass=float(np.mean([r['moved_native_mass'] for r in rs])),
                     moved_route_standard=float(np.mean([r['moved_route_share']['standard'] for r in rs])),
                     moved_route_near=float(np.mean([r['moved_route_share']['near'] for r in rs])),
                     moved_span=[float(np.min([r['moved_distance_span'][0] for r in rs])), float(np.max([r['moved_distance_span'][1] for r in rs]))])
        for k in METRICS:
            o = np.mean([enc[(r['data_seed'], r['init_seed'])][name(k)]['original'] for r in rs])
            n = np.mean([enc[(r['data_seed'], r['init_seed'])][name(k)]['rearranged'] for r in rs])
            entry[name(k)] = dict(original=float(o), rearranged=float(n), change=float(n-o), ratio=float(n/o) if o > 0 else None)
        by_ds[ds] = entry
    h1 = sum(by_ds[d]['next_obs/native']['ratio'] < 1 for d in datasets)
    h2_small = {d: abs(by_ds[d]['planning/shared/standard/error']['change']) < .05 for d in (131, 132, 133, 139) if d in by_ds}
    h2_136 = by_ds[136]['planning/shared/standard/error']['change'] < -.2 if 136 in by_ds else None
    h3 = sum(by_ds[d]['planning/shared/near/error']['change'] < 0 for d in datasets)
    h4 = sum(by_ds[d]['readout/native']['change'] < 0 for d in datasets)
    hyp = dict(H1=dict(count=h1, of=len(datasets), held=h1 >= 5), H2=dict(small=h2_small, d136=h2_136, held=all(h2_small.values()) and bool(h2_136)),
               H3=dict(count=h3, of=len(datasets), held=h3 >= 5), H4=dict(count=h4, of=len(datasets), held=h4 >= 5))
    checks = dict(cells=len(rows), material=len(material), multiset_equal=all(r['multiset_equal'] for r in rows),
                  max_wrapper_error=max(r['wrapper_max_abs_error'] for r in rows),
                  protocol_unchanged=sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA)
    summary = dict(checks=checks, hypotheses=hyp, noise_sd=noise, by_dataset=by_ds,
                   tiny_folds={f"{r['data_seed']}-{r['init_seed']}": enc[(r['data_seed'], r['init_seed'])] for r in rows if not r['material_fold']},
                   encoders={f'{k[0]}-{k[1]}': v for k, v in enc.items()})
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1))

    # Figure: per-encoder original vs rearranged for six key metrics.
    keys = ['next_obs/native', 'readout/native', 'planning/shared/standard/error', 'planning/shared/near/error',
            'planning/native/standard/error', 'planning/native/near/error']
    titles = ['Next-obs MSE (native held-out)', 'log-d readout MSE (native held-out)', 'Planning error, standard goals, shared data',
              'Planning error, near goals, shared data', 'Planning error, standard goals, native data', 'Planning error, near goals, native data']
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    colors = dict(zip(datasets, plt.cm.tab10.colors))
    for ax, k, t in zip(axes.flat, keys, titles):
        for r in material:
            v = enc[(r['data_seed'], r['init_seed'])][k]
            ax.scatter(v['original'], v['rearranged'], s=18, color=colors[r['data_seed']], label=str(r['data_seed']))
        lo = min(min(enc[(r['data_seed'], r['init_seed'])][k].values()) for r in material)
        hi = max(max(enc[(r['data_seed'], r['init_seed'])][k].values()) for r in material)
        ax.plot([lo, hi], [lo, hi], color='0.5', lw=1)
        if k.startswith('next') or k.startswith('readout'):
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(t, fontsize=10); ax.set_xlabel('folded (original)'); ax.set_ylabel('fold removed (rearranged)')
    h, l = axes.flat[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), [f'dataset {k}' for k in uniq], loc='lower center', ncol=len(uniq))
    fig.suptitle('Same encoder, fold removed: points below the diagonal mean the fold was costing performance')
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(OUT/'downstream.png', dpi=150)

    # Example curves.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, ds in zip(axes, (131, 136, 139)):
        ckpt = trials.load(CROSSING/'runs'/f'data{ds}-init{ds}-n4000.pt')
        q, y, yr, _, _ = curves(ckpt['model'], B.Config(**ckpt['config']))
        ax.plot(np.exp(q), y, label='trained (folded)'); ax.plot(np.exp(q), yr, '--', label='rearranged')
        ax.set_xscale('log'); ax.set_title(f'dataset {ds}, init {ds}'); ax.set_xlabel('distance d'); ax.set_ylabel('code')
    axes[0].legend()
    fig.tight_layout(); fig.savefig(OUT/'rearrangement_examples.png', dpi=150)
    print(json.dumps(dict(checks=checks, hypotheses=hyp), indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    assert sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA, 'PROTOCOL.md changed after lock; add an addendum instead'
    if not a.report_only:
        rows = [r for r in json.loads((CROSSING/'audited-summary.json').read_text())['rows'] if r['folded']]
        assert len(rows) == 55 and sum(r['material_fold'] for r in rows) == 51
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(cell, rows):
                pass
    report()


if __name__ == '__main__':
    main()

"""Multi-dimensional SIGReg on ICM occupancy: aliasing and downstream use. See PROTOCOL.md.

From the workspace root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/multidim-sigreg/run.py [--workers 10]
    python3 icm-folding/multidim-sigreg/run.py --report-only
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util
_spec = importlib.util.spec_from_file_location('fold_downstream_harness', HERE.parent/'fold-downstream/run.py')
R = importlib.util.module_from_spec(_spec)  # Protocol 1 harness (paths, trials, behavior, goals, sha)
_spec.loader.exec_module(R)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

B, P, trials = R.B, R.P, R.trials
OUT = HERE/'output'
OCC = R.ROOT/'explanations/occupancy-folding'
PROTOCOL_SHA = '889d04a157e04f473196cdb7edd6885e5161bd37a1e26f237d1931cc2dcb173c'
SOURCES = tuple(range(130, 140))
DIMS = (1, 2, 4, 8, 16)
ARMS = ('ordinary', 'weighted')
DIRECTIONS = 64
STEPS = 4000


class EncoderD(nn.Module):
    """The original encoder with a D-dimensional output layer (same layer order and names)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, dim))

    def forward(self, x):
        return self.net(torch.tanh(4*x))


def sigreg(z, reg, weights, directions):
    """Original 1-d statistic at D = 1; average over normalized random projections otherwise."""
    if z.shape[1] == 1:
        y = z[:, 0]
        if weights is None:
            return reg(y)
        w = weights/weights.mean()  # same operation order as occupancy-folding/run_study.py::penalty
        phase = y[:, None]*reg.frequency
        w = w/w.sum()
        co = (w[:, None]*phase.cos()).sum(0)
        si = (w[:, None]*phase.sin()).sum(0)
        return len(y)*(((co-reg.target).square()+si.square())*reg.weight).sum()
    proj = z@directions.T                                  # [N, M]
    phase = proj[:, :, None]*reg.frequency                 # [N, M, K]
    if weights is None:
        co, si = phase.cos().mean(0), phase.sin().mean(0)
    else:
        w = weights/weights.mean()
        w = (w/w.sum())[:, None, None]
        co, si = (w*phase.cos()).sum(0), (w*phase.sin()).sum(0)
    stat = len(z)*(((co-reg.target).square()+si.square())*reg.weight).sum(-1)
    return stat.mean()


def train(s, dim, arm):
    path = OUT/'runs'/f'{arm}-d{dim}-{s}.pt'
    if path.exists():
        return trials.load(path)
    source = trials.load(R.SOURCES/f'icm-{s}.pt')
    c = B.Config(**source['config'])
    data = {k: source['data'][k] for k in ('x', 'xp')}
    _, plan = trials.study.make_plan(c, s, s, 'icm')
    torch.manual_seed(40000+s)
    model = EncoderD(dim)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    indices = plan['indices'][:STEPS]
    reg_indices = torch.randint(len(data['x']), plan['indices'].shape, generator=torch.Generator().manual_seed(1900000+s))[:STEPS]
    endpoints = torch.cat([data['x'], data['xp']])
    bins = (trials.distances(endpoints).log()/np.log(16)*32).long().clamp(0, 31)
    mass = torch.bincount(bins, minlength=32).float()/len(bins)
    inverse = 1/mass[bins]
    n = len(data['x'])
    reg = B.SIGReg1D()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    dirgen = torch.Generator().manual_seed(3000000+s)
    t0 = time.time()
    history = []
    for step in range(STEPS):
        opt.param_groups[0]['lr'] = trials.study.controls.learning_rate(step, c.jepa_steps)
        ix, ri = indices[step], reg_indices[step]
        z = model(torch.cat([data['x'][ix], data['xp'][ix]]))
        alignment = (z[:c.batch]-z[c.batch:]).square().mean()
        directions = None
        if dim > 1:
            directions = torch.randn(DIRECTIONS, dim, generator=dirgen)
            directions = directions/directions.norm(dim=1, keepdim=True)
        rz = model(torch.cat([data['x'][ri], data['xp'][ri]]))
        weights = inverse[torch.cat([ri, ri+n])] if arm == 'weighted' else None
        penalty = sigreg(rz, reg, weights, directions)
        loss = (1-c.sigreg_weight)*alignment+c.sigreg_weight*penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step+1) % 500 == 0:
            history.append(dict(step=step+1, alignment=float(alignment), regularizer=float(penalty)))
    saved = dict(source=s, dim=dim, arm=arm, initial=initial, model=model.state_dict(), history=history,
                 seconds=time.time()-t0, protocol_sha256=PROTOCOL_SHA, config=vars(c))
    path.parent.mkdir(parents=True, exist_ok=True)
    trials.save_torch(path, saved)
    return saved


def d1_reference_error(saved):
    """Max |param difference| against the stored 1-d checkpoint this D = 1 fit must reproduce."""
    s = saved['source']
    if saved['arm'] == 'ordinary':
        ref = trials.load(R.CROSSING/'runs'/f'data{s}-init{s}-n4000.pt')['model']
    else:
        ref = trials.load(OCC/'output/runs'/f'native-weighted_moments-{s}.pt')['encoder']
    return max(float((saved['model'][k]-ref[k]).abs().max()) for k in ref)


def load_model(saved):
    m = EncoderD(saved['dim'])
    m.load_state_dict(saved['model'])
    return m.double().eval()


@torch.no_grad()
def aliasing(model, c, endpoints):
    d = torch.exp(torch.linspace(0, R.LOG16, 2049, dtype=torch.float64))
    f = model(B.sensor(d, c))
    step = model(B.sensor((d+c.move).clamp(max=16.), c))
    local = (step-f).norm(dim=1)
    dist = torch.cdist(f, f)
    far = (d[:, None]-d[None, :]).abs() > 1.0
    ratio = torch.where(far, dist, torch.full_like(dist, np.inf)).min(1).values/local.clamp_min(1e-300)
    ratio[local == 0] = np.inf
    aliased = (ratio < 1).numpy()
    q = torch.log(d).numpy()
    idx = np.clip(np.rint(np.log(endpoints.numpy())/R.LOG16*2048).astype(int), 0, 2048)
    return dict(aliased_native_mass=float(aliased[idx].mean()), aliased_grid_fraction=float(aliased.mean()),
                min_far_ratio=float(ratio[torch.isfinite(ratio)].min()),
                aliased_span=[float(np.exp(q[aliased].min())), float(np.exp(q[aliased].max()))] if aliased.any() else None)


class Table(nn.Module):
    """Protocol 1 interpolating wrapper, column by column."""
    def __init__(self, table):
        super().__init__()
        self.table = torch.as_tensor(table, dtype=torch.float64)

    def forward(self, x):
        q = torch.log(1/x[:, 1].double().clamp_min(1e-12)).clamp(0, R.LOG16)
        pos = q/R.LOG16*(R.GRID-1)
        i0 = pos.floor().long().clamp(0, R.GRID-2)
        f = (pos-i0)[:, None]
        return (self.table[i0]*(1-f)+self.table[i0+1]*f).float()


class PredictorD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+3, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, dim))

    def forward(self, y, action):
        return y+self.net(torch.cat([y, F.one_hot(action, 3).float()], -1))


def train_predictor(enc, data, common, seed, updates, dim):
    torch.manual_seed(1200000+seed)
    model = PredictorD(dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    with torch.no_grad():
        cal = enc(common['x'])
        mean, scale = cal.mean(0), cal.std(0, unbiased=False).clamp_min(1e-6)
        y, yp = (enc(data['x'])-mean)/scale, (enc(data['xp'])-mean)/scale
    gen = torch.Generator().manual_seed(1300000+seed)
    for _ in range(updates):
        ix = torch.randint(len(y), (256,), generator=gen)
        loss = (model(y[ix], data['action'][ix])-yp[ix]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval(), mean, scale


@torch.no_grad()
def choose(pred, y, goal, bank):
    count, dim = len(bank), y.shape[1]
    states = y[:, None, :].expand(-1, count, -1).reshape(-1, dim)
    rgoal = goal.repeat_interleave(count, 0)
    stage = torch.zeros(len(states))
    for t in range(bank.shape[1]):
        states = pred(states, bank[:, t].repeat(len(y)))
        stage += (states-rgoal).square().sum(-1)
    cost = (states.reshape(len(y), count, dim)-goal[:, None, :]).square().sum(-1)
    cost += .01*stage.reshape(len(y), count)
    cost += 1e-8*(bank.float()-1).square().sum(1)[None]
    return bank[cost.argmin(1), 0]


@torch.no_grad()
def plan(enc, pred, mean, scale, c, start, goal, bank):
    d = start.clone()
    gy = (enc(B.sensor(goal, c))-mean)/scale
    for _ in range(12):
        y = (enc(B.sensor(d, c))-mean)/scale
        d = B.move(d, choose(pred, y, gy, bank), c)
    e = (d-goal).abs()
    return dict(error=float(e.mean()), success=float((e <= c.move+1e-5).float().mean()))


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, n_out))


def next_obs(enc, train_, test, common, seed, dim):
    t = torch.tanh(4*train_['xp'])
    mean, std = t.mean(0), t.std(0, unbiased=False).clamp_min(.05)
    with torch.no_grad():
        code = enc(train_['x'])
        center, scale = code.mean(0), code.std(0, unbiased=False).clamp_min(.05)
        feat = lambda d: torch.cat([(enc(d['x'])-center)/scale, F.one_hot(d['action'], 3).float()], 1)
        tgt = lambda d: (torch.tanh(4*d['xp'])-mean)/std
    torch.manual_seed(5100000+seed)
    net = R.fit(mlp(dim+3, 6), feat(train_), tgt(train_), 5200000+seed)
    with torch.no_grad():
        return dict(native=float(F.mse_loss(net(feat(test)), tgt(test))), common=float(F.mse_loss(net(feat(common)), tgt(common))))


def readout(enc, train_, test, c, seed, dim):
    with torch.no_grad():
        code = enc(train_['x'])
        center, scale = code.mean(0), code.std(0, unbiased=False).clamp_min(.05)
        feat = lambda x: (enc(x)-center)/scale
        logd = lambda x: torch.log(1/x[:, 1].clamp_min(1e-12))[:, None]
    torch.manual_seed(5300000+seed)
    net = R.fit(mlp(dim, 1), feat(train_['x']), logd(train_['x']), 5400000+seed)
    grid = B.sensor(torch.exp(torch.linspace(0, R.LOG16, 1024)), c)
    with torch.no_grad():
        return dict(native=float(F.mse_loss(net(feat(test['x'])), logd(test['x']))), grid=float(F.mse_loss(net(feat(grid)), logd(grid))))


def job(spec):
    torch.set_num_threads(1)
    s, dim, arm = spec
    out = OUT/'cells'/f'{arm}-d{dim}-{s}.json'
    if out.exists():
        return json.loads(out.read_text())
    t0 = time.time()
    saved = train(s, dim, arm)
    model = load_model(saved)
    src = trials.load(R.SOURCES/f'icm-{s}.pt')['data']
    c = B.Config(**saved['config'])
    data = {k: src[k] for k in ('x', 'xp', 'action')}
    common = {k: v for k, v in trials.load(R.SOURCES/f'physical-common-{s}.pt').items() if k in ('x', 'xp', 'action')}
    endpoints = trials.distances(torch.cat([data['x'], data['xp']])).double()
    result = dict(source=s, dim=dim, arm=arm, train_seconds=saved['seconds'], protocol_sha256=PROTOCOL_SHA,
                  **aliasing(model, c, endpoints))
    if dim == 1:
        result['d1_reference_max_abs_error'] = d1_reference_error(saved)
        q = torch.linspace(0, R.LOG16, R.GRID, dtype=torch.float64)
        with torch.no_grad():
            y = model(B.sensor(q.exp(), c))[:, 0].numpy()
        dy = np.diff(y)
        up, down = np.maximum(dy, 0).sum(), np.maximum(-dy, 0).sum()
        result['scalar_backward_fraction'] = float(min(up, down)/max(up+down, 1e-12))
    q = torch.linspace(0, R.LOG16, R.GRID, dtype=torch.float64)
    with torch.no_grad():
        enc = Table(model(B.sensor(q.exp(), c)).numpy())
    episode = torch.arange(len(data['x'])) % c.train_episodes
    mask = episode < int(.8*c.train_episodes)
    tr = {k: v[mask] for k, v in data.items()}
    te = {k: v[~mask] for k, v in data.items()}
    goal_sets = dict(standard=P.goals(c, 10000+s, 64), near=R.near_goals(c, s))
    bank = P.candidate_bank(10000+s)
    reps = []
    for r in range(R.REPS):
        seed = s+100000*r
        rep = dict(next_obs=next_obs(enc, tr, te, common, seed, dim), readout=readout(enc, tr, te, c, seed, dim), planning={})
        for kind, pairs in (('shared', common), ('native', data)):
            pred, mean, scale = train_predictor(enc, pairs, common, seed, 1000, dim)
            rep['planning'][kind] = {g_: plan(enc, pred, mean, scale, c, *goal_sets[g_], bank) for g_ in goal_sets}
        reps.append(rep)
    result['reps'] = reps
    result['seconds'] = time.time()-t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f'{arm}-d{dim}-{s} done {result["seconds"]:.0f}s aliased_mass={result["aliased_native_mass"]:.3f}', flush=True)
    return result


KEYS = {'next_obs': ('next_obs', 'native'), 'readout': ('readout', 'native'),
        'near_shared': ('planning', 'shared', 'near', 'error'), 'standard_shared': ('planning', 'shared', 'standard', 'error'),
        'near_native': ('planning', 'native', 'near', 'error'), 'standard_native': ('planning', 'native', 'standard', 'error'),
        'near_success': ('planning', 'shared', 'near', 'success')}


def report():
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    cells = {(r['arm'], r['dim'], r['source']): r for r in (json.loads(p.read_text()) for p in (OUT/'cells').glob('*.json'))}
    m = lambda r, k: float(np.mean([R.get(rep, KEYS[k]) for rep in r['reps']]))
    aliased = lambda r: r['aliased_native_mass'] > .01
    rows = {f'{a}/d{d}/{s}': dict(aliased_native_mass=r['aliased_native_mass'], aliased_grid_fraction=r['aliased_grid_fraction'],
                                  min_far_ratio=r['min_far_ratio'], aliased_span=r['aliased_span'], **{k: m(r, k) for k in KEYS})
            for (a, d, s), r in cells.items()}
    by = {}
    for a in ARMS:
        for d in DIMS:
            rs = [cells[(a, d, s)] for s in SOURCES]
            by[f'{a}/d{d}'] = dict(aliased=sum(aliased(r) for r in rs), mean_aliased_mass=float(np.mean([r['aliased_native_mass'] for r in rs])),
                                   **{k: float(np.mean([m(r, k) for r in rs])) for k in KEYS})
    d1 = [cells[(a, 1, s)]['d1_reference_max_abs_error'] for a in ARMS for s in SOURCES]
    # Harness equivalence at D = 1 against Protocol 1 cells (folded diagonal encoders).
    harness = {}
    for s in SOURCES:
        p = R.OUT/'cells'/f'data{s}-init{s}.json'
        if p.exists():
            old = json.loads(p.read_text())['arms']['original']
            harness[s] = max(abs(R.get(o, KEYS[k])-R.get(n, KEYS[k])) for o, n in zip(old, cells[('ordinary', 1, s)]['reps']) for k in KEYS)
    pool = [cells[('ordinary', d, s)] for d in DIMS if d > 1 for s in SOURCES]
    rho_next = spearmanr([r['aliased_native_mass'] for r in pool], [m(r, 'next_obs') for r in pool])
    rho_near = spearmanr([r['aliased_native_mass'] for r in pool], [m(r, 'near_shared') for r in pool])
    pairs = [(cells[('ordinary', d, s)], cells[('weighted', d, s)]) for d in DIMS if d > 1 for s in SOURCES]
    pairs = [(o, w) for o, w in pairs if aliased(o) or aliased(w)]
    m5 = sum(w['aliased_native_mass'] < o['aliased_native_mass'] for o, w in pairs)
    preds = dict(M1=dict(max_param_error=max(d1), held=max(d1) == 0.0),
                 M2=dict(count=by['ordinary/d2']['aliased'], held=by['ordinary/d2']['aliased'] >= 5),
                 M3=dict(d2=by['ordinary/d2']['aliased'], d16=by['ordinary/d16']['aliased'], held=by['ordinary/d16']['aliased'] < by['ordinary/d2']['aliased']),
                 M4=dict(rho_next_obs=float(rho_next.statistic), p_next_obs=float(rho_next.pvalue), rho_near=float(rho_near.statistic),
                         p_near=float(rho_near.pvalue), held=bool(rho_next.statistic > .3 and rho_near.statistic > .3)),
                 M5=dict(count=m5, of=len(pairs), held=len(pairs) > 0 and m5 >= .7*len(pairs)))
    d1cal = {s: dict(scalar_backward=cells[('ordinary', 1, s)]['scalar_backward_fraction'], aliased_mass=cells[('ordinary', 1, s)]['aliased_native_mass']) for s in SOURCES}
    checks = dict(cells=len(cells), expected=len(SOURCES)*len(DIMS)*len(ARMS), d1_bit_exact=max(d1) == 0.0,
                  harness_vs_protocol1_max_abs_diff=harness, protocol_unchanged=R.sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA)
    summary = dict(checks=checks, predictions=preds, by_dim=by, d1_calibration=d1cal, cells=rows)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for a, col in zip(ARMS, ('C3', 'C0')):
        for ax, k, lab in zip(axes, ('aliased', 'next_obs', 'near_shared', 'standard_shared'),
                              ('aliased native mass', 'next-obs MSE (native held-out)', 'planning error, near goals', 'planning error, standard goals')):
            vals = [[(cells[(a, d, s)]['aliased_native_mass'] if k == 'aliased' else m(cells[(a, d, s)], k)) for s in SOURCES] for d in DIMS]
            for i, v in enumerate(vals):
                ax.scatter(np.full(10, i)+(-.12 if a == 'ordinary' else .12), v, s=12, color=col, alpha=.6)
            ax.plot(range(len(DIMS)), [np.median(v) for v in vals], color=col, label=a)
            ax.set_xticks(range(len(DIMS))); ax.set_xticklabels([f'D={d}' for d in DIMS]); ax.set_title(lab, fontsize=10)
            if k == 'next_obs':
                ax.set_yscale('log')
    axes[0].legend()
    fig.suptitle('SIGReg on ICM occupancy, by embedding dimension (dots = sources, lines = medians)')
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT/'by_dimension.png', dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(('next_obs', 'near_shared')):
        for d, col in zip(DIMS, plt.cm.viridis(np.linspace(0, .9, len(DIMS)))):
            for a, mk in zip(ARMS, ('o', '^')):
                rs = [cells[(a, d, s)] for s in SOURCES]
                ax[i].scatter([r['aliased_native_mass'] for r in rs], [m(r, k) for r in rs], color=col, marker=mk, s=18, label=f'D={d} {a}')
        ax[i].set_xlabel('aliased native mass'); ax[i].set_ylabel(k)
        if k == 'next_obs':
            ax[i].set_yscale('log')
    ax[1].legend(fontsize=6, ncol=2)
    fig.suptitle('Aliasing versus downstream error (all fits)')
    fig.tight_layout(); fig.savefig(OUT/'aliasing_vs_downstream.png', dpi=150)

    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
    c = B.Config(**trials.load(R.SOURCES/'icm-131.pt')['config'])
    d = torch.exp(torch.linspace(0, R.LOG16, 1025, dtype=torch.float64))
    for row, a in enumerate(ARMS):
        for col, s in enumerate((130, 131, 133, 136, 139)):
            with torch.no_grad():
                f = load_model(trials.load(OUT/'runs'/f'{a}-d2-{s}.pt'))(B.sensor(d, c)).numpy()
            ax = axes[row, col]
            sc = ax.scatter(f[:, 0], f[:, 1], c=np.log(d.numpy()), s=3, cmap='viridis')
            ax.set_title(f'{a}, source {s}: aliased {cells[(a, 2, s)]["aliased_native_mass"]:.2f}', fontsize=9)
            ax.set_aspect('equal', 'datalim')
    fig.colorbar(sc, ax=axes, label='log distance', shrink=.6)
    fig.suptitle('D = 2 embeddings of the distance line (colour = log distance)')
    fig.savefig(OUT/'d2_embeddings.png', dpi=150)
    print(json.dumps(dict(checks=checks, predictions=preds, by_dim={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in by.items()}), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    assert R.sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA, 'PROTOCOL.md changed after lock'
    if not a.report_only:
        specs = [(s, d, arm) for d in sorted(DIMS, reverse=True) for s in SOURCES for arm in ARMS]
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(job, specs):
                pass
    report()


if __name__ == '__main__':
    main()

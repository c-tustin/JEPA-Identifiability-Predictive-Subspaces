"""Protocol 2: downstream value of trained fold repairs, against each baseline's oracle rearrangement.

See REPAIRS_PROTOCOL.md. From the workspace root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/fold-downstream/repairs.py [--workers 10]
    python3 icm-folding/fold-downstream/repairs.py --report-only
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # Protocol 1 harness: Table, next_obs, readout, plan, near_goals, oracle_path
import numpy as np
import torch

B, P, trials = R.B, R.P, R.trials
OUT = HERE/'output/repairs'
OCC = R.ROOT/'explanations/occupancy-folding'
MLP = HERE.parent/'mlp-gradient-study/output'
PROTOCOL_SHA = '3f3101703d5f369d2ae253823fc048ed97c2852f4057e06be29039d6ac9fe6bc'
FOLDED_LATE = (131, 132, 133, 136, 139)

# family -> (baseline path fn, {repair arm: path fn}, sources)
FAMILIES = {
    'scratch': (lambda s: OCC/f'output/runs/native-ordinary-{s}.pt', {
        'weighted_moments': lambda s: OCC/f'output/runs/native-weighted_moments-{s}.pt',
        'weighted_order': lambda s: OCC/f'gradient-followup/output/runs/scratch-g4-weighted_order-{s}.pt',
        'spline_projected': lambda s: OCC/f'hypothesis-guarantee/output/guaranteed/{s}-projected-native.pt',
        'spline_positive': lambda s: OCC/f'hypothesis-guarantee/output/guaranteed/{s}-positive-native.pt'},
        tuple(range(130, 140))),
    'mining': (lambda s: MLP/f'runs/{s}-sigreg-c0-n4000.pt', {
        'global_mining': lambda s: MLP/f'hard-mining/{s}-c100.pt'}, tuple(range(130, 145))),
    'late': (lambda s: OCC/f'gradient-followup/output/runs/repair-g4-ordinary-{s}.pt', {
        'weighted_strong': lambda s: OCC/f'gradient-followup/output/runs/repair-g4-weighted_strong-{s}.pt',
        'weighted_only': lambda s: OCC/f'gradient-followup/output/runs/repair-g4-weighted-{s}.pt'}, FOLDED_LATE),
}
Q = np.linspace(0, R.LOG16, R.GRID)


def source(s):
    path = (R.SOURCES if s < 140 else MLP/'fresh-sources')/f'icm-{s}.pt'
    saved = trials.load(path)
    c = B.Config(**saved['config'])
    data = {k: saved['data'][k] for k in ('x', 'xp', 'action')}
    common_path = R.SOURCES/f'physical-common-{s}.pt'
    if not common_path.exists():
        common_path = OUT/'common'/f'physical-common-{s}.pt'
        if not common_path.exists():
            common_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(B.collect(c, 'random', c.train_episodes, 510000+s), common_path)
    common = {k: v for k, v in trials.load(common_path).items() if k in ('x', 'xp', 'action')}
    return c, data, common, R.sha(path)


def curve(path, c):
    saved = trials.load(path)
    state = saved.get('model', saved.get('encoder'))
    if 'knots' in state:  # spline in apparent height 1/d
        knots = state['knots'].double().numpy()
        if 'z' in state:
            z = state['z'].double().numpy()
        else:
            inc = (0.05+torch.nn.functional.softplus(state['raw'].double())).numpy()*np.diff(knots)
            z = float(state['offset'])+np.concatenate([[0.], np.cumsum(inc)])
        return np.interp(np.exp(-Q), knots, z)
    model = B.Encoder()
    model.load_state_dict(state)
    with torch.no_grad():
        return model.double()(B.sensor(torch.from_numpy(np.exp(Q)), c)).numpy()


def rearrange(y):
    s = 1.0 if np.corrcoef(y, Q)[0, 1] >= 0 else -1.0
    return s*np.sort(s*y)


def fold_stats(y):
    dy = np.diff(y)
    up, down = np.maximum(dy, 0).sum(), np.maximum(-dy, 0).sum()
    backward = float(min(up, down)/max(up+down, 1e-12))
    return dict(backward_fraction=backward, material_fold=backward > 1e-3, any_reversal=backward > 0)


def job(spec):
    torch.set_num_threads(1)
    family, arm, s = spec
    out = OUT/'cells'/f'{family}-{arm}-{s}.json'
    if out.exists():
        return json.loads(out.read_text())
    t0 = time.time()
    c, data, common, source_hash = source(s)
    base_path, repairs, _ = FAMILIES[family]
    if arm == 'baseline':
        path = base_path(s); y = curve(path, c)
    elif arm == 'oracle':
        path = base_path(s); y = rearrange(curve(path, c))
    else:
        path = repairs[arm](s); y = curve(path, c)
    enc = R.Table(y)
    episode = torch.arange(len(data['x'])) % c.train_episodes
    mask = episode < int(.8*c.train_episodes)
    train = {k: v[mask] for k, v in data.items()}
    test = {k: v[~mask] for k, v in data.items()}
    goal_sets = dict(standard=P.goals(c, 10000+s, 64), near=R.near_goals(c, s))
    for v in goal_sets.values():
        R.oracle_path(*v, c)
    bank = P.candidate_bank(10000+s)
    reps = []
    for r in range(R.REPS):
        seed = s+100000*r
        rep = dict(next_obs=R.next_obs(enc, train, test, common, seed), readout=R.readout(enc, train, test, c, seed), planning={})
        for kind, pairs in (('shared', common), ('native', data)):
            pred, mean, scale = P.train_predictor(enc, pairs, common, seed, 1000)
            rep['planning'][kind] = {g: R.plan(enc, pred, mean, scale, c, *goal_sets[g], bank) for g in goal_sets}
        reps.append(rep)
    result = dict(family=family, arm=arm, source=s, checkpoint=str(path.relative_to(R.ROOT)) if R.ROOT in path.parents else str(path),
                  checkpoint_sha256=R.sha(path), source_sha256=source_hash, protocol_sha256=PROTOCOL_SHA,
                  code_std=float(np.std(y)), **fold_stats(y), reps=reps, seconds=time.time()-t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f'{family}-{arm}-{s} done {result["seconds"]:.0f}s backward={result["backward_fraction"]:.4f}', flush=True)
    return result


def specs():
    for family, (_, repairs, sources) in FAMILIES.items():
        for s in sources:
            for arm in ('baseline', 'oracle', *repairs):
                yield (family, arm, s)


KEYS = {'next_obs': ('next_obs', 'native'), 'readout': ('readout', 'native'),
        'near_shared': ('planning', 'shared', 'near', 'error'), 'standard_shared': ('planning', 'shared', 'standard', 'error'),
        'near_native': ('planning', 'native', 'near', 'error'), 'standard_native': ('planning', 'native', 'standard', 'error'),
        'near_success': ('planning', 'shared', 'near', 'success')}


def report():
    import matplotlib.pyplot as plt
    cells = {(r['family'], r['arm'], r['source']): r for r in (json.loads(p.read_text()) for p in (OUT/'cells').glob('*.json'))}
    mean = lambda r, k: float(np.mean([R.get(rep, KEYS[k]) for rep in r['reps']]))
    table, gaps, preds = {}, {}, {}
    for family, (_, repairs, sources) in FAMILIES.items():
        for s in sources:
            base, orc = cells[(family, 'baseline', s)], cells[(family, 'oracle', s)]
            for arm in ('baseline', 'oracle', *repairs):
                r = cells[(family, arm, s)]
                table[f'{family}/{arm}/{s}'] = dict(backward_fraction=r['backward_fraction'], material_fold=r['material_fold'],
                                                    **{k: mean(r, k) for k in KEYS})
            if base['material_fold']:
                for arm in repairs:
                    r = cells[(family, arm, s)]
                    gaps[f'{family}/{arm}/{s}'] = {k: (mean(base, k)-mean(r, k))/(mean(base, k)-mean(orc, k))
                                                   if abs(mean(base, k)-mean(orc, k)) > 1e-12 else None for k in KEYS}
    def g_ok(family, arm, s):
        g = gaps[f'{family}/{arm}/{s}']
        return g['next_obs'] is not None and g['near_shared'] is not None and g['next_obs'] >= .8 and g['near_shared'] >= .8
    def folded(family):
        return [s for s in FAMILIES[family][2] if cells[(family, 'baseline', s)]['material_fold']]
    for arm, key in (('weighted_moments', 'R1'), ('weighted_order', 'R2')):
        n = sum(g_ok('scratch', arm, s) for s in folded('scratch'))
        preds[key] = dict(count=n, of=len(folded('scratch')), held=n >= 4)
    preds['R3'] = {arm: dict(count=(n := sum(g_ok('scratch', arm, s) for s in folded('scratch'))), of=len(folded('scratch')), held=n >= 4)
                   for arm in ('spline_projected', 'spline_positive')}
    eligible = [s for s in folded('mining') if not cells[('mining', 'global_mining', s)]['material_fold']]
    n = sum(g_ok('mining', 'global_mining', s) for s in eligible)
    preds['R4'] = dict(count=n, of=len(eligible), eligible=eligible, held=n == len(eligible))
    strong = sum(g_ok('late', 'weighted_strong', s) for s in FOLDED_LATE)
    weak = sum(all(gaps[f'late/weighted_only/{s}'][k] is not None and gaps[f'late/weighted_only/{s}'][k] < .5 for k in ('next_obs', 'near_shared')) for s in FOLDED_LATE)
    preds['R5'] = dict(strong=strong, weighted_only_below_half=weak, held=strong >= 4 and weak >= 4)
    r6 = {}
    for family, (_, repairs, sources) in FAMILIES.items():
        clean = [s for s in sources if not cells[(family, 'baseline', s)]['material_fold']]
        for arm in repairs:
            if clean:
                r6[f'{family}/{arm}'] = float(np.mean([table[f'{family}/{arm}/{s}']['near_shared']-table[f'{family}/baseline/{s}']['near_shared'] for s in clean]))
    preds['R6'] = dict(changes=r6, held=all(v <= .05 for v in r6.values()))
    medians = {}
    for key in sorted({k.rsplit('/', 1)[0] for k in gaps}):
        vals = [v for k, v in gaps.items() if k.rsplit('/', 1)[0] == key]
        medians[key] = {m: float(np.median([v[m] for v in vals if v[m] is not None])) for m in KEYS}
    checks = dict(cells=len(cells), expected=len(list(specs())),
                  protocol_unchanged=R.sha(HERE/'REPAIRS_PROTOCOL.md') == PROTOCOL_SHA,
                  oracle_unfolded=all(not cells[k]['any_reversal'] for k in cells if k[1] == 'oracle'))
    summary = dict(checks=checks, predictions=preds, median_gap_recovered=medians, gaps=gaps, table=table)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1))

    # Figure: gap recovered per folded source, two measures.
    arms = [('scratch', 'weighted_moments', 'Weighted moments'), ('scratch', 'weighted_order', 'Weighted + order penalty'),
            ('scratch', 'spline_projected', 'Hard order (projected spline)'), ('scratch', 'spline_positive', 'Hard order (positive spline)'),
            ('mining', 'global_mining', 'Global successor mining'), ('late', 'weighted_strong', 'Late: weights + strong order'),
            ('late', 'weighted_only', 'Late: weights only')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, m, title in zip(axes, ('next_obs', 'near_shared', 'standard_shared'),
                            ('Next-observation MSE (native held-out)', 'Planning error, near goals', 'Planning error, standard goals')):
        for i, (f, a, label) in enumerate(arms):
            vals = [gaps[k][m] for k in gaps if k.startswith(f'{f}/{a}/') and gaps[k][m] is not None]
            ax.scatter(np.clip(vals, -1, 1.5), np.full(len(vals), i)+np.linspace(-.15, .15, len(vals)), s=22)
            if vals:
                ax.plot([np.median(vals)]*2, [i-.3, i+.3], color='k', lw=2)
        ax.axvline(0, color='0.6', lw=1); ax.axvline(1, color='0.6', lw=1, ls='--')
        ax.set_xlim(-1.05, 1.55); ax.set_title(title, fontsize=10); ax.set_xlabel('fraction of oracle gap recovered (clipped to [-1, 1.5])')
    axes[0].set_yticks(range(len(arms))); axes[0].set_yticklabels([a[2] for a in arms])
    fig.suptitle('How much of the fold\'s downstream cost each trained repair recovers (1 = as good as removing the fold by oracle)')
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT/'gap_recovered.png', dpi=150)
    print(json.dumps(dict(checks=checks, predictions=preds, median_gap_recovered=medians), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    assert R.sha(HERE/'REPAIRS_PROTOCOL.md') == PROTOCOL_SHA, 'REPAIRS_PROTOCOL.md changed after lock'
    if not a.report_only:
        todo = list(specs())
        for s in range(140, 145):  # create random-policy sets once, before parallel workers
            source(s)
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(job, todo):
                pass
    report()


if __name__ == '__main__':
    main()

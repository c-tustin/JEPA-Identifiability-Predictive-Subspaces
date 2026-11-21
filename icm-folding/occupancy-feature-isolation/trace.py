"""Replay a dose-study SIGReg fit and record the whole code curve f(log d).

The replay uses dose.py's exact initializer, alignment plan, uniform endpoint
plan and update rule; its final state is checked bit-for-bit against the saved
dose checkpoint when one exists. Curves are evaluated on a fixed float64 grid.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dose as D
S, R, torch, np, B = D.S, D.R, D.torch, D.np, D.B
GRID = torch.linspace(0, np.log(16.), 401, dtype=torch.float64)


def curve(m, c):
    mm = B.Encoder().double(); mm.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    with torch.no_grad():
        return mm(B.sensor(GRID.exp(), c)).numpy()


def law(ds, near, far, alpha, cut=2.):
    return torch.cat([ds[near]['x'][ds[near]['d'] < cut], ds[far]['x'][ds[far]['d'] >= cut]])


def replay(near, far, alpha, every=20, steps=4000, x=None, counts=None, setup=None, extra=None):
    """Return curves, final state. `extra(step, model, y_endpoints)` may add a loss term."""
    ds, c, initial, ix, u, mean, std, critic = setup or D.setup()
    if x is None:
        x = law(ds, near, far, alpha)
        counts = (int((ds[near]['d'] < 2).sum()), int((ds[far]['d'] >= 2).sum()))
    ids = S.indices(alpha, counts[0], counts[1], u)
    m = B.Encoder(); m.load_state_dict(initial)
    opt = torch.optim.Adam(m.parameters(), lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']
    curves = [(0, curve(m, c))]
    for step in range(steps):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        code = m(x[ids[step]])
        ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
        alignment = (ya[:512] - ya[512:]).square().mean()
        loss = .9*alignment + .1*reg(code)
        if extra is not None: loss = loss + extra(step, m, code)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % every == 0: curves.append((step + 1, curve(m, c)))
    return curves, m.state_dict()


if __name__ == '__main__':
    near, far, alpha = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    curves, state = replay(near, far, alpha)
    path = HERE/'output/dose/runs'/f'n{near}-f{far}-a{alpha:g}-sigreg.pt'
    same = None
    if path.exists():
        saved = R.trials.load(path)['model']
        same = all(torch.equal(saved[k], state[k]) for k in saved)
    out = HERE/'output/trace'; out.mkdir(parents=True, exist_ok=True)
    np.savez(out/f'n{near}-f{far}-a{alpha:g}.npz', grid=GRID.numpy(), steps=np.array([s for s, _ in curves]),
             curves=np.stack([c for _, c in curves]), replay_matches_saved=same)
    print(dict(near=near, far=far, alpha=alpha, replay_matches_saved=same), flush=True)

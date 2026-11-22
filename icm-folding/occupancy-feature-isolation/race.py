"""Replay geometry-test fits and record the race between monotone reshaping and tail retraction.

For each recorded step: skewness of the code law, spread of the near (heavy) region in standardized units, the
standardized positions of the far branch's median and end, and the backward fraction of the code curve.
Final states are checked bit-for-bit against the saved geometry checkpoints.
"""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import geometry as G

K, R, torch, B = G.K, G.R, G.torch, G.B
GRID = torch.linspace(0, np.log(16.), 401, dtype=torch.float64)
CASES = [('m0.75', 'tanh4', 131), ('m0.75', 'linear', 131), ('m0.875', 'tanh4', 131), ('m0.875', 'linear', 131),
         ('m0.9375', 'tanh4', 131), ('m0.9375', 'linear', 131)]


def stats(m, xd, d, w, c):
    with torch.no_grad():
        y = m(xd).double().numpy(); cv = m(B.sensor(GRID.exp(), c).float()).double().numpy()
    near = d < 2; mu = w @ y; sd = np.sqrt(w @ (y - mu)**2); z = (y - mu)/sd
    if np.corrcoef(np.log(d), y)[0, 1] < 0: z = -z
    wn = w[near]/w[near].sum(); wf = w[~near]/w[~near].sum(); fz = z[~near]
    dy = np.diff(cv); up, dn = np.maximum(dy, 0).sum(), np.maximum(-dy, 0).sum()
    return dict(sd=float(sd), skew=float(w @ z**3), near_spread=float(np.sqrt(wn @ (z[near] - wn @ z[near])**2)),
                near_mean=float(wn @ z[near]), far_median=float(np.interp(.5, np.cumsum(wf), fz)), far_end=float(fz[-1]),
                backward=float(min(up, dn)/max(up + dn, 1e-12)))


def replay(law, mp, init_seed, setup, x, d, every=20):
    ds, c, _, ix, _, _, _, _ = setup
    w = torch.from_numpy(np.load(G.OUT/'laws'/f'{law}.npy'))
    u = torch.rand((4000, 1024), dtype=torch.float64, generator=torch.Generator().manual_seed(9600131))
    cdf = w.cumsum(0); cdf[-1] = 1.; ids = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(w) - 1)
    m = G.MappedEncoder(mp); m.load_state_dict(G.initial(c, init_seed))
    opt = torch.optim.Adam(m.parameters(), lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']; wn = w.numpy()
    rows = [dict(step=0, **stats(m, x, d, wn, c))]
    for step in range(4000):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        code = m(x[ids[step]]); ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
        loss = .9*(ya[:512] - ya[512:]).square().mean() + .1*reg(code)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % every == 0: rows.append(dict(step=step + 1, **stats(m, x, d, wn, c)))
    saved = R.trials.load(G.OUT/'runs'/f'{law}-{mp}-i{init_seed}.pt')['model']
    same = all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    return rows, same


def main():
    torch.set_num_threads(1); setup = K.D.setup(); x, d = K.atoms(setup[0])
    case = CASES[int(sys.argv[1])]; out = G.OUT/'race'; out.mkdir(exist_ok=True)
    rows, same = replay(*case, setup, x, d)
    R.trials.atomic_json(out/f'{case[0]}-{case[1]}-i{case[2]}.json', dict(case=case, replay_matches_saved=same, rows=rows))
    print(case, 'replay matches saved:', same, 'final', {k: round(v, 3) for k, v in rows[-1].items()})


if __name__ == '__main__':
    main()

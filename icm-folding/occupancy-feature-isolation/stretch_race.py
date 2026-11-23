"""Replay stretch-test fits and record when the fold forms versus when the spike stretches (bit-exact check)."""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import stretch as S
import race

K, R, B = S.K, S.R, S.B


def main():
    law, ch, init = sys.argv[1], sys.argv[2], int(sys.argv[3]); torch.set_num_threads(1)
    setup = K.D.setup(); ds, c, _, ix, *_ = setup; x, d = K.atoms(ds)
    w = torch.from_numpy(np.load(S.OUT/'laws'/f'{law}.npy'))
    u = torch.rand((4000, 1024), dtype=torch.float64, generator=torch.Generator().manual_seed(9600131))
    cdf = w.cumsum(0); cdf[-1] = 1.; ids = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(w) - 1)
    m = S.StretchEncoder(ch); m.load_state_dict(S.initial(c, init))
    opt = torch.optim.Adam(m.parameters(), lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']; wn = w.numpy()
    rows = [dict(step=0, **race.stats(m, x, d, wn, c))]
    for step in range(4000):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        code = m(x[ids[step]]); ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
        loss = .9*(ya[:512] - ya[512:]).square().mean() + .1*reg(code)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 20 == 0: rows.append(dict(step=step + 1, **race.stats(m, x, d, wn, c)))
    saved = R.trials.load(S.OUT/'runs'/f'{law}-{ch}-i{init}.pt')['model']
    same = all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    (S.OUT/'race').mkdir(exist_ok=True)
    R.trials.atomic_json(S.OUT/'race'/f'{law}-{ch}-i{init}.json', dict(replay_matches_saved=same, rows=rows))
    print(law, ch, init, 'bit-exact:', same)


if __name__ == '__main__':
    main()

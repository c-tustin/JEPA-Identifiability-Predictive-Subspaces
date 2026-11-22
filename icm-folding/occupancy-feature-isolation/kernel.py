"""Tangent-kernel decoupling at initialization: can the encoder move one distance's code without moving another's?

Decoupling(a, b) = 1 - Θ(a,b)/sqrt(Θ(a,a)Θ(b,b)), with Θ the empirical neural tangent kernel of the scalar encoder.
0 means the two codes can only move together; larger means they can be moved independently.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import geometry as G

K, R, B = G.K, G.R, G.B
PAIRS = {'near cluster (d 1 vs 1.9)': (1., 1.9), 'tail (d 4 vs 16)': (4., 16.), 'tail (d 3 vs 12)': (3., 12.),
         'junction to tail (d 2 vs 6)': (2., 6.)}


def decoupling(input_map, init_seed, c):
    m = G.MappedEncoder(input_map).double(); m.load_state_dict({k: v.double() for k, v in G.initial(c, init_seed).items()})
    out = {}
    for label, (a, b) in PAIRS.items():
        J = []
        for dist in (a, b):
            m.zero_grad(); m(B.sensor(torch.tensor([dist], dtype=torch.float64), c).double()).sum().backward()
            J.append(torch.cat([p.grad.flatten() for p in m.parameters()]).clone())
        out[label] = float(1 - (J[0] @ J[1])/(J[0].norm()*J[1].norm()))
    return out


def main():
    _, c, *_ = K.D.setup()
    rows = [dict(input_map=mp, init=i, **decoupling(mp, i, c)) for mp in G.MAPS for i in G.INITS]
    R.trials.atomic_json(HERE/'output'/'kernel.json', rows)
    for r in rows: print(r)


if __name__ == '__main__':
    main()

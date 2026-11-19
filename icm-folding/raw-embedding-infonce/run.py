"""InfoNCE between fixed raw sensor coordinates and each object's own code."""
import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('raw_infonce_reference', HERE.parent/'mlp-gradient-study/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = HERE/'output'
STEPS = 4000


def critic():
    return nn.Sequential(nn.Linear(1, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 6))


def logits(raw, code, scorer):
    # Rows are embeddings; columns are raw observations. No learned raw branch.
    predicted = scorer(code[:, None])
    fixed = raw.detach()
    distance = (predicted.square().mean(1)[:, None]+fixed.square().mean(1)[None, :]
                -2*(predicted@fixed.T)/fixed.shape[1])
    return -distance/.2


def objective(raw, code, scorer):
    return F.cross_entropy(logits(raw, code, scorer), torch.arange(len(code), device=code.device))


def setup(seed, steps=STEPS):
    c, train, test, source_hash = base.source(seed)
    torch.manual_seed(40000+seed)
    encoder = base.B.Encoder()
    # Separate seed: adding the critic cannot change encoder initialization.
    torch.manual_seed(6100000+seed)
    scorer = critic()
    gen = torch.Generator().manual_seed(2100000+seed)
    plan = torch.randint(len(train['x']), (steps, 512), generator=gen)
    endpoints = torch.cat([train['x'], train['xp']])
    mean = endpoints.mean(0)
    std = endpoints.std(0, unbiased=False).clamp_min(1e-6)
    return c, train, test, source_hash, encoder, scorer, plan, mean, std


def fit(seed, steps=STEPS):
    path = OUT/'runs'/f'{seed}-n{steps}.pt'
    if path.exists():
        return
    c, train, test, source_hash, encoder, scorer, plan, mean, std = setup(seed, steps)
    initial = copy.deepcopy(encoder.state_dict())
    opt = torch.optim.Adam(list(encoder.parameters())+list(scorer.parameters()), lr=.001)
    frames = [dict(step=0, **base.geometry(encoder, c))]
    for step, ix in enumerate(plan):
        phase = max(0, (step-steps/2)/max(1, steps/2-1))
        opt.param_groups[0]['lr'] = .001*.5*(1+base.np.cos(base.np.pi*phase))
        raw = torch.cat([train['x'][ix[:128]], train['xp'][ix[:128]]])
        code = encoder(raw)
        loss = objective((raw-mean)/std, code, scorer)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step+1 in base.FRAMES or step+1 == steps:
            with torch.no_grad():
                scores = logits((raw-mean)/std, encoder(raw), scorer)
                accuracy = (scores.argmax(1)==torch.arange(len(raw))).float().mean().item()
            frames.append(dict(step=step+1, loss=loss.item(), candidate_accuracy=accuracy,
                               **base.geometry(encoder, c)))
    result = dict(seed=seed, steps=steps, model=encoder.state_dict(), initial=initial,
                  critic=scorer.state_dict(), raw_mean=mean, raw_std=std,
                  source_hash=source_hash, plan_hash=base.sha(plan), config=vars(c),
                  code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_hash=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
                  frames=frames, geometry=base.geometry(encoder, c, 10001),
                  prediction=base.downstream(encoder, train, test, c, seed))
    base.trials.save_torch(path, result)
    print(seed, 'fold', result['geometry']['folded'], 'loss', frames[-1]['loss'],
          'heldout', result['prediction']['native_mse'], flush=True)


def checks():
    torch.manual_seed(54)
    scorer = critic().double()
    raw = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
    code = torch.randn(7, dtype=torch.float64, requires_grad=True)
    score = logits(raw, code, scorer)
    explicit = torch.stack([torch.stack([-(scorer(z[None])-x).square().mean()/.2
                                        for x in raw.detach()]) for z in code])
    torch.testing.assert_close(score, explicit)
    assert torch.autograd.gradcheck(lambda z: objective(raw, z, scorer), (code,))
    perm = torch.randperm(len(code))
    torch.testing.assert_close(objective(raw, code, scorer), objective(raw[perm], code[perm], scorer))
    # Construct known compatible pairs: shuffling only one side must hurt.
    identity = nn.Identity()
    known = torch.linspace(-2, 2, 7, dtype=torch.float64)
    assert objective(known[:, None], known, identity) < objective(known.flip(0)[:, None], known, identity)
    objective(raw, code, scorer).backward()
    assert raw.grad is None and code.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in scorer.parameters())
    encoder = base.B.Encoder()
    objective(raw.float().detach(), encoder(raw.float().detach()), critic()).backward()
    assert sum(p.grad.abs().sum() for p in encoder.parameters())>0
    OUT.mkdir(exist_ok=True)
    base.trials.atomic_json(OUT/'formula-checks.json', dict(explicit_score=True, code_gradcheck=True,
        matched_permutation=True, mismatch_sensitivity=True, raw_branch_detached=True,
        critic_and_encoder_gradients=True))
    print('Raw/code loss and gradient checks passed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(130, 145)))
    p.add_argument('--steps', type=int, default=STEPS)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.check:
        checks()
    else:
        for seed in args.seeds:
            fit(seed, args.steps)

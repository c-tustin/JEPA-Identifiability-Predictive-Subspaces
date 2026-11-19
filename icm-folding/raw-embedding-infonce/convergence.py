"""Recover dense histories by replay; require exact saved encoder/critic/decoder weights."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run as original

OUT = original.OUT/'convergence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert torch.equal(value, expected[key]), key


def replay(seed):
    OUT.mkdir(exist_ok=True)
    path = OUT/f'{seed}.npz'
    audit_path = OUT/f'{seed}.json'
    if path.exists() and audit_path.exists():
        import json
        audit = json.loads(audit_path.read_text())
        assert audit['replay_code_sha256'] == sha(__file__)
        assert audit['array_sha256'] == sha(path)
        return
    c, train, test, source_hash, encoder, scorer, plan, mean, std = original.setup(seed)
    saved_path = original.OUT/'runs'/f'{seed}-n4000.pt'
    saved = original.base.trials.load(saved_path)
    assert saved['source_hash'] == source_hash
    assert saved['plan_hash'] == original.base.sha(plan)
    assert saved['code_hash'] == sha(original.HERE/'run.py')
    require_exact(encoder.state_dict(), saved['initial'])
    opt = torch.optim.Adam(list(encoder.parameters())+list(scorer.parameters()), lr=.001)
    # Fixed, independently chosen endpoint candidate sets; never used by optimizer.
    pools = {}
    for name, data, offset in [('train', train, 0), ('heldout', test, 1)]:
        gen = torch.Generator().manual_seed(7400000+10*seed+offset)
        ix = torch.randperm(len(data['x']), generator=gen)[:128]
        raw = torch.cat([data['x'][ix], data['xp'][ix]])
        pools[name] = (raw, (raw-mean)/std)
    history = {k:[] for k in ('encoder_updates','train_nce','heldout_nce','train_raw_mse','heldout_raw_mse')}

    @torch.no_grad()
    def measure(step):
        history['encoder_updates'].append(step)
        for name, (raw, fixed) in pools.items():
            z = encoder(raw)
            history[name+'_nce'].append(original.objective(fixed,z,scorer).item())
            # A diagnostic in standardized raw-sensor units, not an added MSE loss.
            history[name+'_raw_mse'].append(F.mse_loss(scorer(z[:,None]),fixed).item())

    measure(0)
    losses, rates = [], []
    recorded = {f['step']:f for f in saved['frames'] if f['step']}
    for step, ix in enumerate(plan):
        phase = max(0, (step-original.STEPS/2)/max(1, original.STEPS/2-1))
        lr = .001*.5*(1+np.cos(np.pi*phase))
        opt.param_groups[0]['lr'] = lr
        raw = torch.cat([train['x'][ix[:128]], train['xp'][ix[:128]]])
        loss = original.objective((raw-mean)/std,encoder(raw),scorer)
        losses.append(loss.item()); rates.append(lr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step+1 in recorded:
            assert loss.item() == recorded[step+1]['loss'], (seed,step)
        if (step+1)%25 == 0:
            measure(step+1)
    require_exact(encoder.state_dict(), saved['model'])
    require_exact(scorer.state_dict(), saved['critic'])

    # Replay the separate downstream fit using the frozen final representation.
    _, target_mean, target_std = original.base.standardized_successors(train)
    with torch.no_grad():
        code = encoder(train['x'])
        center = code.mean()
        scale = code.std(unbiased=False).clamp_min(.05)
        def features(data):
            z = ((encoder(data['x'])-center)/scale)[:,None]
            return torch.cat([z,F.one_hot(data['action'],3).float()],1)
        datasets = {}
        for name, data in [('train',train),('heldout',test),('coverage',original.base.common_data(c))]:
            datasets[name] = (features(data),(torch.tanh(4*data['xp'])-target_mean)/target_std)
    torch.manual_seed(5100000+seed)
    decoder = nn.Sequential(nn.Linear(4,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,6))
    decoder_opt = torch.optim.Adam(decoder.parameters(),lr=.001)
    gen = torch.Generator().manual_seed(5200000+seed)
    decoder_history = {k:[] for k in ('decoder_updates','decoder_train_mse','decoder_heldout_mse','decoder_coverage_mse')}

    @torch.no_grad()
    def measure_decoder(step):
        decoder_history['decoder_updates'].append(step)
        for name,(xx,yy) in datasets.items():
            decoder_history['decoder_'+name+'_mse'].append(F.mse_loss(decoder(xx),yy).item())

    measure_decoder(0)
    xx, yy = datasets['train']
    for step in range(1500):
        ix = torch.randint(len(xx),(256,),generator=gen)
        loss = F.mse_loss(decoder(xx[ix]),yy[ix])
        decoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        decoder_opt.step()
        if (step+1)%50 == 0:
            measure_decoder(step+1)
    require_exact(decoder.state_dict(), saved['prediction']['decoder'])
    for name, old_name in [('train','train'),('heldout','native'),('coverage','common')]:
        assert decoder_history['decoder_'+name+'_mse'][-1] == saved['prediction'][old_name+'_mse']
    np.savez_compressed(path,seed=seed,**history,**decoder_history,update_loss=losses,learning_rate=rates)
    original.base.trials.atomic_json(audit_path,dict(seed=seed,date='2026-11-19',
        replay_code_sha256=sha(__file__),training_code_sha256=sha(original.HERE/'run.py'),
        original_checkpoint_sha256=sha(saved_path),source_sha256=source_hash,
        batch_plan_sha256=saved['plan_hash'],array_sha256=sha(path),
        encoder_bit_exact=True,critic_bit_exact=True,decoder_bit_exact=True,
        original_saved_losses_exact=True,final_full_dataset_errors_exact=True,
        encoder_evaluation_interval=25,decoder_evaluation_interval=50,
        encoder_fixed_candidates_per_split=256,decoder_evaluation='whole datasets'))
    print(seed,'encoder, critic and decoder replay bit-for-bit; dense history saved',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(130,145)))
    args=p.parse_args()
    torch.set_num_threads(1)
    for seed in args.seeds:
        replay(seed)

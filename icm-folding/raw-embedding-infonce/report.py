"""Independent checkpoint reload, dense fold checks, and descriptive comparisons."""
import hashlib
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import HERE, OUT, base, setup, critic, objective, STEPS


def verify_prediction(model, record, train, test, c):
    _, mean, std = base.standardized_successors(train)
    decoder = torch.nn.Sequential(torch.nn.Linear(4,64), torch.nn.ELU(), torch.nn.Linear(64,64),
                                  torch.nn.ELU(), torch.nn.Linear(64,6))
    decoder.load_state_dict(record['prediction']['decoder'])
    with torch.no_grad():
        codes = model(train['x'])
        center, scale = codes.mean(), codes.std(unbiased=False).clamp_min(.05)
        for name, data in [('native', test), ('common', base.common_data(c))]:
            inputs = torch.cat([((model(data['x'])-center)/scale)[:,None],
                                torch.nn.functional.one_hot(data['action'],3).float()],1)
            target = (torch.tanh(4*data['xp'])-mean)/std
            mse = (decoder(inputs)-target).square().mean().item()
            assert abs(mse-record['prediction'][name+'_mse']) < 1e-8


def main():
    torch.set_num_threads(1)
    rows, checks, curves = [], [], {}
    for seed in range(130,145):
        path = OUT/'runs'/f'{seed}-n{STEPS}.pt'
        r = base.trials.load(path)
        c, train, test, source_hash, model, scorer, plan, mean, std = setup(seed)
        assert r['source_hash']==source_hash and r['plan_hash']==base.sha(plan)
        assert r['code_hash']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
        assert r['protocol_hash']==hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v, r['initial'][k], rtol=0, atol=0)
        torch.testing.assert_close(mean, r['raw_mean'], rtol=0, atol=0)
        torch.testing.assert_close(std, r['raw_std'], rtol=0, atol=0)
        original = base.trials.load(base.OUT/'runs'/f'{seed}-infonce-c0-n4000.pt')
        assert r['plan_hash']==original['plan_hash'] and r['source_hash']==original['source_hash']
        for k,v in r['initial'].items():
            torch.testing.assert_close(v, original['initial'][k], rtol=0, atol=0)
        model.load_state_dict(r['model'])
        scorer.load_state_dict(r['critic'])
        g = base.geometry(model,c,20001)
        assert g['folded']==r['geometry']['folded']
        assert abs(g['backward_fraction']-r['geometry']['backward_fraction'])<1e-6
        assert abs(g['std']-r['geometry']['std']) < .005*max(g['std'], 1e-6)
        assert g['std']>1e-5, 'Collapsed encoder must not be called a successful no-fold result'
        verify_prediction(model,r,train,test,c)
        # Evaluate identical-object retrieval loss on a fixed, held-out endpoint batch.
        with torch.no_grad():
            gen = torch.Generator().manual_seed(7300000+seed)
            ix = torch.randperm(len(test['x']),generator=gen)[:128]
            raw = torch.cat([test['x'][ix],test['xp'][ix]])
            heldout_loss = objective((raw-mean)/std,model(raw),scorer).item()
        row = dict(seed=seed,method='raw-own-code',
            **{k:v for k,v in g.items() if k not in ('q','y','slope')},
            native_mse=r['prediction']['native_mse'],common_mse=r['prediction']['common_mse'],
            heldout_infonce=heldout_loss,final_training_infonce=r['frames'][-1]['loss'])
        rows.append(row)
        curves[seed]=g
        checks.append(dict(seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            dense_float64_points=20001,source_and_plan_matched=True,initialization_matched=True,
            training_raw_statistics_verified=True,downstream_mse_recomputed=True))
        for method in ('sigreg','vicreg','infonce'):
            ref=base.trials.load(base.OUT/'runs'/f'{seed}-{method}-c0-n4000.pt')
            rows.append(dict(seed=seed,method=method,
                **{k:v for k,v in ref['geometry'].items() if k not in ('q','y','slope')},
                native_mse=ref['prediction']['native_mse'],common_mse=ref['prediction']['common_mse']))
    summaries=[]
    lines=['# Raw observation–own embedding InfoNCE results','',
           'Completed November 18, 2026. All 15 new fits use 4,000 updates from initialization.',
           'Positive pairs are a raw observation and its own scalar code, with 255 other raw observations as negatives.',
           'The learned quadratic critic, fixed raw standardization and candidate count are specified in [the protocol](../PROTOCOL.md).','',
           '| Sources | Objective | Folds | Material folds | Mean held-out MSE | Mean coverage MSE |',
           '|---|---|---:|---:|---:|---:|']
    for cohort,seeds in [('Historical',range(130,140)),('Reused follow-up',range(140,145))]:
        for method in ('raw-own-code','sigreg','vicreg','infonce'):
            group=[r for r in rows if r['seed'] in seeds and r['method']==method]
            summary=dict(cohort=cohort,method=method,n=len(group),folded=sum(r['folded'] for r in group),
                         material_folded=sum(r['material_fold'] for r in group),
                         **{k:float(np.mean([r[k] for r in group])) for k in ('native_mse','common_mse','std','affine_r2')})
            summaries.append(summary)
            label={'raw-own-code':'Raw observation / own code','sigreg':'SIGReg','vicreg':'VICReg (scalar)',
                   'infonce':'Temporal InfoNCE (distance)'}[method]
            lines.append(f"| {cohort} | {label} | {summary['folded']}/{len(group)} | {summary['material_folded']}/{len(group)} | {summary['native_mse']:.6f} | {summary['common_mse']:.6f} |")
    own=[r for r in rows if r['method']=='raw-own-code']
    lines += ['',f"Minimum final raw/code encoder standard deviation: {min(r['std'] for r in own):.6f}.",
        f"Mean held-out raw/code InfoNCE: {np.mean([r['heldout_infonce'] for r in own]):.6f}; uniform candidate baseline log(256) = {np.log(256):.6f}.",'',
        'MSE is next-observation prediction from a frozen scalar code and action, using a separately trained decoder and targets whitened on training data. Coverage MSE uses the original uniform-distance stress set, not another native held-out draw.',
        'Comparisons are descriptive: the raw/code run changes the critic, pairing, candidate count, score units and computation. Historical baselines are reloaded from the earlier verified study; they are not rerun or overwritten. The follow-up sources were already available, so this is not fresh prospective validation.',
        'A 20,001-point derivative check is empirical and can miss inter-grid folds. No injectivity theorem, hyperparameter robustness, generalization to other renderers, or late repair is established. Identical observations can be negatives, so candidate-index accuracy has an unavoidable ambiguity.','',
        '![All final encoder curves](curves.png)','',
        '| Seed | Fold | Backward fraction | Std | Held-out MSE | Raw/code held-out loss |',
        '|---|---:|---:|---:|---:|---:|']
    for r in own:
        lines.append(f"| {r['seed']} | {int(r['folded'])} | {r['backward_fraction']:.6g} | {r['std']:.4f} | {r['native_mse']:.6f} | {r['heldout_infonce']:.4f} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    base.trials.atomic_json(OUT/'summary.json',dict(summary=summaries,runs=rows))
    base.trials.atomic_json(OUT/'verification.json',dict(count=len(checks),runs=checks,
        formula_checks=json.loads((OUT/'formula-checks.json').read_text()),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (HERE/'run.py',HERE/'report.py',HERE/'PROTOCOL.md')}))
    fig,axes=plt.subplots(3,5,figsize=(15,8))
    for seed,ax in zip(range(130,145),axes.flat):
        g=curves[seed]
        ax.plot(np.exp(g['q']),g['y'],color='#147b79' if not g['folded'] else '#b94348')
        ax.set_title(f"{seed}: {'fold' if g['folded'] else 'no detected fold'}",fontsize=10)
        ax.set_xlabel('Distance (evaluation only)')
        ax.set_ylabel('Scalar code')
    fig.suptitle('Raw observation / own embedding InfoNCE — final encoders')
    fig.tight_layout()
    fig.savefig(OUT/'curves.png',dpi=150)
    plt.close(fig)
    print('\n'.join(lines[:17]),flush=True)


if __name__=='__main__':
    main()

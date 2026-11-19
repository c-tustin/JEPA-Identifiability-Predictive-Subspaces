"""Reload and audit all crossed fits, then summarize finite-grid geometry."""
import copy
import json
from pathlib import Path
import run as R
import numpy as np
import torch


def main():
    torch.set_num_threads(1)
    rows=[]; files={}; initial={}; plans={}; replay={}
    for path in sorted((R.OUT/'runs').glob('*-n4000.pt')):
        s=R.trials.load(path); i=s['identity']; d,j=i['data_seed'],i['init_seed']
        assert i['code_hash']==R.digest((R.HERE/'run.py').read_bytes())
        assert i['protocol_hash']==R.digest((R.HERE/'PROTOCOL.md').read_bytes())
        assert i['initial_hash']==R.state_hash(s['initial'])
        assert initial.setdefault(j,i['initial_hash'])==i['initial_hash']
        assert plans.setdefault(d,(i['alignment_plan_hash'],i['regularizer_plan_hash']))==(i['alignment_plan_hash'],i['regularizer_plan_hash'])
        c=R.B.Config(**s['config'])
        model=R.B.Encoder();model.load_state_dict(s['model'])
        g=R.geometry(model,c,20001)
        assert g==s['geometry']
        source_path=R.LEGACY/'output/sources'/f'icm-{d}.pt'
        assert R.digest(source_path.read_bytes())==i['source_hash']
        source=R.trials.load(source_path)['data']
        ds=1/torch.cat([source['x'],source['xp']])[:,1]
        roots_in_support=[x for x in g['roots'] if float(ds.min())<=x<=float(ds.max())]
        m=copy.deepcopy(model);m.load_state_dict(s['initial'])
        initial_geometry=R.geometry(m,c,20001)
        if d==j:
            original=R.trials.load(R.LEGACY/'output/encoders'/f'icm-{d}.pt')['encoder']
            error=max(float((v-original[k]).abs().max()) for k,v in s['model'].items())
            assert error==0
            replay[d]=error
        rows.append(dict(data_seed=d,init_seed=j,**g,roots_in_source_support=roots_in_support,
                         initial_folded=initial_geometry['folded'],near_mass=float((ds<2).float().mean())))
        files[path.name]=R.digest(path.read_bytes())
    assert len(rows)==100 and len(replay)==10
    # Regenerate the paired streams independently of checkpoint metadata.
    for seed in R.SEEDS:
        m,p=R.trials.study.make_plan(c,seed,seed,'icm')
        ri=torch.randint(len(source['x']),p['indices'].shape,generator=torch.Generator().manual_seed(1900000+seed))
        assert initial[seed]==R.state_hash(m.state_dict())
        assert plans[seed]==(R.digest(p['indices'].numpy().tobytes()),R.digest(ri.numpy().tobytes()))
    matrix=np.array([[next(r for r in rows if r['data_seed']==d and r['init_seed']==j)['folded'] for j in R.SEEDS] for d in R.SEEDS])
    material=np.array([[next(r for r in rows if r['data_seed']==d and r['init_seed']==j)['material_fold'] for j in R.SEEDS] for d in R.SEEDS])
    summary=dict(completed=100,folds=int(matrix.sum()),material_folds=int(material.sum()),
                 by_dataset={d:int(matrix[k].sum()) for k,d in enumerate(R.SEEDS)},
                 by_initialization={j:int(matrix[:,k].sum()) for k,j in enumerate(R.SEEDS)},
                 source_support_folds=sum(bool(r['roots_in_source_support']) for r in rows),rows=rows)
    R.trials.atomic_json(R.OUT/'audited-summary.json',summary)
    R.trials.atomic_json(R.OUT/'verification.json',dict(passed=True,checkpoints=100,dense_reload_exact=True,
        regenerated_initialization_and_plan_hashes=True,diagonal_parameter_errors=replay,
        checkpoint_hashes=files,report_hash=R.digest(Path(__file__).read_bytes())))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.3))
    for ax,mat,title in zip(axes,[matrix,material],['Any detected final turn','Material final turn (backward fraction > 0.001)']):
        ax.imshow(mat,vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=range(10),xticklabels=R.SEEDS,yticks=range(10),yticklabels=R.SEEDS,
               xlabel='Encoder initialization seed',ylabel='Dataset + fixed batch-plan seed',title=title)
        for y in range(10):
            for x in range(10):ax.text(x,y,'F' if mat[y,x] else '·',ha='center',va='center',color='white' if mat[y,x] else 'black')
    fig.suptitle('Historical ICM data × initialization: 100 completed fits; all 10 diagonals bit-exact')
    fig.tight_layout();fig.savefig(R.OUT/'crossing.png',dpi=180);plt.close(fig)
    lines=['# Completed dataset × initialization crossing','',
           f'All 100 fits completed; all ten diagonal models reproduce historical parameters bit-for-bit. Independent final 20,001-point float64 reload checks pass for every cell. Detected final turns: {matrix.sum()}/100; material turns: {material.sum()}/100; turns inside source support: {summary["source_support_folds"]}/100.','',
           '| Dataset | Final folds / 10 | Material folds / 10 | Endpoint mass below 2 |','| --- | ---: | ---: | ---: |']
    for k,d in enumerate(R.SEEDS):
        near=next(r['near_mass'] for r in rows if r['data_seed']==d)
        lines.append(f'| {d} | {matrix[k].sum()} | {material[k].sum()} | {near:.3%} |')
    lines+=['','Datasets 131/132/133/136/139 fold for every tested initialization; 130/138 for none. Initialization changes outcomes for 134/135/137, including small turns. The dataset/sampling condition strongly controls final branch choice in this historical panel. It does not uniquely identify occupancy: rows also vary temporal pairs and minibatch plans. Initial seed 136 has a sampled turn, so final folds must not all be called newly created folds.','',
            'Reused sources, ten initializations, one plan per dataset, finite training budget and finite derivative grids limit this inference. There is no analytic injectivity claim. See [audit](verification.json), [audited measurements](audited-summary.json), and [matrix](crossing.png).','']
    (R.OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()

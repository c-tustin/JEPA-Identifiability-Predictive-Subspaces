"""Audit completed source interventions and render their full outcome matrix."""
import json
import platform
from pathlib import Path
import run as T
import torch
import numpy as np

C=T.C


def main():
    torch.set_num_threads(1)
    rows=[];hashes={};iid=[];ablation=[]
    sources={s:C.trials.load(C.LEGACY/'output/sources'/f'icm-{s}.pt') for s in C.SEEDS}
    source_hashes={s:C.digest((C.LEGACY/'output/sources'/f'icm-{s}.pt').read_bytes()) for s in C.SEEDS}
    c=C.B.Config(**sources[131]['config'])
    renderer_errors={}
    for seed,source in sources.items():
        assert source['config']==sources[131]['config']
        x=torch.cat([source['data']['x'],source['data']['xp']])
        distances=1/x[:,1]
        error=float((C.B.sensor(distances,c)-x).abs().max())
        assert error<1e-6
        renderer_errors[seed]=error
    initial,plan=C.trials.study.make_plan(c,131,131,'icm')
    ih=C.state_hash(initial.state_dict());ah=C.digest(plan['indices'].numpy().tobytes())
    ri=torch.randint(len(sources[131]['data']['x']),plan['indices'].shape,generator=torch.Generator().manual_seed(1900131))
    rh=C.digest(ri.numpy().tobytes())
    finite_difference_errors=[]
    for group,paths in [('cross',sorted((T.OUT/'runs').glob('*.pt'))),('iid',sorted((T.OUT/'iid-endpoints/runs').glob('*.pt')))]:
        assert len(paths)==(100 if group=='cross' else 30),(group,len(paths))
        for path in paths:
            s=C.trials.load(path);i=s['identity']
            assert i['initial_hash']==ih==C.state_hash(s['initial'])
            assert i['alignment_plan_hash']==ah
            if group=='cross':
                assert i['regularizer_plan_hash']==rh
                assert i['alignment_source_hash']==source_hashes[i['alignment_source']]
                code,protocol='run.py','PROTOCOL.md'
            else:
                assert i['alignment_source_hash']==source_hashes[131]
                ids=torch.randint(2*len(sources[131]['data']['x']),(4000,1024),generator=torch.Generator().manual_seed(i['plan_seed']))
                assert i['endpoint_plan_hash']==C.digest(ids.numpy().tobytes())
                code,protocol='iid_endpoints.py','IID_PROTOCOL.md'
            assert i['regularizer_source_hash']==source_hashes[i['regularizer_source']]
            assert i['code_hash']==C.digest((T.HERE/code).read_bytes())
            assert i['protocol_hash']==C.digest((T.HERE/protocol).read_bytes())
            model=C.B.Encoder();model.load_state_dict(s['model'])
            g=C.geometry(model,c,20001)
            assert all(s['geometry'][k]==v for k,v in g.items())
            # Check autograd slopes by central differences at a separate set
            # of locations, independent of the geometry reporting function.
            model=model.double();q=torch.linspace(.01,np.log(16)-.01,31,dtype=torch.float64,requires_grad=True)
            y=model(C.B.sensor(q.exp(),c));slope=torch.autograd.grad(y.sum(),q)[0]
            eps=1e-5
            with torch.no_grad():fd=(model(C.B.sensor((q+eps).exp(),c))-model(C.B.sensor((q-eps).exp(),c)))/(2*eps)
            error=float((slope-fd).abs().max());assert error<1e-6
            finite_difference_errors.append(error)
            data=sources[i['regularizer_source']]['data'];ds=1/torch.cat([data['x'],data['xp']])[:,1]
            within=[d for d in g['roots'] if float(ds.min())<=d<=float(ds.max())]
            row=dict(**i,**g,roots_in_regularizer_support=within)
            (rows if group=='cross' else iid).append(row)
            hashes[str(path.relative_to(T.HERE))]=C.digest(path.read_bytes())
    paths=sorted((T.OUT/'no-alignment/runs').glob('*.pt'))
    assert len(paths)==10
    for path in paths:
        s=C.trials.load(path);i=s['identity']
        assert i['initial_hash']==ih==C.state_hash(s['initial'])
        assert i['plan_hash']==rh
        assert i['source_hash']==source_hashes[i['regularizer_source']]
        assert i['code_hash']==C.digest((T.HERE/'alignment_ablation.py').read_bytes())
        assert i['protocol_hash']==C.digest((T.HERE/'ABLATION_PROTOCOL.md').read_bytes())
        m=C.B.Encoder();m.load_state_dict(s['model']);g=C.geometry(m,c,20001)
        assert g==s['geometry'];ablation.append(dict(**i,**g))
        hashes[str(path.relative_to(T.HERE))]=C.digest(path.read_bytes())
    get=lambda a,r:next(s for s in rows if s['alignment_source']==a and s['regularizer_source']==r)
    mat=np.array([[get(a,r)['folded'] for r in C.SEEDS] for a in C.SEEDS])
    material=np.array([[get(a,r)['material_fold'] for r in C.SEEDS] for a in C.SEEDS])
    imat=np.array([[next(s for s in iid if s['plan_seed']==p and s['regularizer_source']==r)['folded'] for r in C.SEEDS] for p in [2900131,2900132,2900133]])
    baseline=np.array([get(131,r)['folded'] for r in C.SEEDS])
    amat=np.array([[next(s for s in ablation if s['regularizer_source']==r)['folded'] for r in C.SEEDS]])
    disagreements=int((mat!=baseline[None,:]).sum())
    result=dict(completed=100,iid_completed=30,folds=int(mat.sum()),material_folds=int(material.sum()),
        alignment_changes_relative_to_row131=disagreements,
        folds_by_regularizer_source={s:int(mat[:,k].sum()) for k,s in enumerate(C.SEEDS)},
        folds_by_alignment_source={s:int(mat[k].sum()) for k,s in enumerate(C.SEEDS)},
        iid_folds_by_regularizer_source={s:int(imat[:,k].sum()) for k,s in enumerate(C.SEEDS)},
        iid_disagreements_with_paired_row131=int((imat!=baseline[None,:]).sum()),
        no_alignment_fold_count=int(amat.sum()),no_alignment_disagreements=int((amat!=baseline[None,:]).sum()),
        rows=rows,iid_rows=iid,no_alignment_rows=ablation)
    C.trials.atomic_json(T.OUT/'audited-summary.json',result)
    C.trials.atomic_json(T.OUT/'verification.json',dict(passed=True,checkpoints=140,dense_reload_exact=True,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform()),
        initial_weights_and_index_plans_regenerated=True,source_hashes_verified=True,
        common_renderer_max_errors=renderer_errors,
        max_slope_finite_difference_error=max(finite_difference_errors),checkpoint_hashes=hashes,
        report_hash=C.digest(Path(__file__).read_bytes())))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'height_ratios':[3,1]})
    for ax,m,title,ys in [(axes[0,0],mat,'Any detected final turn',C.SEEDS),
                          (axes[0,1],material,'Material final turn',C.SEEDS),
                          (axes[1,0],imat,'Independent endpoint draws: three new plans',[1,2,3]),
                          (axes[1,1],amat,'Temporal alignment removed',['None'])]:
        ax.imshow(m,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=range(10),xticklabels=C.SEEDS,yticks=range(len(ys)),yticklabels=ys,
               xlabel='SIGReg source dataset',ylabel='Alignment source' if len(ys)==10 else 'IID batch plan',title=title)
        for y in range(len(ys)):
            for x in range(10):ax.text(x,y,'F' if m[y,x] else '·',ha='center',va='center',color='white' if m[y,x] else 'black')
    axes[1,1].set_ylabel('Alignment')
    fig.suptitle('Which loss carries the folding dataset effect? Common initialization: 131',fontsize=15)
    fig.tight_layout();fig.savefig(T.OUT/'source_crossing.png',dpi=180);plt.close(fig)
    lines=['# Completed loss-source intervention','',
        f'All 100 source-crossing, 30 independent-endpoint and ten alignment-removal fits completed and passed independent checkpoint reload and identity checks. Derivative finite differences also passed for all 130 crossing/IID fits. Full crossing: {mat.sum()}/100 final turns, {material.sum()}/100 material turns. The (131,131) cell reproduces historical encoder 131 exactly.','',
        '| SIGReg dataset | Final folds / 10 alignment datasets | IID endpoint folds / 3 plans | No alignment |','| --- | ---: | ---: | --- |']
    for k,s in enumerate(C.SEEDS):lines.append(f'| {s} | {mat[:,k].sum()} | {imat[:,k].sum()} | {"Fold" if amat[0,k] else "No detected turn"} |')
    lines+=['',f'Changing the alignment source away from 131 changes final binary fold status in {disagreements}/100 cells. IID endpoint outcomes differ from the paired row-131 outcome in {result["iid_disagreements_with_paired_row131"]}/30 fits. These are descriptive counts in a fixed historical panel, not independent-source statistical estimates.','',
        'The crossed design intervenes separately on the data supplied to each loss. It identifies the regularizer-data pathway if outcomes follow columns while alignment changes leave their status unchanged. The iid follow-up preserves each empirical endpoint distribution in expectation while removing paired-endpoint dependence; persistent folds there do not require paired endpoints. Alignment can still affect turn position, scale, timing and the training path. These experiments do not remove alignment from the objective.','',
        f'The separate alignment-removal ablation does remove that term: {amat.sum()}/10 final folds, with {result["no_alignment_disagreements"]}/10 status differences from the paired row-131 baseline. Any fold in this ablation demonstrates that temporal alignment is not necessary for folding in that tested configuration. This is 0.1 SIGReg-only training with the same data, initialization and regularizer plan.','',
        'Endpoint distribution means the full sensor distribution with its multiplicities, not only mass below a threshold, coarse histogram, or number of modes. Initialization, input transform, objective and optimizer also participate in fold development. One common initialization in this crossing, three iid plans, reused sources, finite budget and derivative grids limit generalization; no all-input guarantee or unique actor-learning cause is proved.','',
        '[Full matrix](source_crossing.png) · [Audited measurements](audited-summary.json) · [Verification](verification.json) · [Main protocol](../PROTOCOL.md) · [IID protocol](../IID_PROTOCOL.md) · [Alignment-removal protocol](../ABLATION_PROTOCOL.md)','']
    (T.OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','iid_rows','no_alignment_rows']},indent=2))


if __name__=='__main__':main()

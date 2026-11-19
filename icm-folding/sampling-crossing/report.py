"""Independently audit the complete matrix and describe crossed factor effects."""
import argparse
import itertools
import json
from pathlib import Path
import run as S
R,np,torch=S.R,S.np,S.torch


def progress():
    rows=[json.loads(p.read_text()) for p in sorted((S.OUT/'cells').glob('*-n4000.json'))]
    R.trials.atomic_json(S.OUT/'progress.json',dict(completed=len(rows),planned=500,
        status='training_complete_audit_pending' if len(rows)==500 else 'in_progress',
        folds=sum(r['folded'] for r in rows),material_folds=sum(r['material_fold'] for r in rows)))
    return rows


def decomposition(x):
    x=x.astype(float);mu=x.mean();a=x.mean((1,2))-mu;b=x.mean((0,2))-mu;c=x.mean((0,1))-mu
    ab=x.mean(2)-mu-a[:,None]-b[None,:]
    ac=x.mean(1)-mu-a[:,None]-c[None,:]
    bc=x.mean(0)-mu-b[:,None]-c[None,:]
    terms=dict(dataset=a[:,None,None],initialization=b[None,:,None],sampling=c[None,None,:],
        dataset_initialization=ab[:,:,None],dataset_sampling=ac[:,None,:],initialization_sampling=bc[None,:,:])
    terms['three_way']=x-mu-sum(terms.values())
    ss={k:float(np.square(np.broadcast_to(v,x.shape)).sum()) for k,v in terms.items()}
    total=float(np.square(x-mu).sum());assert abs(sum(ss.values())-total)<1e-8
    return dict(total_sum_squares=total,sum_squares=ss,
        fractions={k:(v/total if total else 0.) for k,v in ss.items()})


def analyze(rows,key):
    x=np.array([[[next(r for r in rows if (r['data_seed'],r['init_seed'],r['plan'])==(d,j,p))[key]
        for p in S.PLANS] for j in S.INIT] for d in S.DATA],dtype=bool)
    changed=np.any(x!=x[:,:,:1],axis=2)
    sampling_disagreement=sum(int(np.logical_xor(x[:,:,p],x[:,:,q]).sum())
        for p,q in itertools.combinations(S.PLANS,2))
    data_disagreement=sum(int(np.logical_xor(x[d,:,:],x[e,:,:]).sum())
        for d,e in itertools.combinations(range(10),2))
    return dict(total=int(x.sum()),by_dataset={d:int(x[k].sum()) for k,d in enumerate(S.DATA)},
        by_initialization={j:int(x[:,k,:].sum()) for k,j in enumerate(S.INIT)},
        by_plan={p:int(x[:,:,p].sum()) for p in S.PLANS},
        dataset_init_pairs_changing_across_plans=int(changed.sum()),
        sampling_pair_disagreements=sampling_disagreement,sampling_pair_comparisons=1000,
        matched_dataset_disagreements=data_disagreement,matched_dataset_comparisons=2250,
        variance_decomposition=decomposition(x)),x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--progress-only',action='store_true');a=parser.parse_args()
    torch.set_num_threads(1);cells=progress()
    if a.progress_only:return
    audit=json.loads((S.OUT/'preflight.json').read_text());assert audit['passed']
    assert audit['dependencies']==S.dependencies()
    rows=[];hashes={};seen=set()
    sources={d:S.source(d) for d in S.DATA}
    initial={j:S.initial(j).state_dict() for j in S.INIT}
    fresh=S.preflight()
    for key in ('source_hashes','initial_hashes','plan_hashes','config','dependencies'):
        # JSON roundtrip normalizes integer dictionary keys.
        assert json.loads(json.dumps(fresh[key]))==audit[key]
    for path in sorted((S.OUT/'runs').glob('*-n4000.pt')):
        s=R.trials.load(path);i=s['identity'];d,j,p=i['data_seed'],i['init_seed'],i['plan']
        assert (d,j,p) not in seen;seen.add((d,j,p))
        assert d in S.DATA and j in S.INIT and p in S.PLANS and i['steps']==4000
        c,data,source_hash=sources[d]
        assert i['source_hash']==source_hash and i['dependencies']==audit['dependencies']
        assert i['initial_hash']==R.state_hash(s['initial'])==audit['initial_hashes'][str(j)]
        assert all(torch.equal(v,initial[j][k]) for k,v in s['initial'].items())
        assert i['plan_hashes']==audit['plan_hashes'][str(p)]
        assert vars(c)==s['config']==audit['config'] and s['step']==4000
        assert [f['step'] for f in s['frames']]==list(range(0,4001,100))
        model=S.B.Encoder();model.load_state_dict(s['model']);g=R.geometry(model,c,20001)
        assert g==s['geometry']
        all_d=1/torch.cat([data['x'],data['xp']])[:,1]
        in_support=[r for r in g['roots'] if float(all_d.min())<=r<=float(all_d.max())]
        row=dict(data_seed=d,init_seed=j,plan=p,**g,roots_in_source_support=in_support,
                 initial_folded=s['frames'][0]['folded'],seconds=s['seconds'])
        cell=next(r for r in cells if (r['data_seed'],r['init_seed'],r['plan'])==(d,j,p))
        assert all(cell[k]==v for k,v in dict(**g,seconds=s['seconds']).items())
        rows.append(row);hashes[path.name]=R.digest(path.read_bytes())
    assert len(rows)==len(cells)==500 and len(seen)==500
    folds,x=analyze(rows,'folded');material,m=analyze(rows,'material_fold')
    summary=dict(status='complete_audited',completed=500,planned=500,folds=folds,material=material,
        source_support_folds=sum(bool(r['roots_in_source_support']) for r in rows),rows=rows)
    R.trials.atomic_json(S.OUT/'summary.json',summary)
    R.trials.atomic_json(S.OUT/'verification.json',dict(passed=True,checkpoints=500,
        geometry_reload_exact=True,initial_weights_and_plans_regenerated=True,
        sources_and_dependencies_verified=True,preflight_replays=audit['historical_replay_max_error'],
        resume_model_and_adam_exact=audit['resume_model_and_adam_exact'],
        report_hash=R.digest(Path(__file__).read_bytes()),checkpoint_hashes=hashes))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,tensor,title in zip(axes,(x,m),('Any final turn','Material final turn')):
        counts=tensor.sum(2);im=ax.imshow(counts,vmin=0,vmax=5,cmap='Blues')
        ax.set(xticks=range(10),xticklabels=S.INIT,yticks=range(10),yticklabels=S.DATA,
               xlabel='Encoder initialization',ylabel='Saved dataset',title=title)
        for d in range(10):
            for j in range(10):ax.text(j,d,f'{counts[d,j]}/5',ha='center',va='center',fontsize=8,
                                     color='white' if counts[d,j]>=3 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Plans ending folded (of 5)',shrink=.75)
    fig.suptitle('Data × initialization × independent sampling: 500 fits')
    fig.savefig(S.OUT/'sampling_crossing.png',dpi=180,bbox_inches='tight');plt.close(fig)
    text=f'''# Independent data and minibatch sampling: completed results

All 500 fits completed and passed independent checkpoint reload, source/initialization/plan checks, and 20,001-point float64 derivative evaluation. Two original full-run replays and the short model/Adam interruption-resume check passed exactly before launch. Final detected folds: **{folds['total']}/500**, including **{material['total']}/500 material folds**.

Every dataset/init pair was trained with five fresh, independently seeded alignment/regularizer plan pairs. At each plan label, all datasets and initializations share exact index tensors. Thus dataset identity no longer determines a unique training plan.

| Dataset | Any final fold / 50 | Material final fold / 50 |
| --- | ---: | ---: |
'''
    for d in S.DATA:text+=f"| {d} | {folds['by_dataset'][d]}/50 | {material['by_dataset'][d]}/50 |\n"
    text+='\n| Batch-plan label | Any final fold / 100 | Material final fold / 100 |\n| --- | ---: | ---: |\n'
    for p in S.PLANS:text+=f"| {p} | {folds['by_plan'][p]}/100 | {material['by_plan'][p]}/100 |\n"
    text+=f'''
Across the five plans, {folds['dataset_init_pairs_changing_across_plans']}/100 fixed dataset/init pairs change binary fold status; {material['dataset_init_pairs_changing_across_plans']}/100 change material-fold status. Across all plan pairs at fixed data/init, disagreement occurs in {folds['sampling_pair_disagreements']}/1000 binary-fold comparisons and {material['sampling_pair_disagreements']}/1000 material-fold comparisons. Across dataset pairs at fixed init/plan, disagreement occurs in {folds['matched_dataset_disagreements']}/2250 and {material['matched_dataset_disagreements']}/2250 comparisons, respectively. These are descriptive pairwise disagreement rates, not independent statistical tests.

## Descriptive balanced-factor decomposition

Fraction of centered sum of squares for binary cell outcomes, including interactions. No residual is treated as independent noise; the three-way term is retained. These fractions describe this particular finite crossed matrix, not population causal shares.

| Term | Any fold | Material fold |
| --- | ---: | ---: |
'''
    for key,v in folds['variance_decomposition']['fractions'].items():
        text+=f"| {key.replace('_',' × ')} | {v:.3%} | {material['variance_decomposition']['fractions'][key]:.3%} |\n"
    text+='''
These fits isolate encoder-training minibatch randomness from the fixed saved datasets conditional on the tested initialization bank and optimizer. They do not distinguish actor-training randomness from collection randomness, establish that occupancy is the unique dataset mechanism, or assess fresh-policy generalization. Shared episode/time indices do not match the physical observations across datasets. Five plans are a limited sampling replication. Fold detection remains a finite-grid diagnostic; no continuous-domain injectivity guarantee or downstream planning benefit is claimed.

[Cell results and decompositions](output/summary.json) · [Independent audit](output/verification.json) · [Preflight](output/preflight.json) · [Fold matrices](output/sampling_crossing.png) · [Protocol](PROTOCOL.md)
'''
    (S.HERE/'RESULTS.md').write_text(text)
    R.trials.atomic_json(S.OUT/'progress.json',dict(completed=500,planned=500,status='complete_audited',
        folds=folds['total'],material_folds=material['total']))
    print(json.dumps(dict(completed=500,folds=folds,material=material)),flush=True)

if __name__=='__main__':main()

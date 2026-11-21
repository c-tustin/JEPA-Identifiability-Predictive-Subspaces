"""Audit every mass/objective cell and retain minor as well as large turns."""
import itertools
import json
from pathlib import Path
import dose as D

S,R,torch,np,B=D.S,D.R,D.torch,D.np,D.B


def main():
    torch.set_num_threads(1);ds,c,initial,ix,u,mean,std,critic=D.setup()
    paths=sorted((D.OUT/'runs').glob('*.pt'));assert len(paths)==120
    ids_hashes={};rows=[];hashes={};fd_errors=[]
    for path in paths:
        s=R.trials.load(path);i=s['identity'];n,f,a,method=[i[k] for k in ['near','far','alpha','method']]
        assert i['dependencies']==D.dependencies()
        assert i['initial_hash']==R.state_hash(initial)==R.state_hash(s['initial'])
        assert i['critic_initial_hash']==R.state_hash(critic)==R.state_hash(s['critic_initial'])
        assert i['alignment_plan_hash']==R.digest(ix.numpy().tobytes())
        assert i['uniform_plan_hash']==R.digest(u.numpy().tobytes())
        key=(n,f,a)
        if key not in ids_hashes:
            ids=S.indices(a,len(ds[n]['near']),len(ds[f]['far']),u)
            ids_hashes[key]=R.digest(ids.numpy().tobytes())
        assert i['endpoint_index_hash']==ids_hashes[key]
        assert torch.equal(s['raw_mean'],mean) and torch.equal(s['raw_std'],std)
        assert abs(s['realized_near_mass']-a)<.001
        for seed,h in i['source_hashes'].items():assert h==ds[seed]['source_hash']
        m=B.Encoder();m.load_state_dict(s['model']);g=R.geometry(m,c,20001)
        assert all(s['geometry'][k]==v for k,v in g.items())
        lo,hi=s['target_support'];g['roots_in_source_support']=[d for d in g['roots'] if lo<=d<=hi]
        assert g==s['geometry']
        md=m.double();q=torch.linspace(.01,np.log(16)-.01,31,dtype=torch.float64,requires_grad=True)
        y=md(B.sensor(q.exp(),c));derivative=torch.autograd.grad(y.sum(),q)[0]
        with torch.no_grad():finite=(md(B.sensor((q+1e-5).exp(),c))-md(B.sensor((q-1e-5).exp(),c)))/2e-5
        error=float((finite-derivative).abs().max());assert error<1e-6;fd_errors.append(error)
        rows.append(dict(name=path.stem,near=n,far=f,alpha=a,method=method,**g))
        hashes[path.name]=R.digest(path.read_bytes())
    counts={method:dict(folds=sum(r['folded'] for r in rows if r['method']==method),
                       material=sum(r['material_fold'] for r in rows if r['method']==method),
                       minimum_std=min(r['std'] for r in rows if r['method']==method)) for method in D.METHODS}
    summary=dict(completed=120,counts=counts,rows=rows)
    R.trials.atomic_json(D.OUT/'summary.json',summary)
    R.trials.atomic_json(D.OUT/'verification.json',dict(passed=True,checkpoints=120,dense_reload_exact=True,
        sources_components_initializers_and_plans_verified=True,raw_normalizer_fixed=True,
        maximum_derivative_finite_difference_error=max(fd_errors),checkpoint_hashes=hashes,
        report_hash=R.digest(Path(__file__).read_bytes())))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,8))
    templates=list(itertools.product([130,136],repeat=2))
    for ax,method in zip(axes,D.METHODS):
        rs=[[next(r for r in rows if (r['near'],r['far'],r['alpha'],r['method'])==(n,f,a,method)) for a in D.MASSES] for n,f in templates]
        mat=np.array([[r['backward_fraction'] for r in line] for line in rs])
        im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=.4,aspect='auto')
        ax.set(xticks=range(10),xticklabels=[f'{a:.3%}' for a in D.MASSES],
               yticks=range(4),yticklabels=[f'{n}/{f}' for n,f in templates],ylabel='Near/far profile',
               title=dict(sigreg='SIGReg',raw_infonce='Raw/own-code InfoNCE',variance_control='Separate-input variance control')[method])
        for y,line in enumerate(rs):
            for x,r in enumerate(line):ax.text(x,y,'F' if r['material_fold'] else ('tiny' if r['folded'] else '·'),ha='center',va='center',fontsize=8,color='white' if mat[y,x]>.2 else 'black')
    axes[-1].set_xlabel('Target mass below distance 2 (conditional profiles held fixed)')
    fig.suptitle('F = material turn; tiny = smaller sampled turn; color = backward variation fraction')
    fig.tight_layout();fig.savefig(D.OUT/'dose_response.png',dpi=180);plt.close(fig)
    lines=['# Completed mass × conditional shape × objective sweep','',
        'All 120 fits passed independent dense float64 reload, component/source/initializer/plan checks, fixed raw normalization and derivative finite differences. Counts are descriptive across four reused conditional-profile templates and one encoder initializer.','',
        '| Near mass | SIGReg material / 4 | Raw/own-code InfoNCE material / 4 | Variance control material / 4 |',
        '| --- | ---: | ---: | ---: |']
    for a in D.MASSES:
        vals=[sum(r['material_fold'] for r in rows if r['alpha']==a and r['method']==m) for m in D.METHODS]
        lines.append(f'| {a:.3%} | '+' | '.join(map(str,vals))+' |')
    lines+=['','Totals (any turn / material turn): '+', '.join(f'{m}: {counts[m]["folds"]}/40 / {counts[m]["material"]}/40' for m in D.METHODS)+'.','',
        'The mass intervention creates a substantial-fold window for SIGReg: all templates fold materially at 62.5%, 68.75%, 75% and 81.25%, none at 25%, 50%, 93.75% or 96.875%. At 87.5% and 90.625%, conditional shape affects the branch (3/4 and 1/4 material folds). Thus a universal threshold based only on near mass is false at the boundary. The historical substantial-fold masses (68.58–87.33%) and nonfolded masses (92.98–95.39%) lie on different sides of this experimentally manipulated window, consistent with the stage-1 mass swaps.','',
        'Raw/own-code InfoNCE has no detected final turn in any of the 40 cells. The variance control has numerous small turns, including two just above the material threshold at 25% near mass; it must not be described as universally fold-free. It uses the explicitly specified separate-input control, not the earlier joint-data VICReg replay. Candidate counts, loss weights, critics and compute differ across methods; this is an occupancy dose test within each method, not a method ranking.','',
        'This narrows the feature to a near-cluster/far-tail mass imbalance with a shape-dependent boundary, but does not establish a universal sufficient statistic. Objective-level risk comparisons, critic/collision checks, new initializations/plans, new sources, and temporal InfoNCE remain needed before completing the full goal.','',
        '[Figure](dose_response.png) · [Measurements](summary.json) · [Audit](verification.json) · [Protocol](../../DOSE_PROTOCOL.md)','']
    (D.OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()

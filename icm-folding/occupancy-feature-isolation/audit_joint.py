"""Audit the actual same-transition four-objective bridge."""
import json
from pathlib import Path
import joint as J

F,D,S,R,torch,np,B=J.F,J.D,J.S,J.R,J.torch,J.np,J.B


def main():
    torch.set_num_threads(1);ds,c,initial,_,_,mean,std,critic=F.setup()
    pre=json.loads((J.OUT/'preflight.json').read_text())
    assert pre['passed'] and pre['shared_forward_calls']==1 and pre['dependencies']==J.dependencies()
    paths=sorted((J.OUT/'runs').glob('*.pt'));assert len(paths)==160
    u=torch.rand((4000,512),dtype=torch.float64,generator=torch.Generator().manual_seed(9000131))
    rows=[];hashes={};plans={}
    for path in paths:
        s=R.trials.load(path);i=s['identity'];seed,a,method=[i[k] for k in ['source','alpha','method']]
        assert i['dependencies']==J.dependencies()
        assert i['source_hash']==ds[seed]['source_hash']
        assert i['initial_hash']==R.state_hash(initial)==R.state_hash(s['initial'])
        assert i['critic_initial_hash']==R.state_hash(critic)
        assert i['uniform_hash']==R.digest(u.numpy().tobytes())
        key=(seed,a)
        if key not in plans:
            data,ids,measure=J.prepare(ds[seed],a,u)
            assert abs(measure['weights'][0]+measure['weights'][1]/2-a)<1e-12
            plans[key]=(R.digest(ids.numpy().tobytes()),measure)
        ph,measure=plans[key];assert i['plan_hash']==ph and s['measure']==measure
        assert torch.equal(s['raw_mean'],mean) and torch.equal(s['raw_std'],std)
        model=B.Encoder();model.load_state_dict(s['model']);g=R.geometry(model,c,20001)
        assert all(s['geometry'][k]==v for k,v in g.items())
        lo,hi=s['support'];g['roots_in_source_support']=[d for d in g['roots'] if lo<=d<=hi]
        assert g==s['geometry']
        rows.append(dict(source=seed,alpha=a,method=method,is_native=a==measure['native_mass'],measure=measure,**g))
        hashes[path.name]=R.digest(path.read_bytes())
    groups=[]
    for condition in ['native',.5,.75,.9375]:
        for method in J.METHODS:
            rs=[r for r in rows if r['method']==method and (r['is_native'] if condition=='native' else r['alpha']==condition)]
            assert len(rs)==10
            groups.append(dict(condition=condition,method=method,folds=sum(r['folded'] for r in rs),
                material=sum(r['material_fold'] for r in rs),support_turns=sum(bool(r['roots_in_source_support']) for r in rs),
                minimum_std=min(r['std'] for r in rs)))
    summary=dict(completed=160,groups=groups,rows=rows,
        maximum_near_shape_tv_bound=max(r['measure']['near_shape_tv_bound'] for r in rows),
        maximum_far_shape_tv_bound=max(r['measure']['far_shape_tv_bound'] for r in rows))
    R.trials.atomic_json(J.OUT/'summary.json',summary)
    R.trials.atomic_json(J.OUT/'verification.json',dict(passed=True,checkpoints=160,dense_reload_exact=True,
        sources_initializers_and_plans_verified=True,transition_region_masses_verified=True,
        shared_forward_preflight=pre,checkpoint_hashes=hashes,report_hash=R.digest(Path(__file__).read_bytes())))
    lines=['# Joint-data temporal InfoNCE and scalar VICReg bridge','',
        'All 160 fits passed independent dense reload, source/initializer/plan, target-region and shared-input checks. Native corners recover the empirical joint distribution exactly. Every method receives the same sampled transition rows and one encoder forward per update.','',
        '| Endpoint near mass | Objective | Any turn / 10 | Material / 10 | Turns within stored range / 10 |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in groups:lines.append(f'| {r["condition"]} | {r["method"]} | {r["folds"]} | {r["material"]} | {r["support_turns"]} |')
    lines+=['','This is a joint-data bridge, not a pure-marginal intervention. Transition-category weights change with near mass, and the conditional endpoint profiles change by a bounded amount because cross-transition probability is held fixed. Maximum near/far conditional-mixture TV bounds are '+f'{summary["maximum_near_shape_tv_bound"]:.5f} / {summary["maximum_far_shape_tv_bound"]:.5f}'+'. These changes are retained and prevent interpreting this alone as an occupancy-only causal test. The separate endpoint-only study supplies that control.','',
        'Temporal and raw/own-code InfoNCE results are reported separately. Scalar VICReg can have small turns under these interventions; its tested behavior is not an injectivity guarantee. Differences in candidate counts, losses and critics mean this is not a matched-compute method ranking.','',
        '[All measurements](summary.json) · [Independent audit](verification.json) · [Protocol](../../JOINT_PROTOCOL.md)','']
    (J.OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()

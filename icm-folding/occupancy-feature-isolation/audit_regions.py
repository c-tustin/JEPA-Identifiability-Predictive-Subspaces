"""Audit regional interventions, including all recorded counterexamples."""
import argparse
import json
from pathlib import Path
import fresh_fit as F
import fresh_shape as C
import support_split as U
import far_regions as G

D,S,R,torch,np,B=F.D,F.S,F.R,F.torch,F.D.np,F.B


def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['shapes','split','common','far']);a=p.parse_args()
    torch.set_num_threads(1);ds,c,initial,ix,u,_,_,_=F.setup()
    if a.group in ['split','common']:
        u=torch.rand((4000,1024),dtype=torch.float64,generator=torch.Generator().manual_seed(9200131))
    if a.group=='shapes':out=C.OUT;paths=sorted(out.glob('*/runs/*.pt'));expected=12
    elif a.group=='split':out=U.OUT;paths=sorted(out.glob('split*/pair*/runs/*.pt'));expected=64
    elif a.group=='common':out=U.OUT/'common';paths=sorted((out/'runs').glob('*.pt'));expected=40
    else:out=G.OUT;paths=sorted((out/'runs').glob('*.pt'));expected=24
    assert len(paths)==expected,(len(paths),expected)
    if a.group=='common':pool=U.common_pool(ds)
    rows=[];hashes={};replays={}
    for path in paths:
        s=R.trials.load(path);i=s['identity'];spec={}
        assert i['dependencies']==D.dependencies()
        assert R.state_hash(s['initial'])==R.state_hash(initial)==i['initial_hash']
        for seed,h in i['source_hashes'].items():assert h==ds[seed]['source_hash']
        if a.group in ['shapes','split']:
            scope=json.loads((path.parent.parent/'scope'/f'{path.stem}.json').read_text())
            assert scope['checkpoint_hash']==R.digest(path.read_bytes())
            pair,bits=scope['pair'],scope['bits'];cut=2 if a.group=='shapes' else scope['cut']
            spec=dict(pair=pair,bits=bits,cut=cut)
            proto='FRESH_SHAPE_PROTOCOL.md' if a.group=='shapes' else 'SPLIT_SUPPORT_PROTOCOL.md'
            code='fresh_shape.py' if a.group=='shapes' else 'support_split.py'
            assert scope['governing_protocol_hash']==R.digest((S.HERE/proto).read_bytes())
            assert scope['wrapper_hash']==R.digest((S.HERE/code).read_bytes())
            if a.group=='shapes':
                n,f=[pair[b] for b in bits];alpha=.5;changed=ds
            else:
                changed={seed:dict(v,near=v['x'][v['d']<cut],far=v['x'][v['d']>=cut],mass=float((v['d']<cut).double().mean())) for seed,v in ds.items()}
                ms,n,f=[pair[b] for b in bits];alpha=changed[ms]['mass']
            ids=S.indices(alpha,len(changed[n]['near']),len(changed[f]['far']),u)
            assert i['endpoint_index_hash']==R.digest(ids.numpy().tobytes())
            if a.group=='shapes' and n==f:
                old=R.trials.load(F.OUT/'runs'/path.name)
                err=max(float((v-old['model'][k]).abs().max()) for k,v in s['model'].items())
                assert err==scope['diagonal_max_parameter_error']==0;replays[str(path.relative_to(out))]=err
        elif a.group=='common':
            assert i['wrapper_hash']==U.own_hashes()['wrapper_hash'] and i['governing_protocol_hash']==U.own_hashes()['governing_protocol_hash']
            x,bins,counts,profiles=pool;pair,bits=i['pair'],i['bits'];ms,n,f=[pair[b] for b in bits]
            alpha=ds[ms]['mass'];target=alpha*profiles[n][0]+(1-alpha)*profiles[f][1]
            weights=target[bins]/counts[bins];assert torch.all(weights>0)
            assert abs(float(target[:8].sum())-alpha)<1e-12
            assert i['target_bin_mass']==target.tolist() and i['pool_hash']==R.digest(x.numpy().tobytes())
            cdf=weights.cumsum(0);cdf[-1]=1;ids=torch.searchsorted(cdf,u.flatten()).reshape(u.shape).clamp_max(len(x)-1)
            assert i['index_hash']==R.digest(ids.numpy().tobytes())
            spec=dict(pair=pair,bits=bits,alpha=alpha)
        else:
            assert i['governing_protocol_hash']==R.digest((S.HERE/'FAR_REGIONS_PROTOCOL.md').read_bytes())
            assert i['code_hash']==R.digest((S.HERE/'far_regions.py').read_bytes())
            assert i['training_helper_hash']==R.digest((S.HERE/'train_marginal.py').read_bytes())
            pair,bits=i['pair'],i['bits'];x,ids,distribution=G.distribution(pair,bits,ds,u)
            assert i['index_hash']==R.digest(ids.numpy().tobytes())
            for k,v in distribution.items():assert i[k]==v
            if distribution['native_cdf_shortcut']:
                n=pair[0];f=pair[bits[2]]
                ref=R.trials.load(C.OUT/f'{pair[0]}-{pair[1]}'/'runs'/f'n{n}-f{f}-a0.5-sigreg.pt')
                err=max(float((v-ref['model'][k]).abs().max()) for k,v in s['model'].items());assert err==0
                replays[path.name]=err
            spec=distribution
        m=B.Encoder();m.load_state_dict(s['model']);g=R.geometry(m,c,20001)
        assert all(s['geometry'][k]==v for k,v in g.items())
        lo,hi=s.get('target_support',s.get('support'));g['roots_in_source_support']=[d for d in g['roots'] if lo<=d<=hi]
        if 'roots_in_source_support' in s['geometry']:assert g==s['geometry']
        rows.append(dict(name=str(path.relative_to(out)),**spec,**g))
        hashes[str(path.relative_to(out))]=R.digest(path.read_bytes())
    summary=dict(group=a.group,completed=len(rows),folds=sum(r['folded'] for r in rows),
                 material=sum(r['material_fold'] for r in rows),rows=rows)
    R.trials.atomic_json(out/f'{a.group}-summary.json',summary)
    R.trials.atomic_json(out/f'{a.group}-verification.json',dict(passed=True,checkpoints=len(rows),
        dense_reload_exact=True,regional_distributions_and_plans_regenerated=True,source_hashes_verified=True,
        native_corner_parameter_errors=replays,checkpoint_hashes=hashes,
        report_hash=R.digest(Path(__file__).read_bytes())))
    lines=[f'# {a.group}: audited regional intervention results','',
           f'All {len(rows)} fits passed independent dense reload and source/component/plan checks. Any turns: {summary["folds"]}; material turns: {summary["material"]}. Native-corner exact parameter replays: {len(replays)}.','',
           '| Case | Components | Turn | Material | Roots |','| --- | --- | --- | --- | --- |']
    for r in rows:lines.append(f'| {r.get("pair")}'+(f' / cut {r["cut"]}' if 'cut' in r else '')+f' | {r.get("bits")} | {r["folded"]} | {r["material_fold"]} | '+', '.join(f'{d:.3f}' for d in r['roots'])+' |')
    lines+=['','All cases are retained, including tiny/extrapolative turns. These results do not by themselves define a universal mass or moment criterion.','',
            f'[Measurements]({a.group}-summary.json) · [Independent audit]({a.group}-verification.json)','']
    (out/f'{a.group}-RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()

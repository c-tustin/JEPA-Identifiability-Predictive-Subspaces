"""Independently audit unchanged dose-fit runs under explicit wrapper scopes."""
import argparse
import json
from pathlib import Path
import dose as D
import robustness as V
import fresh_fit as F

S,R,torch,np,B=D.S,D.R,D.torch,D.np,D.B


def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['robustness','fresh']);args=p.parse_args()
    torch.set_num_threads(1);is_fresh=args.group=='fresh'
    if is_fresh:
        ds,c,initial,ix,u,mean,std,critic=F.setup();out=F.OUT
        lock=F.prediction_lock();assert lock==json.loads((S.HERE/'PREDICTIONS_LOCK.json').read_text())
        paths=sorted((out/'runs').glob('*.pt'));assert len(paths)==80
    else:
        ds,c,ix,mean,std,initials,critics,uniforms,meta=V.setup();out=V.OUT
        assert json.loads(json.dumps(meta))==json.loads((out/'preflight.json').read_text())
        paths=sorted(out.glob('init*-plan*/runs/*.pt'));assert len(paths)==120
    rows=[];hashes={};cached={}
    for path in paths:
        s=R.trials.load(path);i=s['identity'];n,f,a,method=[i[k] for k in ['near','far','alpha','method']]
        scope_path=path.parent.parent/'scope'/f'{path.stem}.json';scope=json.loads(scope_path.read_text())
        assert scope['checkpoint_hash']==R.digest(path.read_bytes())
        if is_fresh:
            assert scope['governing_protocol_hash']==lock['protocol_hash']
            assert scope['prediction_file_hash']==lock['files'][str(n)]
            assert scope['wrapper_hash']==R.digest((S.HERE/'fresh_fit.py').read_bytes())
            assert scope['is_native']==(a==ds[n]['mass']) and scope['source']==n
            seed,plan=131,0
        else:
            seed,plan=scope['initialization'],scope['plan_label']
            initial,critic,u=initials[seed],critics[seed],uniforms[plan]
            assert scope['governing_protocol_hash']==meta['governing_protocol_hash']
            assert scope['wrapper_hash']==meta['wrapper_hash']
        assert i['dependencies']==D.dependencies()
        assert i['initial_hash']==R.state_hash(initial)==R.state_hash(s['initial'])
        assert i['critic_initial_hash']==R.state_hash(critic)==R.state_hash(s['critic_initial'])
        assert i['alignment_plan_hash']==R.digest(ix.numpy().tobytes())
        assert i['uniform_plan_hash']==R.digest(u.numpy().tobytes())
        key=(n,f,a,plan)
        if key not in cached:
            ids=S.indices(a,len(ds[n]['near']),len(ds[f]['far']),u)
            cached[key]=R.digest(ids.numpy().tobytes())
        assert i['endpoint_index_hash']==cached[key]
        assert torch.equal(s['raw_mean'],mean) and torch.equal(s['raw_std'],std)
        for source,h in i['source_hashes'].items():assert h==ds[source]['source_hash']
        m=B.Encoder();m.load_state_dict(s['model']);g=R.geometry(m,c,20001)
        assert all(s['geometry'][k]==v for k,v in g.items())
        lo,hi=s['target_support'];g['roots_in_source_support']=[d for d in g['roots'] if lo<=d<=hi]
        assert g==s['geometry']
        rows.append(dict(source=n,alpha=a,method=method,initialization=seed,plan=plan,
                         is_native=(a==ds[n]['mass']) if is_fresh else False,**g))
        hashes[str(path.relative_to(out))]=scope['checkpoint_hash']
    summary=dict(group=args.group,completed=len(rows),rows=rows)
    if is_fresh:
        preds={s:json.loads((S.OUT/'fresh/predictions'/f'{s}.json').read_text()) for s in range(9100,9110)}
        checks=[]
        for seed in preds:
            r=next(r for r in rows if r['source']==seed and r['is_native'] and r['method']=='sigreg')
            pr=preds[seed]
            checks.append(dict(source=seed,near_mass=pr['near_mass'],observed_material=r['material_fold'],observed_any=r['folded'],
                bank_predicts=pr['risk_bank_predicts_fold'],coarse_predicts=pr['coarse_window_predicts_fold'],
                risk_margin=pr['risk_difference_folded_minus_ordered']))
        summary.update(predictions=checks,bank_correct=sum(x['bank_predicts']==x['observed_material'] for x in checks),
                       coarse_correct=sum(x['coarse_predicts']==x['observed_material'] for x in checks))
    summary['counts']=[dict(alpha=a,method=m,cells=len(q),folds=sum(r['folded'] for r in q),material=sum(r['material_fold'] for r in q))
        for a in [.5,.75,.9375] for m in ['sigreg','raw_infonce']
        if (q:=[r for r in rows if r['alpha']==a and r['method']==m])]
    R.trials.atomic_json(out/'summary.json',summary)
    R.trials.atomic_json(out/'verification.json',dict(passed=True,checkpoints=len(rows),dense_reload_exact=True,
        source_scope_initializer_and_plan_checks=True,checkpoint_hashes=hashes,
        report_hash=R.digest(Path(__file__).read_bytes())))
    title='New-source controlled validation' if is_fresh else 'New initialization and sampling replication'
    lines=['# '+title,'',f'All {len(rows)} fits passed independent dense float64 reload and source/scope/initializer/plan checks.','',
        '| Near mass | Objective | Fits | Any turn | Material turn |','| --- | --- | ---: | ---: | ---: |']
    for r in summary['counts']:lines.append(f'| {r["alpha"]:.3%} | {r["method"]} | {r["cells"]} | {r["folds"]} | {r["material"]} |')
    if is_fresh:
        lines+=['',f'The frozen twelve-encoder risk bank predicts material status correctly in {summary["bank_correct"]}/10 native fits; the coarse near-mass window in {summary["coarse_correct"]}/10. Predictions were recorded and hash-locked before encoder fitting; the bank was frozen before generating the ten policy sources. No sources were replaced.','',
            '| Source | Native near mass | Observed material fold | Risk-bank prediction | Coarse mass prediction |',
            '| --- | ---: | --- | --- | --- |']
        for r in summary['predictions']:lines.append(f'| {r["source"]} | {r["near_mass"]:.3%} | {r["observed_material"]} | {r["bank_predicts"]} | {r["coarse_predicts"]} |')
        lines+=['','These are newly trained actors and newly collected trajectories in the same renderer. The independent endpoint interventions hold source-131 alignment fixed; temporal InfoNCE and original joint-data VICReg are checked separately. A finite reference-bank predictor is not a universal distribution theorem.']
    else:lines+=['','Each mass/method cell includes both conditional templates, five contiguous encoder initializations and two fresh sampling plans. This tests the core low/middle/high contrast; boundary masses and other worlds are not covered by this replication.']
    lines+=['','[All measurements](summary.json) · [Independent audit](verification.json)','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()

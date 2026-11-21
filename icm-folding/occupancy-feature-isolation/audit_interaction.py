"""Exact detached controls and finite-pair-batch objective checks."""
import argparse
import itertools
import json
from pathlib import Path
import interaction as A
import temporal_interaction as V

F,D,S,R,torch,np,B,J=A.F,A.D,A.S,A.R,A.torch,A.np,A.B,A.J


def expected_penalty(y,yp,weights,reg,pairs=512):
    n=2*pairs;phase=y[:,None]*reg.frequency;other=yp[:,None]*reg.frequency
    cc=(weights[:,None]*(phase.cos()+other.cos())/2).sum(0)
    ss=(weights[:,None]*(phase.sin()+other.sin())/2).sum(0)
    uu=(weights[:,None]*(.5+.5*(phase-other).cos())).sum(0)
    return (reg.weight*(n*((cc-reg.target).square()+ss.square())+n/pairs*(uu-cc.square()-ss.square()))).sum()


def formula_check():
    reg=B.SIGReg1D().double();y=torch.tensor([-.5,.8],dtype=torch.float64)
    yp=torch.tensor([-.1,1.2],dtype=torch.float64);w=torch.tensor([.3,.7],dtype=torch.float64)
    actual=0.
    for a,b in itertools.product(range(2),repeat=2):
        actual+=float(w[a]*w[b]*reg(torch.cat([y[[a,b]],yp[[a,b]]])))
    expected=float(expected_penalty(y,yp,w,reg,pairs=2));assert abs(actual-expected)<1e-12
    return dict(enumerated=actual,formula=expected)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['raw','temporal']);a=p.parse_args()
    torch.set_num_threads(1);module=A if a.mode=='raw' else V
    out=module.OUT;paths=sorted((out/'runs').glob('*.pt'));assert len(paths)==(36 if a.mode=='raw' else 30)
    ds,c,_,_,_,mean,std,_=F.setup();reg=B.SIGReg1D().double();rows=[];hashes={};saved_states={}
    u=torch.rand((4000,512),dtype=torch.float64,generator=torch.Generator().manual_seed(9300131))
    for path in paths:
        s=R.trials.load(path);i=s['identity'];source=i['source'];seed=i['initialization']
        assert i['code_hash']==R.digest(Path(module.__file__).read_bytes())
        proto='INTERACTION_PROTOCOL.md' if a.mode=='raw' else 'TEMPORAL_INTERACTION_PROTOCOL.md'
        assert i['protocol_hash']==R.digest((S.HERE/proto).read_bytes()) and i['dependencies']==J.dependencies()
        assert i['source_hash']==ds[source]['source_hash']
        torch.manual_seed(40000+seed);initial=B.Encoder().state_dict()
        assert i['initial_hash']==R.state_hash(initial)==R.state_hash(s['initial'])
        data,ids,measure=J.prepare(ds[source],.75,u)
        assert i['plan_hash']==R.digest(ids.numpy().tobytes()) and measure==s['measure']
        if a.mode=='raw':
            torch.manual_seed(6100000+seed);critic=D.N.critic().state_dict()
            assert i['critic_initial_hash']==R.state_hash(critic)==R.state_hash(s['critic_initial'])
            assert torch.equal(mean,s['raw_mean']) and torch.equal(std,s['raw_std'])
        model=B.Encoder();model.load_state_dict(s['model']);g=R.geometry(model,c,20001)
        assert all(s['geometry'][k]==v for k,v in g.items())
        lo,hi=s['support'];g['roots_in_source_support']=[d for d in g['roots'] if lo<=d<=hi]
        assert g==s['geometry']
        weights=torch.cat([torch.full((n,),w/n,dtype=torch.float64) for w,n in zip(measure['weights'],measure['counts']) if n])
        assert abs(float(weights.sum())-1)<1e-12
        model=model.double()
        with torch.no_grad():
            y=model(data['x'].double());yp=model(data['xp'].double())
            alignment=float((weights*(y-yp).square()).sum())
            penalty=float(expected_penalty(y,yp,weights,reg))
        rows.append(dict(**i,**g,expected_alignment=alignment,expected_sigreg=penalty,
                         expected_original_objective=.9*alignment+.1*penalty))
        hashes[path.name]=R.digest(path.read_bytes())
        saved_states[(source,seed,i['lambda_info'],i['detached'])]=s
    replays=[];comparisons=[]
    for source,seed in itertools.product([131,136,9109],[131,132]):
        baseline=saved_states[source,seed,0.,False];detached=saved_states[source,seed,100.,True]
        error=max(float((v-baseline['model'][k]).abs().max()) for k,v in detached['model'].items());assert error==0
        states_equal=all(torch.equal(v,baseline['optimizer']['state'][k][key]) for k in range(6) for key,v in detached['optimizer']['state'][k].items())
        replays.append(dict(source=source,initialization=seed,encoder_parameter_error=error,encoder_adam_states_exact=states_equal))
        get=lambda lam,dt:next(r for r in rows if (r['source'],r['initialization'],r['lambda_info'],r['detached'])==(source,seed,lam,dt))
        high,zero=get(100.,False),get(0.,False)
        comparisons.append(dict(source=source,initialization=seed,baseline_material=zero['material_fold'],
            attached100_material=high['material_fold'],detached100_material=get(100.,True)['material_fold'],
            original_objective_change_attached100=high['expected_original_objective']-zero['expected_original_objective'],
            gaussian_penalty_change_attached100=high['expected_sigreg']-zero['expected_sigreg']))
    groups=[]
    for lam,dt in module.ARMS:
        rs=[r for r in rows if r['lambda_info']==lam and r['detached']==dt]
        groups.append(dict(lambda_info=lam,detached=dt,cases=len(rs),folds=sum(r['folded'] for r in rs),material=sum(r['material_fold'] for r in rs)))
    summary=dict(mode=a.mode,completed=len(rows),groups=groups,detached_replays=replays,high_weight_comparisons=comparisons,rows=rows)
    R.trials.atomic_json(out/'summary.json',summary)
    R.trials.atomic_json(out/'verification.json',dict(passed=True,checkpoints=len(rows),dense_reload_exact=True,
        source_plan_and_initializer_checks=True,detached_encoder_replay_exact=True,
        finite_pair_batch_formula_check=formula_check(),checkpoint_hashes=hashes,
        report_hash=R.digest(Path(__file__).read_bytes())))
    lines=['# '+a.mode+' InfoNCE encoder-gradient intervention','',
        f'All {len(rows)} fits passed independent dense reload and source/plan/initializer checks. All six detached controls reproduce lambda-zero encoder parameters exactly.','',
        '| InfoNCE weight | Code detached | Any turn / 6 | Material / 6 |','| ---: | --- | ---: | ---: |']
    for g in groups:lines.append(f'| {g["lambda_info"]:g} | {g["detached"]} | {g["folds"]} | {g["material"]} |')
    lines+=['','At weight 100, the changes in the original SIGReg/alignment objective are '+', '.join(f'{r["original_objective_change_attached100"]:+.4f}' for r in comparisons)+'. These are exact expected finite-pair-batch risks under the controlled joint law, not the plug-in full-data penalty with its multiplier changed. The formula passes exact enumeration on a two-transition toy case.','',
        'The attached/detached contrast isolates the effect of the discrimination gradient on the encoder. It does not establish an all-input ordering guarantee, prove every contrastive gradient is anti-folding, or identify a universal occupancy threshold. All tested weights and failures are retained.','',
        '[Measurements](summary.json) · [Independent audit](verification.json)','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':main()

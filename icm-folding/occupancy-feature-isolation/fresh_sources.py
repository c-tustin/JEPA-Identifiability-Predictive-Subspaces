"""Freeze a risk bank, generate new sources, predict before encoder fitting."""
import argparse
import json
from pathlib import Path
import mechanism as M

D,S,R,torch,np,B=M.D,M.S,M.R,M.torch,M.np,M.B
OUT=S.OUT/'fresh'


def bank():
    items=[]
    for n in [130,136]:
        for f in [130,136]:
            for a in [.5,.75,.9375]:
                p=D.OUT/'runs'/f'n{n}-f{f}-a{a:g}-sigreg.pt';s=R.trials.load(p)
                items.append(dict(path=str(p),sha=R.digest(p.read_bytes()),folded=s['geometry']['material_fold']))
    return items


def predict(seed,items):
    source_path=OUT/'sources'/f'icm-{seed}.pt';s=R.trials.load(source_path)
    c=B.Config(**s['config']);x=torch.cat([s['data']['x'],s['data']['xp']]).double()
    alpha=float(((1/x[:,1])<2).double().mean());reg=B.SIGReg1D().double()
    align=R.trials.load(R.LEGACY/'output/sources/icm-131.pt')['data'];vals=[]
    for item in items:
        p=Path(item['path']);assert R.digest(p.read_bytes())==item['sha']
        m=B.Encoder().double();m.load_state_dict(R.trials.load(p)['model'])
        with torch.no_grad():
            y=m(x);phase=y[:,None]*reg.frequency;cc=phase.cos().mean(0);ss=phase.sin().mean(0)
            penalty=((1023*(cc.square()+ss.square())-2048*reg.target*cc+1024*reg.target.square()+1)*reg.weight).sum()
            la=(m(align['x'].double())-m(align['xp'].double())).square().mean()
        vals.append(dict(**item,risk=float(.9*la+.1*penalty)))
    folded=min(v['risk'] for v in vals if v['folded']);ordered=min(v['risk'] for v in vals if not v['folded'])
    result=dict(source_seed=seed,source_hash=R.digest(source_path.read_bytes()),near_mass=alpha,
        risk_difference_folded_minus_ordered=folded-ordered,risk_bank_predicts_fold=folded<ordered,
        coarse_window_predicts_fold=.6<alpha<.91,bank_risks=vals,
        code_hash=R.digest(Path(__file__).read_bytes()),protocol_hash=R.digest((S.HERE/'FRESH_PROTOCOL.md').read_bytes()),
        prediction_precedes_any_new_encoder_fit=True)
    R.trials.atomic_json(OUT/'predictions'/f'{seed}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='bank_risks'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-bank',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);items=bank()
    if a.freeze_bank:
        assert not list((OUT/'sources').glob('*.pt'))
        R.trials.atomic_json(S.HERE/'BANK.json',dict(items=items,source_seeds=list(range(9100,9110)),
            protocol_hash=R.digest((S.HERE/'FRESH_PROTOCOL.md').read_bytes())));print('Risk bank frozen before source generation');return
    frozen=json.loads((S.HERE/'BANK.json').read_text());assert frozen['items']==items
    assert frozen['protocol_hash']==R.digest((S.HERE/'FRESH_PROTOCOL.md').read_bytes())
    c=B.Config(**R.trials.load(R.LEGACY/'output/sources/icm-131.pt')['config'])
    for k,seed in enumerate(range(9100,9110)):
        if k%a.shards!=a.shard:continue
        R.trials.source_data(c,seed,OUT)
        predict(seed,items)


if __name__=='__main__':main()

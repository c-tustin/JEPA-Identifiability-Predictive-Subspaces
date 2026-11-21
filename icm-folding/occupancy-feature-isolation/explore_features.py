"""Exploratory moments; correlations are not promoted to causal criteria."""
import json
from pathlib import Path
import fresh_fit as F

D,S,R,torch,np,B=F.D,F.S,F.R,F.torch,F.D.np,F.B


def moments(n,f,a):
    mean=a*n.mean()+(1-a)*f.mean()
    var=a*(n-mean).square().mean()+(1-a)*(f-mean).square().mean()
    skew=(a*(n-mean).pow(3).mean()+(1-a)*(f-mean).pow(3).mean())/var.pow(1.5)
    kurt=(a*(n-mean).pow(4).mean()+(1-a)*(f-mean).pow(4).mean())/var.square()
    return dict(mean=float(mean),std=float(var.sqrt()),skew=float(skew),kurtosis=float(kurt))


def main():
    torch.set_num_threads(1);ds,c,initial,*_=F.setup();model=B.Encoder().double();model.load_state_dict(initial)
    values={}
    for s,data in ds.items():
        for region in ['near','far']:
            x=data[region].double();h=x[:,1]
            with torch.no_grad():z=model(x)
            values[s,region]=dict(height=h,log_distance=-h.log(),saturated_height=torch.tanh(4*h),initial_code=z)
    specs=[]
    for path in (S.OUT/'cells').glob('*.json'):
        i=json.loads(path.read_text());specs.append(('hybrid',i,i['near_source'],i['far_source'],i['alpha']))
    for folder,label in [(D.OUT,'dose'),(F.OUT,'fresh')]:
        for path in (folder/'cells').glob('*sigreg.json'):
            i=json.loads(path.read_text());specs.append((label,i,i['near'],i['far'],i['alpha']))
    rows=[]
    for label,i,n,f,a in specs:
        row=dict(group=label,name=i['name'],near=n,far=f,alpha=a,folded=i['folded'],material=i['material_fold'])
        row['moments']={k:moments(values[n,'near'][k],values[f,'far'][k],a) for k in values[n,'near']}
        hn,hf=values[n,'near']['height'],values[f,'far']['height']
        row['height_component']=dict(near_std=float(hn.std(unbiased=False)),far_std=float(hf.std(unbiased=False)),
            mean_gap=float(hn.mean()-hf.mean()),std_ratio=float(hn.std(unbiased=False)/hf.std(unbiased=False)))
        rows.append(row)
    R.trials.atomic_json(S.OUT/'exploratory_features.json',dict(exploratory=True,
        warning='Moment associations are not isolated distribution features; interventions are still required.',rows=rows,
        code_hash=R.digest(Path(__file__).read_bytes())))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,['height','saturated_height','initial_code']):
        for state,color in [(False,'tab:blue'),(True,'tab:red')]:
            q=[r for r in rows if r['material']==state]
            ax.scatter([abs(r['moments'][key]['skew']) for r in q],[r['moments'][key]['kurtosis'] for r in q],s=15,alpha=.6,c=color,label='Material fold' if state else 'No material fold')
        ax.set(title=key.replace('_',' '),xlabel='Absolute skewness',ylabel='Pearson kurtosis');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(S.OUT/'exploratory_moments.png',dpi=180);plt.close(fig)
    for r in rows:
        if r['group']=='fresh' and r['alpha']==.5:
            print(json.dumps(dict(source=r['near'],material=r['material'],height=r['moments']['height'],saturated=r['moments']['saturated_height'],initial=r['moments']['initial_code'])))


if __name__=='__main__':main()

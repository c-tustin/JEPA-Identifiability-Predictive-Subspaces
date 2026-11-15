"""Focused multimodal results and local ICM update mechanisms."""
import hashlib
import json
from pathlib import Path
import sys
from analyze import OUT,HERE,old,B
from torch.func import functional_call
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RED='#c13c45';TEAL='#147b79';ORANGE='#d48a18'


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def main():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    previous=HERE.parent/'output'
    methods=('ordinary','inverse_gradient','weighted_moments')
    labels=('Ordinary SIGReg','Divide gradients by occupancy','Weight SIGReg averages')
    colors=(RED,ORANGE,TEAL)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,(dist,title) in enumerate((('bimodal','Two peaks: distances 2 and 8'),('trimodal','Three peaks: distances 2, 5 and 12'))):
        r=old.trials.load(previous/'runs'/f'{dist}-ordinary-131.pt')
        centers=np.exp((np.arange(64)+.5)*np.log(16)/64)
        axes[0,col].plot(centers,r['mass'],c='#435774',lw=2)
        axes[0,col].set(title=title,ylabel='Probability per log-distance bin',xlabel='Physical distance')
        for method,label,color in zip(methods,labels,colors):
            run=old.trials.load(previous/'runs'/f'{dist}-{method}-131.pt')
            q=run['q'];y=run['frames'][-1]['y']
            sign=1 if np.cov(q,y)[0,1]>=0 else -1
            y=sign*(y-y.mean())/y.std()
            axes[1,col].plot(np.exp(q),y,c=color,lw=2,label=label)
            turns=run['metrics']['roots']
            if turns:axes[1,col].scatter(turns,np.interp(turns,np.exp(q),y),c=color,s=35,zorder=4)
        axes[1,col].set(xlabel='Physical distance',ylabel='Final learned code (display scale)')
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Change only SIGReg occupancy; keep alignment pairs fixed\nFinal curves shown for seed 131; dots mark remaining turns',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,'bimodal_trimodal_clear')

    record=old.trials.load(OUT/'icm-adam-components-131.pt')
    derivatives=[];curve_records=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fractions=np.linspace(0,1,51)
    for s in record['snapshots']:
        model=B.ICM().double()
        p={n:v.double().requires_grad_() for n,v in s['before'].items()}
        args=(s['x'].double(),s['xp'].double(),s['action'])
        e,_=functional_call(model,p,args)
        gg=torch.autograd.grad(e.mean(),tuple(p.values()),allow_unused=True)
        grad={n:torch.zeros_like(p[n]) if g is None else g for n,g in zip(p,gg)}
        dots={label:sum(float((grad[n]*d.double()).sum()) for n,d in direction.items()) for label,direction in s['directions'].items()}
        full={n:s['actual'][n].double()-s['before'][n].double() for n in p}
        dots['actual']=sum(float((grad[n]*full[n]).sum()) for n in p)
        derivatives.append(dict(update=s['update'],directional_derivatives=dots))
        if s['update'] not in (400,500,600):continue
        ax=axes[(s['update']-400)//100]
        with torch.no_grad():base=float(functional_call(model,{n:v.double() for n,v in s['before'].items()},args)[0].mean())
        curves={}
        for label,color,direction in (('Inverse-loss component',ORANGE,s['directions']['inverse']),
                 ('Forward-loss component',TEAL,s['directions']['forward']),('Actual full update',RED,full)):
            values=[]
            for fraction in fractions:
                # Match the original finite intervention's float32 parameter
                # addition, then evaluate its function in double precision.
                pars={n:(s['before'][n]+float(fraction)*direction[n].float()).double() for n in p}
                with torch.no_grad():values.append(float(functional_call(model,pars,args)[0].mean())-base)
            ax.plot(fractions,values,c=color,label=label,lw=2)
            curves[label]=values
        ax.axhline(0,c='black',lw=.7)
        ax.set(title=f'Update {s["update"]}',xlabel='Fraction of the saved update applied',ylabel='Change in forward prediction error')
        curve_records.append(dict(update=s['update'],fractions=fractions.tolist(),changes=curves))
        # Recheck the two named component interventions at full strength.
        for label,key in (('Inverse-loss component','inverse_component'),('Forward-loss component','forward_component')):
            expected=s['probes'][key]['forward_error']-s['probes']['before']['forward_error']
            assert np.isclose(curves[label][-1],expected,rtol=0,atol=1e-10)
    axes[0].legend(fontsize=8)
    fig.suptitle('Actual ICM Adam updates can increase forward error on the same fixed batch\nAbove zero = worse; small fractions distinguish an uphill direction from overshooting',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'actual_update_mechanism')
    old.trials.atomic_json(OUT/'directional-derivatives.json',dict(derivatives=derivatives,curves=curve_records))

    joint=old.trials.load(HERE.parent/'gradient-followup/output/joint-predictor-131.pt')
    frames=[f for f in joint['frames'] if 100<=f['update']<=600]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,key,title in zip(axes,('error','target_variance','normalized_error'),
        ('Raw prediction error rises','The target features spread out','Relative error initially improves')):
        ax.plot([f['update'] for f in frames],[f['fixed_grid'][key] for f in frames],'-o',c=TEAL)
        ax.set(title=title,xlabel='Actor/ICM updates',yscale='log')
    axes[0].set_ylabel('Forward squared error')
    axes[1].set_ylabel('Half-total target feature variance')
    axes[2].set_ylabel('Error / target variance')
    fig.suptitle('Same fixed physical distances and actions throughout · seed 131\nThe rise cannot be explained by changing occupancy alone',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.83));save(fig,'fixed_input_scale_and_error')
    summary=dict(seed=131,frames=[dict(update=f['update'],**f['fixed_grid']) for f in frames])
    old.trials.atomic_json(OUT/'fixed-input-summary.json',summary)
    html=['<!doctype html><meta charset="utf-8"><title>Mechanistic interpretation</title><style>body{font:18px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}img{max-width:100%}</style><h1>Mechanistic interpretation</h1><p><a href="../README.md">Explanation and derivations</a></p>']
    for name in ('bimodal_trimodal_clear','fixed_input_scale_and_error','actual_update_mechanism'):
        html.append(f'<h2>{name.replace("_"," ")}</h2><img src="{name}.png">')
    (OUT/'index.html').write_text('\n'.join(html))
    old.trials.atomic_json(OUT/'source-hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'analyze.py',HERE/'figures.py',Path(B.__file__))})


if __name__=='__main__':main()

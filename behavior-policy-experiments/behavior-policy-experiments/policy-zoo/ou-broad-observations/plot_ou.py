"""Readable complete-cohort OU comparison figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kempner-matplotlib')
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ou as run
import analyze_ou as analysis


def save(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(run.OUT/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    groups,missing=analysis.collect();assert not missing
    labels=['Original\nOU','Independent\nOU views','Independent\nbroad views']
    keys=['affine_log_r2','gaussian_r2','orthogonal_mse','goal_error']
    titles=['Recovery over distances 1–16','Recovery on Gaussian states','Recovery up to sign: MSE ↓','Physical goal error ↓']
    fig,axes=plt.subplots(1,4,figsize=(15,4.5))
    for ax,key,title in zip(axes,keys,titles):
        for i,seed in enumerate(run.SEEDS):
            ax.plot(range(3),[analysis.value(groups[a][i],key) for a in run.ARMS],color='gray',alpha=.16,lw=.6)
        means=[np.mean([analysis.value(r,key) for r in groups[a]]) for a in run.ARMS]
        ax.plot(range(3),means,'-o',color='#d65a21',lw=2)
        ax.set_xticks(range(3),labels,fontsize=8);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same OU alignment pairs; change which observations SIGReg sees\nGray lines retain all 60 paired seeds; orange is the mean.')
    save(fig,'ou_comparison')
    torch.set_num_threads(1)
    fig,axes=plt.subplots(2,3,figsize=(12,7));distance=np.geomspace(1,16,1025)
    target=np.log(distance/4)/run.original.config().sigma
    colors=['#397db5','#9471b5','#16856c']
    for ax,seed in zip(axes.flat,run.SEEDS[:6]):
        for arm,color,label in zip(run.ARMS,colors,['Original OU','Independent OU views','Broad views']):
            state=torch.load(run.OUT/f'encoder_{arm}_seed{seed}.pt',map_location='cpu')
            model=run.previous.behavior.Encoder();model.load_state_dict(state['encoder'])
            with torch.no_grad():y=model(run.previous.behavior.sensor(torch.tensor(distance,dtype=torch.float32),run.original.config())).numpy()
            sign=1 if np.cov(target,y)[0,1]>=0 else -1
            ax.plot(distance,sign*y,color=color,label=label)
        ax.plot(distance,target,'--',color='gray',label='True log-distance')
        ax.set(xscale='log',title=f'Seed {seed}',xlabel='Physical distance',ylabel='Output (sign aligned only)')
        ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First six OU runs: only a sign change is allowed; no fitted scale or offset')
    save(fig,'ou_representation_curves')
    (run.OUT/'FIGURES.md').write_text('# OU comparison figures\n\n![All paired runs](ou_comparison.png)\n\n![First six representations](ou_representation_curves.png)\n\n[Results](RESULTS.md)\n')


if __name__=='__main__':main()

"""Show folds during unconstrained MLP training, including later removal."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/icm-gradient-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT,trials

fig,axes=plt.subplots(1,2,figsize=(11,4))
r=trials.load(OUT/'hard-mining/131-c100.pt')
frames=[f for f in r['frames'] if f['step'] in (100,250,500,1000,4000)]
orient=1 if np.corrcoef(frames[-1]['q'],frames[-1]['y'])[0,1]>0 else -1
for f,color in zip(frames,plt.cm.viridis(np.linspace(.05,.95,len(frames)))):
    y=f['y'];normalized=orient*(y-y.mean())/max(y.std(),1e-8)
    axes[0].plot(np.exp(f['q']),normalized,color=color,label=f"Update {f['step']}")
    for root in f['roots']:
        val=np.interp(np.log(root),f['q'],normalized)
        axes[0].scatter(root,val,color=color,s=30,edgecolors='black',linewidths=.5)
axes[0].set(title='Source 131: a transient fold is removed',xlabel='Physical distance (evaluation only)',ylabel='Standardized code')
axes[0].legend(frameon=False,fontsize=8)
for seed in range(130,140):
    for kind,color in [('baseline','#b94348'),('mining','#147b79')]:
        path=OUT/'runs'/f'{seed}-sigreg-c0-n4000.pt' if kind=='baseline' else OUT/'hard-mining'/f'{seed}-c100.pt'
        r=trials.load(path);x=[];y=[]
        for f in r['frames']:
            if f['roots']:axes[1].scatter([f['step']]*len(f['roots']),f['roots'],color=color,s=9,alpha=.6)
            x.append(f['step']);y.append(min(f['roots']) if f['roots'] else np.nan)
        axes[1].plot(x,y,color=color,lw=.8,alpha=.4)
axes[1].plot([],[],color='#b94348',label='SIGReg baseline')
axes[1].plot([],[],color='#147b79',label='Global mining 100')
axes[1].set(title='Turning points across ten development sources',xlabel='Encoder updates',ylabel='Turning-point distance',yscale='log',ylim=(1,16),yticks=(1,2,4,8,16))
axes[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
axes[1].legend(frameon=False,fontsize=8)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'10_training_history.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
page=OUT/'index.html';page.write_text(page.read_text()+'<h2>Transient folds and training history</h2><img src="10_training_history.png">\n')
print('Saved training-history figure.')

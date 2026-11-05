"""Report all reflection controls; dataset means are the testing unit."""

import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/kempner-qualitative-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from run_crossed import LEVELS,OUT,behavior,load
from report_crossed import fold_flag,save


def main():
    torch.set_num_threads(1)
    c=behavior.Config()
    labels=('ordinary','remove_reflection','remove_random')
    scores,flags={},{}
    for arm in labels:
        matrix=np.zeros((5,5));folds=np.zeros((5,5),bool)
        for row,d in enumerate(LEVELS):
            for col,i in enumerate(LEVELS):
                path=(OUT/'crossed'/f'icm-data{d}-init{i}.pt' if arm=='ordinary' else OUT/'reflection-controls'/f'{arm}-data{d}-init{i}.pt')
                saved=load(path)
                matrix[row,col]=saved['metrics']['common_grid_r2']
                model=behavior.Encoder();model.load_state_dict(saved['encoder'])
                folds[row,col]=fold_flag(model,c)
        scores[arm]=matrix;flags[arm]=folds
    difference=scores['remove_reflection']-scores['remove_random']
    data_means=difference.mean(1)
    wins=int((data_means>1e-10).sum());losses=int((data_means<-1e-10).sum())
    p=1. if wins+losses==0 else float(stats.binomtest(wins,wins+losses,.5).pvalue)
    rng=np.random.default_rng(20261104)
    # Reuse one resampled set of columns for all sampled rows in a replicate.
    bootstrap=[]
    for _ in range(20000):
        r=rng.integers(0,5,5);col=rng.integers(0,5,5)
        bootstrap.append(float(difference[np.ix_(r,col)].mean()))
    result=dict(means={arm:float(scores[arm].mean()) for arm in labels},
                folds={arm:int(flags[arm].sum()) for arm in labels},
                matrices={arm:scores[arm].tolist() for arm in labels},
                reflection_minus_random_mean=float(difference.mean()),
                data_mean_differences=data_means.tolist(),data_wins=wins,data_losses=losses,
                exact_data_sign_p=p,crossed_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                scope='Post hoc 5-data × 5-init control; 25 cells are not independent trials. Bootstrap is exploratory with only five levels per factor.')
    (OUT/'reflection-control-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11,4.2),layout='constrained')
    for ax,arm in zip(axes,labels):
        im=ax.imshow(scores[arm],vmin=0,vmax=1,cmap='viridis')
        for r in range(5):
            for col in range(5):ax.text(col,r,f"{scores[arm][r,col]:.2f}"+(' F' if flags[arm][r,col] else ''),ha='center',va='center',fontsize=9,color='white' if scores[arm][r,col]<.5 else 'black')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=LEVELS,yticklabels=LEVELS,xlabel='Initialization seed',ylabel='ICM dataset seed',
               title={'ordinary':'Original alignment','remove_reflection':'Remove reflection alignment','remove_random':'Remove equal random alignment'}[arm])
    fig.colorbar(im,ax=axes,label='Shared-grid R²',shrink=.65)
    fig.suptitle('Does reflection alignment explain the learned fold?\nEvery SIGReg batch is unchanged. F = reversed grid step. All 25 matched combinations per condition are shown.')
    save(fig,'reflection_alignment_control')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

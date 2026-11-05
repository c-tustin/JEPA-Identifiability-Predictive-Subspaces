"""All crossed fits and targeted test results, with readable figures."""

import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_crossed import EVAL_SEED, LEVELS, OUT, POLICIES, ZOO_OUT, behavior, geometry, load, reference, controls


def save(fig, name):
    folder = OUT/"figures"
    folder.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder/f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fold_flag(model, c):
    """Use the zoo's original 1024-point fold definition exactly."""
    with torch.no_grad():
        z = torch.linspace(np.log(c.distance_min/4)/c.sigma, np.log(c.distance_max/4)/c.sigma, 1024)
        y = model(behavior.observe(z,c.sigma))
        sign = 1 if torch.cov(torch.stack([z,y]))[0,1] >= 0 else -1
        y = sign*(y-y.mean())/y.std(unbiased=False).clamp_min(1e-8)
        return bool((y.diff() < -1e-6).any())


def main():
    torch.set_num_threads(1)
    c = behavior.Config(**json.loads((ZOO_OUT/"config.json").read_text())["config"])
    runs = {(p,d,i):load(OUT/"crossed"/f"{p}-data{d}-init{i}.pt") for p in POLICIES for d in LEVELS for i in LEVELS}
    matrices, summary = {}, []
    fig, axes = plt.subplots(1,4,figsize=(14,4.3),layout="constrained")
    for ax, policy in zip(axes, POLICIES):
        score = np.array([[runs[policy,d,i]["metrics"]["common_grid_r2"] for i in LEVELS] for d in LEVELS])
        flags = np.zeros_like(score,dtype=bool)
        for row,d in enumerate(LEVELS):
            for col,i in enumerate(LEVELS):
                model = behavior.Encoder()
                model.load_state_dict(runs[policy,d,i]["encoder"])
                flags[row,col] = fold_flag(model,c)
                ax.text(col,row,f"{score[row,col]:.2f}"+(" F" if flags[row,col] else ""),
                        ha="center",va="center",fontsize=9,color="white" if score[row,col]<.5 else "black")
        image = ax.imshow(score,vmin=0,vmax=1,cmap="viridis",aspect="equal")
        ax.set(xticks=range(5),yticks=range(5),xticklabels=LEVELS,yticklabels=LEVELS,
               xlabel="Encoder initialization seed",ylabel="Dataset/minibatch seed",title=policy.upper() if policy in ("icm","ou") else policy.title())
        grand = score.mean()
        row_mean,col_mean = score.mean(1),score.mean(0)
        ss = dict(dataset=float(5*np.sum((row_mean-grand)**2)),
                  initialization=float(5*np.sum((col_mean-grand)**2)),
                  interaction=float(np.sum((score-row_mean[:,None]-col_mean[None,:]+grand)**2)))
        total = float(np.sum((score-grand)**2))
        assert abs(sum(ss.values())-total)<1e-12
        summary.append(dict(policy=policy,mean_r2=float(grand),r2_sd=float(score.std(ddof=1)),
                            folds=int(flags.sum()),ss=ss,total_ss=total,
                            variation_shares={k:v/total if total else 0. for k,v in ss.items()},
                            dataset_means=row_mean.tolist(),initialization_means=col_mean.tolist()))
        matrices[policy]=dict(r2=score.tolist(),folded=flags.tolist())
    fig.colorbar(image, ax=axes, label="Shared-grid R²",shrink=.65)
    fig.suptitle("Same datasets, different initializations: all 100 fits\nF = a reversed grid step. Rows fix collected pairs and batch indices; columns fix initial weights. No best-run selection.")
    save(fig,"crossed_data_initialization")

    fig, axes = plt.subplots(5,5,figsize=(14,11),layout="constrained")
    def display(curve,q):
        sign = 1 if np.cov(curve,q)[0,1]>=0 else -1
        return sign*(curve-curve.mean())/max(curve.std(),1e-8)
    for row,d in enumerate(LEVELS):
        for col,i in enumerate(LEVELS):
            ax=axes[row,col]
            run=runs['icm',d,i]
            q=run['final']['log_distance']
            for kind,color,style,label in [('initial','.55','--','Untrained'),('final','#D55E00','-','After ICM pairs')]:
                ax.plot(np.exp(q),display(run[kind]['curve'],q),color=color,ls=style,lw=1.8,label=label)
            ax.set_xscale('log');ax.set_xticks([1,2,4,8,16],['1','2','4','8','16'])
            ax.set_title(f"Data {d}, init {i} · R²={run['metrics']['common_grid_r2']:.2f}",fontsize=9)
            ax.grid(alpha=.15)
            if row==4:ax.set_xlabel('Physical distance')
            if col==0:ax.set_ylabel('Learned coordinate')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='outside lower center',frameon=False)
    fig.suptitle('ICM: does a fold come from the initial function or from training?\nEvery dataset × initialization is shown. Separate affine display normalization per curve; no unfolding.')
    save(fig,'icm_initial_and_final_curves')

    tests=json.loads((OUT/'targeted-tests.json').read_text())['families']
    associations=tests['per_seed_associations']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for r in associations:
        axes[0].plot([0,1],[-r['gaussian_rho'],r['entropy_rho']],color='.6',alpha=.6,marker='o',ms=3)
    axes[0].set(xticks=[0,1],xticklabels=['Gaussian shape\n−correlation with discrepancy','Coverage entropy\ncorrelation'],ylabel='Within-seed policy rank association',title='Gaussian shape tracks recovery more consistently')
    initial=json.loads((OUT/'initialization.json').read_text())
    for policy,color in [('icm','#D55E00'),('random','#0072B2'),('count','#009E73'),('ou','#CC79A7')]:
        final=[json.loads((ZOO_OUT/'runs'/f'pure-{policy}_seed{s}.json').read_text())['metrics']['common_grid_r2'] for s in initial['seeds']]
        axes[1].scatter(initial['r2'],final,label=policy,s=28,alpha=.8,color=color)
    axes[1].plot([0,1],[0,1],ls='--',color='.4')
    axes[1].set(xlabel='Truly untrained encoder R²',ylabel='Final R² after pure-policy training',title='Initialization is a head start, not a guarantee')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Existing 20-seed zoo: paired comparisons, not independent grid points\nAssociations do not identify a causal mechanism; points on the right share each seed’s initialization.')
    save(fig,'policy_shape_and_initialization')
    (OUT/'crossed-summary.json').write_text(json.dumps(dict(levels=LEVELS,matrices=matrices,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

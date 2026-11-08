"""Exact finite-dataset expected loss for candidate output vectors."""

import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import numpy as np
from scipy import special, stats
import torch

from run_crossed import LEVELS, OUT, POLICIES, behavior, load, sha, HERE


def objective(y, yp, regularizer, batch_endpoints=1024):
    """Pairs are iid draws with replacement; endpoints may be dependent.

    phi is the POOLED endpoint characteristic function, so source and
    successor marginals need not be identical for this identity to hold.
    """
    t=regularizer.frequency.numpy()
    target=regularizer.target.numpy()
    w=regularizer.weight.numpy()
    phase=np.concatenate([y,yp])[:,None]*t
    real,imag=np.cos(phase).mean(0),np.sin(phase).mean(0)
    phi_square=real**2+imag**2
    psi=np.cos((y-yp)[:,None]*t).mean(0)
    penalty=np.sum(w*(batch_endpoints*((real-target)**2+imag**2)+1+psi-2*phi_square))
    alignment=np.mean((y-yp)**2)
    return dict(alignment=float(alignment),sigreg=float(penalty),objective=float(.9*alignment+.1*penalty))


def gaussianize(x):
    # Average ranks retain exact ties; never split equal observations by index.
    return special.ndtri((stats.rankdata(x,method='average')-.5)/len(x))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    regularizer=behavior.SIGReg1D().double()
    rows=[]
    profiles=[]
    check=None
    centers=np.linspace(0,np.log(16),81)
    for policy in POLICIES:
        for seed in LEVELS:
            data=load(OUT/'sources'/f'{policy}-{seed}.pt')
            # q from z is diagnostic only; no learned encoder sees these labels.
            q=np.log(4)+.7*np.concatenate([data['z'].numpy(),data['zp'].numpy()])
            n=len(q)//2
            mono=gaussianize(q)
            baseline=objective(mono[:n],mono[n:],regularizer)
            candidates=[]
            for center in centers:
                y=gaussianize((q-center)**2)
                candidates.append(objective(y[:n],y[n:],regularizer))
            best=int(np.argmin([r['objective'] for r in candidates]))
            rows.append(dict(policy=policy,data_seed=seed,monotone=baseline,
                             best_grid_fold=candidates[best],best_center_distance=float(np.exp(centers[best])),
                             fold_minus_monotone=candidates[best]['objective']-baseline['objective'],
                             best_center_is_grid_boundary=bool(best in (0,len(centers)-1))))
            profiles.append(dict(policy=policy,data_seed=seed,losses=[r['objective'] for r in candidates],monotone_loss=baseline['objective']))
            if policy=='icm' and seed==8:
                rng=np.random.default_rng(20261104)
                sampled=[]
                for _ in range(512):
                    ix=rng.integers(0,n,512)
                    a,b=mono[:n][ix],mono[n:][ix]
                    sampled.append(.9*np.mean((a-b)**2)+.1*float(regularizer(torch.from_numpy(np.concatenate([a,b])))))
                se=float(np.std(sampled,ddof=1)/np.sqrt(len(sampled)))
                check=dict(analytic=baseline['objective'],sampled_mean=float(np.mean(sampled)),mc_se=se)
                assert abs(check['analytic']-check['sampled_mean'])<5*se+.001
            print(policy,seed,rows[-1]['best_center_distance'],rows[-1]['fold_minus_monotone'],flush=True)
    result=dict(candidate_protocol_sha256=sha(HERE/'PROXY_PROTOCOL.md'),rows=rows,profiles=profiles,
                center_distances=np.exp(centers).tolist(),formula_check=check,
                scope='Post hoc oracle rank-code comparators, fixed scale; exact expected loss of iid minibatch draws from the stored pair pool. Not fitted networks or global optima.')
    (OUT/'pair-objective.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained',sharey=True)
    for ax,policy in zip(axes,POLICIES):
        for profile in profiles:
            if profile['policy']==policy:
                ax.plot(np.exp(centers),np.asarray(profile['losses'])-profile['monotone_loss'],label=f"Data {profile['data_seed']}")
        ax.axhline(0,color='.4',ls='--')
        ax.set_xscale('log');ax.set_xticks([1,2,4,8,16],['1','2','4','8','16'])
        ax.set(title=policy,xlabel='Candidate turning distance')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Candidate expected loss − monotone-code loss')
    axes[-1].legend(frameon=False,fontsize=8)
    fig.suptitle('Does a data-only folded-code proxy explain the learned turn?\nBelow zero = this oracle fold beats the monotone quantile code. Same fixed output scale; not trained-network comparisons.')
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'pair_objective_proxy.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()

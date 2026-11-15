"""Do rare far views escape to extreme codes at high concentration?"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from guaranteed_models import old,dataset,target_measure,OUT,COMPARISON,key


def main():
    torch.set_num_threads(1);rows=[];diagnostic=None
    for seed in range(130,140):
        ds=dataset(seed);far=ds['d'].numpy()>=2
        for a in (.875,.9375):
            for tail in ('one','two'):
                r=old.trials.load(COMPARISON/'runs'/f'{key(seed,"weighted",a,tail)}.pt')
                model=old.ScaledEncoder(4).double();model.load_state_dict(r['encoder'])
                with torch.no_grad():h=model(ds['endpoints'].double()).numpy()
                mass=target_measure(ds,a,tail);p=mass[ds['bins'].numpy()]/ds['counts'][ds['bins']].numpy()
                near_p=p[~far]/p[~far].sum();far_p=p[far]/p[far].sum()
                rows.append(dict(seed=seed,level=a,tail=tail,
                    near_mean=float(near_p@h[~far]),near_abs_mean=float(near_p@abs(h[~far])),
                    far_mean=float(far_p@h[far]),far_abs_mean=float(far_p@abs(h[far])),
                    minimum_abs_far_code=float(abs(h[far]).min()),
                    far_fraction_abs_above3=float(far_p@(abs(h[far])>3)),
                    near_fraction_abs_above3=float(near_p@(abs(h[~far])>3))))
                if seed==131 and a==.9375 and tail=='one':diagnostic=(h[~far],near_p,h[far],far_p)
    reg=old.trials.behavior.SIGReg1D().double();t=reg.frequency.numpy();weight=reg.weight.numpy();target=reg.target.numpy()
    yn,pn,yf,pf=diagnostic;center=float(pf@yf);residual=yf-center
    near_cf=pn@np.exp(1j*yn[:,None]*t)
    far_cf=pf@np.exp(1j*residual[:,None]*t)
    far_dspread=pf@(1j*residual[:,None]*t*np.exp(1j*residual[:,None]*t))
    centers=np.linspace(-16,16,321);curves=[]
    for a in (.875,.9375):
        shifted=np.exp(1j*centers[:,None]*t)*far_cf
        total=a*near_cf+(1-a)*shifted
        error=total-target
        value=1024*np.sum(weight*abs(error)**2,axis=1)
        derivative=2*1024*np.sum(weight*np.real(np.conj(error)*(1-a)*1j*t*shifted),axis=1)
        self_dispersion=float(1024*(1-a)**2*2*np.sum(weight*np.real(np.conj(far_cf)*far_dspread)))
        phi=a*near_cf+(1-a)*np.exp(1j*center*t)*far_cf
        exact_at_center=float(2*1024*np.sum(weight*np.real(np.conj(phi-target)*(1-a)*1j*t*np.exp(1j*center*t)*far_cf)))
        # Independent finite-difference check of the implemented finite-grid loss.
        def loss(c):return float(1024*np.sum(weight*abs(a*near_cf+(1-a)*np.exp(1j*c*t)*far_cf-target)**2))
        eps=1e-5;fd=(loss(center+eps)-loss(center-eps))/(2*eps)
        assert np.isclose(fd,exact_at_center,rtol=1e-5,atol=1e-7)
        curves.append(dict(level=a,centers=centers.tolist(),discrepancy=value.tolist(),derivative=derivative.tolist(),
            actual_center=center,derivative_at_actual_center=exact_at_center,
            far_self_dispersion_derivative=self_dispersion,finite_difference=fd))
    assert np.isclose(curves[1]['far_self_dispersion_derivative']/curves[0]['far_self_dispersion_derivative'],.25)
    summary=[]
    for a in (.875,.9375):
        g=[r for r in rows if r['level']==a]
        summary.append(dict(level=a,n=len(g),mean_abs_far_code=float(np.mean([r['far_abs_mean'] for r in g])),
            far_fraction_abs_above3=float(np.mean([r['far_fraction_abs_above3'] for r in g])),
            minimum_abs_far_code=min(r['minimum_abs_far_code'] for r in g)))
    old.trials.atomic_json(OUT/'tail-code-summary.json',dict(summary=summary,runs=rows,translation_diagnostics=curves,
        note='Conditional output shapes held fixed from source-131 high-concentration model. Not a smooth-encoder or training intervention.'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for i,a in enumerate((.875,.9375)):
        vals=[r['far_abs_mean'] for r in rows if r['level']==a]
        axes[0].scatter(np.full(len(vals),i),vals,c=('#d48a18' if i==0 else '#147b79'),alpha=.7)
    axes[0].set(xticks=(0,1),xticklabels=('87.5% near','93.75% near'),ylabel='Mean absolute code on far observations',title='Rare far views get extreme codes')
    for r,c in zip(curves,('#d48a18','#147b79')):
        axes[1].plot(centers,np.array(r['discrepancy'])/1024,c=c,label=f"{100*r['level']:g}% near")
        axes[2].plot(centers,np.array(r['derivative'])/1024,c=c)
    for ax in axes[1:]:ax.axvline(center,c='gray',ls='--');ax.set_xlabel('Counterfactual center of far-code cluster')
    axes[1].set(title='Regularizer response to translating the tail',ylabel='SIGReg / endpoint multiplier');axes[1].legend(fontsize=8)
    axes[2].set(title='Translation force weakens far from the bulk',ylabel='Derivative of normalized discrepancy');axes[2].axhline(0,c='black',lw=.7)
    fig.suptitle('A candidate second failure mode: rare far observations parked at extreme codes\nRight panels hold conditional code shapes fixed; dashed line is the trained far-code mean',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.85))
    fig.savefig(OUT/'06_rare_tail_codes.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'06_rare_tail_codes.pdf',bbox_inches='tight');plt.close(fig)
    print(summary,flush=True);print('Actual far center and forces',center,[(r['level'],r['derivative_at_actual_center'],r['far_self_dispersion_derivative']) for r in curves],flush=True)


if __name__=='__main__':main()

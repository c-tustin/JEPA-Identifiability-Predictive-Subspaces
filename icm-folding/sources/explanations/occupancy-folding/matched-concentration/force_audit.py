"""How the fixed-state slope force depends on occupancy concentration."""
import json
import numpy as np
import torch
from torch.func import functional_call,jvp
from scipy.optimize import brentq
from run_comparison import OUT,old,dataset,target_measure,TAILS,metrics,EPS,SPLIT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(1)
    ds=dataset(131);c=ds['c'];data=ds['data'];n=len(data['x'])
    saved=old.trials.load(OUT/'runs/131-native_paired.pt')
    reg=old.trials.behavior.SIGReg1D().double()
    xx=ds['endpoints'].double();b=ds['bins'].numpy();count=ds['counts'].numpy()
    concentrations=np.linspace(.1,.98,45)
    records=[]
    for step in (100,250):
        model=old.ScaledEncoder(4).double();model.load_state_dict(saved['states'][step])
        parameters={name:p.detach().clone().requires_grad_() for name,p in model.named_parameters()}
        def slope_at(qvalue):
            q=torch.tensor([qvalue],dtype=torch.float64,requires_grad=True)
            y=functional_call(model,parameters,(old.trials.behavior.sensor(q.exp(),c),))
            slope=torch.autograd.grad(y.sum(),q,create_graph=True)[0]
            return q,slope
        probes=[('far slope',np.log(15.8))]
        if step==250:
            grid=np.linspace(0,np.log(16),1025)
            vals=[float(slope_at(q)[1].detach()) for q in grid]
            crossing=np.flatnonzero(np.array(vals[:-1])*np.array(vals[1:])<0)
            assert len(crossing)==1
            i=crossing[0];turn=brentq(lambda q:float(slope_at(q)[1].detach()),grid[i],grid[i+1],xtol=1e-12)
            probes.append(('native turning slope',turn))
        for label,qvalue in probes:
            q,slope=slope_at(qvalue)
            curvature=float(torch.autograd.grad(slope.sum(),q,retain_graph=True)[0])
            gs=torch.autograd.grad(slope.sum(),tuple(parameters.values()),allow_unused=True)
            v={name:torch.zeros_like(parameters[name]) if g is None else g.detach() for name,g in zip(parameters,gs)}
            h,dh=jvp(lambda pars:functional_call(model,pars,(xx,)),(parameters,),(v,))
            h=h.detach();dh=dh.detach()
            phase=h[:,None]*reg.frequency
            co=phase.cos();si=phase.sin()
            dco=-si*reg.frequency*dh[:,None]
            dsi=co*reg.frequency*dh[:,None]
            align_projection=float(.9*2*((h[:n]-h[n:])*(dh[:n]-dh[n:])).mean())
            for tail in TAILS:
                projections=[]
                for a in concentrations:
                    mass=target_measure(ds,float(a),tail)
                    pp=torch.from_numpy(mass[b]/count[b])[:,None]
                    C=(pp*co).sum(0);S=(pp*si).sum(0)
                    dC=(pp*dco).sum(0);dS=(pp*dsi).sum(0)
                    projection=float(.1*2*1024*(reg.weight*((C-reg.target)*dC+S*dS)).sum())
                    projections.append(projection)
                coefficients=np.polyfit(concentrations,projections,2)
                residual=float(np.max(abs(np.polyval(coefficients,concentrations)-projections)))
                assert residual<1e-8*max(1.,max(abs(np.array(projections))))
                # Independent direct reverse-mode check at the decisive mass.
                mass=target_measure(ds,.875,tail)
                pp=torch.from_numpy(mass[b]/count[b])[:,None]
                direct_h=functional_call(model,parameters,(xx,));ph=direct_h[:,None]*reg.frequency
                C=(pp*ph.cos()).sum(0);S=(pp*ph.sin()).sum(0)
                loss=.1*1024*(reg.weight*((C-reg.target).square()+S.square())).sum()
                grad=torch.autograd.grad(loss,tuple(parameters.values()))
                reverse=sum(float((g*v[name]).sum()) for name,g in zip(parameters,grad))
                predicted=float(np.polyval(coefficients,.875))
                assert np.isclose(reverse,predicted,rtol=1e-9,atol=1e-9),(step,label,tail,reverse,predicted)
                records.append(dict(step=step,probe=label,distance=float(np.exp(qvalue)),raw_slope=float(slope.detach()),
                    curvature=curvature,tail=tail,concentrations=concentrations.tolist(),
                    regularizer_slope_change_per_lr=(-np.array(projections)).tolist(),
                    full_slope_change_per_lr=(-np.array(projections)-align_projection).tolist(),
                    regularizer_gradient_polynomial=coefficients.tolist(),quadratic_residual=residual,
                    direct_reverse_projection=reverse,polynomial_projection=predicted,
                    alignment_projection=align_projection))
    old.trials.atomic_json(OUT/'local-force-audit.json',records)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    from report_comparison import COLORS,LABELS
    for ax,(step,probe) in zip(axes,((100,'far slope'),(250,'far slope'),(250,'native turning slope'))):
        for r in records:
            if r['step']==step and r['probe']==probe:
                ax.plot(100*np.array(r['concentrations']),r['full_slope_change_per_lr'],c=COLORS[r['tail']],label=LABELS[r['tail']])
        ax.axhline(0,c='black',lw=.7)
        ax.set(title=f'Update {step}: {probe}',xlabel='Probability below distance 2 (%)',ylabel='Predicted raw-slope change / learning rate')
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed weights: concentration can reverse the slope pressure; its threshold changes with the weights\nPopulation-gradient diagnostic; it does not replay Adam or predict the full training trajectory',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.86))
    fig.savefig(OUT/'07_local_slope_force.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'07_local_slope_force.pdf',bbox_inches='tight');plt.close(fig)
    print('Verified 12 quadratic slope-force curves against direct autograd.',flush=True)


if __name__=='__main__':main()

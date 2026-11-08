"""Locate which observations contribute to the current regularizer gradient.

The chain-rule coefficients are computed from the entire actual batch, then
split by distance. We do not apply a different SIGReg loss independently to bins.
These are current-gradient contributions, not a decomposition of all past Adam.
"""
import json
import numpy as np
import torch
from torch.func import functional_call,jvp
import common_gc as study


def main():
    torch.set_num_threads(1)
    rows=[];arrays={};max_error=0.
    for path in sorted((study.OUT/'snapshots').glob('*.pt')):
        saved=torch.load(path,map_location='cpu')
        name,seed,step=saved['condition'],saved['seed'],saved['update']
        phase='early' if name.endswith('Rearly') else 'late'
        pool=study.pooled(study.factor_data(seed,0 if phase=='early' else 128))
        inputs=pool[saved['regularizer_indices']]
        model=study.behavior.Encoder();model.load_state_dict(saved['before'])
        named=list(model.named_parameters());parameters=[p for _,p in named]
        y=model(inputs);loss=.1*study.behavior.SIGReg1D()(y)
        coefficients=torch.autograd.grad(loss,y,retain_graph=True)[0].detach()
        distance=1/inputs[:,1]
        masks={'near':distance<2,'middle':(distance>=2)&(distance<4),'far':distance>=4}
        gradients={}
        for key,mask in masks.items():
            values=torch.autograd.grad(y,parameters,grad_outputs=coefficients*mask,retain_graph=True)
            gradients[key]={n:g.detach() for (n,_),g in zip(named,values)}
        error=max(float((sum(gradients[b][n] for b in masks)-saved['attribution']['gradient_regularizer'][n]).abs().max()) for n,_ in named)
        max_error=max(max_error,error)
        scale=max(float(g.abs().max()) for g in saved['attribution']['gradient_regularizer'].values())
        assert error<1e-5*(1+scale)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(saved['optimizer_before'])
        lr=study.controls.learning_rate(step,study.original.config().jepa_steps)
        optimizer.param_groups[0]['lr']=lr
        for n,p in named:p.grad=saved['attribution']['gradient_alignment'][n]+saved['attribution']['gradient_regularizer'][n]
        optimizer.step()
        beta1,beta2=optimizer.param_groups[0]['betas'];eps=optimizer.param_groups[0]['eps']
        q=torch.linspace(0,np.log(16),257)
        def evaluate(weights):
            def forward(log_distance):return functional_call(model,weights,(study.behavior.sensor(log_distance.exp(),study.original.config()),))
            return jvp(forward,(q,),(torch.ones_like(q),))
        before=saved['before'];initial_curve=saved['attribution']['curve_before']
        sign=1 if np.cov(q.numpy(),initial_curve)[0,1]>=0 else -1
        entry={'condition':name,'seed':seed,'update':step,'gradient_reconstruction_error':error,
               'relative_gradient_reconstruction_error':error/(1+scale),'regions':{}}
        for region in masks:
            direction={}
            for n,p in named:
                state=optimizer.state[p];t=int(state['step'])
                denominator=(state['exp_avg_sq']/(1-beta2**t)).sqrt()+eps
                direction[n]=-lr*(1-beta1)*gradients[region][n]/(1-beta1**t)/denominator
            _,(dy,ds)=jvp(evaluate,(before,),(direction,))
            ds=sign*ds.detach().numpy()
            values={'sample_count':int(masks[region].sum()),
                    'gradient_norm':float(torch.cat([g.flatten() for g in gradients[region].values()]).norm())}
            for query,mask in [('near',q.numpy()<np.log(2)),('middle',(q.numpy()>=np.log(2))&(q.numpy()<np.log(4))),('far',q.numpy()>=np.log(4))]:
                values['mean_slope_impulse_at_'+query]=float(ds[mask].mean())
            entry['regions'][region]=values
            arrays[f'{name}_seed{seed}_update{step}_{region}']=ds
        rows.append(entry)
    study.save_json(study.OUT/'GRADIENT_REGIONS.json',{'rows':rows,'max_gradient_reconstruction_error':max_error,
        'max_relative_gradient_reconstruction_error':max(r['relative_gradient_reconstruction_error'] for r in rows),
        'scope':'Full-batch chain-rule coefficients split by source distance; projected through the actual Adam denominator and local network Jacobian. Current impulse only, not all accumulated regularizer history. No added significance tests.'})
    np.savez_compressed(study.OUT/'gradient_region_impulses.npz',distance=np.exp(q.numpy()),**arrays)
    print('Region gradient snapshots:',len(rows),'max reconstruction error',max_error)


if __name__=='__main__':main()

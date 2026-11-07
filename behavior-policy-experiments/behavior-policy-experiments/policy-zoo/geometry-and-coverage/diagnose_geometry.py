"""Marginal geometry pressure and expected implemented losses on fixed pools."""
import json
import numpy as np
from scipy.special import ndtri
import torch
import common_gc as study


@torch.no_grad()
def expected_losses(model,data,pool):
    y=model(data['x']).double().numpy()
    yp=model(data['xp']).double().numpy()
    outputs=model(pool).double().numpy()
    regularizer=study.behavior.SIGReg1D().double()
    t=regularizer.frequency.numpy();target=regularizer.target.numpy();weight=regularizer.weight.numpy()
    phase=outputs[:,None]*t
    real=np.cos(phase).mean(0);imag=np.sin(phase).mean(0)
    squared=real**2+imag**2
    discrepancy=float(np.sum(weight*((real-target)**2+imag**2)))
    # Exact expected batch-size-scaled CF loss for B iid pooled observations.
    expected=1024*discrepancy+float(np.sum(weight*(1-squared)))
    alignment=float(np.mean((y-yp)**2))
    return {'alignment':alignment,'marginal_cf_discrepancy':discrepancy,
            'expected_sigreg_iid_1024':expected,'expected_total':.9*alignment+.1*expected,
            'pooled_output_mean':float(outputs.mean()),'pooled_output_std':float(outputs.std())}


def monotone_reference(pool,distance):
    q=torch.log(1/pool[:,1]).double().numpy()
    q=np.sort(q)
    # Average ranks retain duplicate observations rather than invent an order.
    values,first,counts=np.unique(q,return_index=True,return_counts=True)
    probabilities=(first+(counts-1)/2+.5)/len(q)
    curve=np.interp(np.log(distance),values,ndtri(probabilities))
    return curve,{'observed_min_distance':float(np.exp(values[0])),
                  'observed_max_distance':float(np.exp(values[-1])),
                  'scope':'Monotone empirical Gaussian-quantile comparator; constant outside observed support. Not a trained network or global optimizer.'}


def main():
    torch.set_num_threads(1)
    rows=[];quantile_arrays={}
    for seed in study.FACTOR_SEEDS:
        parent=torch.load(study.prior.OUT/f'encoder_icm_age000_seed{seed}.pt',map_location='cpu')
        before=study.behavior.Encoder();before.load_state_dict(parent['encoder']);before.eval()
        for a in ('early','late'):
            data=study.factor_data(seed,0 if a=='early' else 128)
            for r in ('early','late'):
                name=f'factor_A{a}_R{r}'
                path=study.OUT/f'{name}_seed{seed}.json'
                if not path.exists():continue
                pool=study.pooled(study.factor_data(seed,0 if r=='early' else 128))
                final=study.behavior.Encoder()
                final.load_state_dict(torch.load(study.OUT/f'encoder_{name}_seed{seed}.pt',map_location='cpu')['encoder']);final.eval()
                arrays=np.load(study.OUT/f'{name}_seed{seed}.npz')
                curve,support=monotone_reference(pool,arrays['distance'])
                quantile_arrays[f'{name}_seed{seed}']=curve
                rows.append({'condition':name,'seed':seed,'before':expected_losses(before,data,pool),
                             'after':expected_losses(final,data,pool),'monotone_reference_support':support})
    study.save_json(study.OUT/'MARGINAL_DIAGNOSTICS.json',{'rows':rows,
        'formula':'E SIGReg_B = B*sum w*|phi-target|^2 + sum w*(1-|phi|^2), for B=1024 iid observations from fixed pool.',
        'scope':'Full-pool expected finite-batch losses; no new significance tests.'})
    if rows:
        np.savez_compressed(study.OUT/'monotone_references.npz',distance=arrays['distance'],**quantile_arrays)
    print('Full-pool diagnostics:',len(rows))


if __name__=='__main__':main()

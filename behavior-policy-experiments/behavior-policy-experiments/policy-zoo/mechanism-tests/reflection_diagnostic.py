"""Check the reflecting-boundary alias suggested by the post hoc loss proxy."""

import json
import numpy as np
from run_crossed import LEVELS,OUT,POLICIES,behavior,load
from pair_objective import gaussianize,objective


def main():
    regularizer=behavior.SIGReg1D().double()
    rows=[]
    for policy in POLICIES:
        for seed in LEVELS:
            data=load(OUT/'sources'/f'{policy}-{seed}.pt')
            # Read physical coordinates only for this oracle diagnostic.
            d=1/data['x'][:,1].double().numpy()
            dp=1/data['xp'][:,1].double().numpy()
            action=data['action'].numpy()
            near=(action==0)&(d<1.25)
            far=(action==2)&(d>15.75)
            mono=gaussianize(np.concatenate([d,dp]))
            mirror=gaussianize((np.concatenate([d,dp])-1.125)**2)
            n=len(d)
            alignment_gain=(mono[:n]-mono[n:])**2-(mirror[:n]-mirror[n:])**2
            identity_error=float(np.max(np.abs((d[near]-1.125)**2-(dp[near]-1.125)**2))) if near.any() else None
            if near.any():assert identity_error<1e-6
            rows.append(dict(policy=policy,data_seed=seed,near_reflection_fraction=float(near.mean()),
                far_reflection_fraction=float(far.mean()),mirror_identity_max_error=identity_error,
                monotone=objective(mono[:n],mono[n:],regularizer),mirror=objective(mirror[:n],mirror[n:],regularizer),
                alignment_gain_near_weighted=float(np.mean(alignment_gain*near)),
                alignment_gain_other_weighted=float(np.mean(alignment_gain*(~near)))))
    (OUT/'reflection-diagnostic.json').write_text(json.dumps(dict(
        post_hoc=True,center=1.125,rows=rows,
        identity="At an approach reflection, d_next = 2.25 - d, hence (d_next-1.125)^2=(d-1.125)^2.",
        scope="Oracle rank-code diagnostic on the stored data, not the learned encoder or a prediction of its root. Physical policies only have actual reflections; OU action=-1."),indent=2)+'\n')
    for p in POLICIES:
        selected=[r for r in rows if r['policy']==p]
        print(p,'near reflection',np.mean([r['near_reflection_fraction'] for r in selected]),
              'loss gain',np.mean([r['monotone']['objective']-r['mirror']['objective'] for r in selected]),flush=True)


if __name__=='__main__':main()

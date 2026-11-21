"""Audit exact-risk comparisons and every restricted-critic probe."""
import json
from pathlib import Path
import critic_probe as P
import mechanism as M

D,S,R,torch,np,B=P.D,P.S,P.R,P.torch,P.np,P.B


def main():
    torch.set_num_threads(1);ds,c,_,_,_,mean,std,_=D.setup()
    risk=json.loads((S.OUT/'mechanism.json').read_text())
    assert risk['code_hash']==R.digest((S.HERE/'mechanism.py').read_bytes())
    assert risk['protocol_hash']==R.digest((S.HERE/'MECHANISM_PROTOCOL.md').read_bytes())
    assert risk['checks']==M.checks()
    reg=B.SIGReg1D().double();risk_checks=[]
    for template in risk['templates']:
        n,f=template['near'],template['far']
        for cand in template['candidates']:
            a=cand['trained_mass'];path=D.OUT/'runs'/f'n{n}-f{f}-a{a:g}-sigreg.pt'
            assert cand['checkpoint_hash']==R.digest(path.read_bytes())
            m=B.Encoder().double();m.load_state_dict(R.trials.load(path)['model'])
            with torch.no_grad():
                yn=m(ds[n]['near'].double());yf=m(ds[f]['far'].double())
                align=(m(ds[131]['data']['x'].double())-m(ds[131]['data']['xp'].double())).square().mean()
                for target in [.33,.67,.9]:
                    pn=yn[:,None]*reg.frequency;pf=yf[:,None]*reg.frequency
                    cc=target*pn.cos().mean(0)+(1-target)*pf.cos().mean(0)
                    ss=target*pn.sin().mean(0)+(1-target)*pf.sin().mean(0)
                    direct=float(.9*align+.1*((1023*(cc.square()+ss.square())-2048*reg.target*cc+1024*reg.target.square()+1)*reg.weight).sum())
                    error=abs(direct-float(np.polyval(cand['objective_quadratic'],target)))
                    assert error<1e-9;risk_checks.append(error)
    paths=sorted((P.OUT/'runs').glob('*.pt'));assert len(paths)==54
    test_u=torch.rand((64,256),dtype=torch.float64,generator=torch.Generator().manual_seed(8700131))
    rows=[];hashes={}
    for path in paths:
        s=R.trials.load(path);i=s['identity'];t=i['template'];a=i['target_mass'];train_a=i['encoder_trained_mass']
        ep=D.OUT/'runs'/f'n{t}-f{t}-a{train_a:g}-sigreg.pt'
        assert i['encoder_checkpoint_hash']==R.digest(ep.read_bytes())
        assert i['source_hash']==ds[t]['source_hash']
        assert i['code_hash']==R.digest((S.HERE/'critic_probe.py').read_bytes())
        assert i['protocol_hash']==R.digest((S.HERE/'MECHANISM_PROTOCOL.md').read_bytes())
        torch.manual_seed(i['critic_seed']);critic=D.N.critic()
        assert i['initial_critic_hash']==R.state_hash(critic.state_dict())==R.state_hash(s['initial_critic'])
        x=torch.cat([ds[t]['near'],ds[t]['far']]);n=len(ds[t]['near'])
        enc=B.Encoder();enc.load_state_dict(R.trials.load(ep)['model'])
        with torch.no_grad():z=enc(x)
        zm=a*z[:n].mean()+(1-a)*z[n:].mean()
        zs=(a*z[:n].square().mean()+(1-a)*z[n:].square().mean()-zm.square()).clamp_min(1e-12).sqrt()
        assert torch.equal(zm,s['code_mean']) and torch.equal(zs,s['code_std'])
        assert torch.equal(mean,s['raw_mean']) and torch.equal(std,s['raw_std'])
        ids=S.indices(a,n,len(x)-n,test_u);assert i['test_plan_hash']==R.digest(ids.numpy().tobytes())
        critic.load_state_dict(s['critic']);v=P.evaluate(critic,(x-mean)/std,((z-zm)/zs).detach(),ids)
        assert v==s['validation']
        rows.append(dict(**i,validation=v,late_validation_change=s['frames'][-1]['validation']['mean']-s['frames'][-3]['validation']['mean']))
        hashes[path.name]=R.digest(path.read_bytes())
    contrasts=[]
    for t in [130,136]:
        for target in P.MASSES:
            for seed in P.SEEDS:
                select=lambda a:next(r['validation']['mean'] for r in rows if (r['template'],r['target_mass'],r['critic_seed'],r['encoder_trained_mass'])==(t,target,seed,a))
                contrasts.append(dict(template=t,target_mass=target,critic_seed=seed,
                    folded_minus_ordered_low=select(.75)-select(.5),folded_minus_ordered_high=select(.75)-select(.9375)))
    gaps=[r[k] for r in contrasts for k in ['folded_minus_ordered_low','folded_minus_ordered_high']]
    result=dict(critic_fits=54,pairwise_comparisons=36,folded_worse_comparisons=sum(x>0 for x in gaps),
        minimum_gap=min(gaps),maximum_gap=max(gaps),mean_gap=float(np.mean(gaps)),
        maximum_late_validation_change=max(abs(r['late_validation_change']) for r in rows),contrasts=contrasts,rows=rows)
    R.trials.atomic_json(P.OUT/'summary.json',result)
    R.trials.atomic_json(P.OUT/'verification.json',dict(passed=True,critic_checkpoints=54,
        final_validation_exact_after_reload=True,normalizers_initializers_and_source_hashes_verified=True,
        exact_risk_formula_max_error=max(risk_checks),checkpoint_hashes=hashes,
        report_hash=R.digest(Path(__file__).read_bytes())))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,t in zip(axes,[130,136]):
        for a in P.MASSES:
            means=[];lo=[];hi=[]
            for target in P.MASSES:
                vals=[r['validation']['mean'] for r in rows if (r['template'],r['target_mass'],r['encoder_trained_mass'])==(t,target,a)]
                means.append(np.mean(vals));lo.append(min(vals));hi.append(max(vals))
            ax.plot(P.MASSES,means,'o-',label=f'Encoder trained at {a:g}'+(' (folded)' if a==.75 else ' (ordered grid)'))
            ax.fill_between(P.MASSES,lo,hi,alpha=.15)
        ax.set(title=f'Near/far profile {t}/{t}',xlabel='Target near mass',ylabel='Validation InfoNCE loss')
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(P.OUT/'critic_comparison.png',dpi=180);plt.close(fig)
    text=f'''# Objective mechanism: exact risks and fixed-encoder critic checks

The finite-iid-batch SIGReg risk formula agrees exactly with an enumerated
two-atom/two-draw example, and with direct moment calculations for all twelve
fixed candidates at three additional mixture masses (maximum error
{max(risk_checks):.3g}). Each candidate's risk is quadratic in near mass.

Comparing the candidates trained at masses 0.5,0.75,0.9375 yields crossings
between the low-mass ordered and middle-mass folded candidate at
0.5834–0.5977, and between that folded candidate and the high-mass ordered
candidate at 0.8825–0.9110 across four conditional profiles. This supplies an
objective-level reason for an intermediate mass window: the relative
near/far weights change the relative Gaussian-matching costs. All five old
substantial-fold masses lie inside every one of these candidate windows;
the five original nonfolded masses lie above every window.

This comparison is restricted to three fitted candidates per profile. It is
not a proof of global optimality or exact training-branch prediction. In
particular the dose study has an ordered 87.5% case despite the three-candidate
ranking preferring the folded candidate there; optimizer path and additional
candidate solutions matter near boundaries.

For raw/own-code InfoNCE, identical codes give identical critic score rows.
With two distinct positive labels the average loss is at least log 2. The
checked two-point example has loss 0.693147 at collision, 0.598139 after a
small correctly associated separation, and 0.798139 with the reversed
association. SIGReg assigns exactly the same loss to the two permutations.
This is a structural collision-cost example, not an injectivity certificate
for finite real-data fits.

All 54 frozen-encoder critic probes passed independent checkpoint reload and
validation-loss reconstruction. At two profiles, three target masses and three
critic initializations, the middle-mass folded encoder has higher InfoNCE
loss than both ordered candidates in **{sum(x>0 for x in gaps)}/36** paired comparisons.
The excess ranges from {min(gaps):.4f} to {max(gaps):.4f} loss units.
Codes are affinely normalized using the target distribution to control
critic input conditioning; raw normalization, critic class, update budget,
initial states and candidate plans are paired. The maximum absolute change
between the last and third-last validation checkpoints is
{result['maximum_late_validation_change']:.4f}; these finite optimizations do
not prove the critics reach their global optima.

Together these tests support a specific mechanism: intermediate near/tail
mass can favor a folded Gaussian-matching solution in the tested model,
whereas raw/own-code discrimination penalizes its ambiguous association with
observations and can retain a non-Gaussian ordered code. They do not establish
that InfoNCE can never fold, that occupancy alone determines every optimizer
path, or that temporal InfoNCE has already been tested by this raw/code probe.
New initializer/plan replication and new-source predictions remain pending.

[Exact-risk plots](../mechanism_risks.png) · [Exact-risk values](../mechanism.json) ·
[Critic comparisons](critic_comparison.png) · [All critic measurements](summary.json) ·
[Audit](verification.json) · [Protocol](../../MECHANISM_PROTOCOL.md)
'''
    (P.OUT/'RESULTS.md').write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','contrasts']},indent=2))


if __name__=='__main__':main()

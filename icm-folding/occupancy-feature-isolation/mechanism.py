"""Exact finite-batch SIGReg risk and a checked contrastive collision example."""
import itertools
import json
from pathlib import Path
import dose as D

S,R,torch,np,B=D.S,D.R,D.torch,D.np,D.B


def risk_coefficients(near,far,reg,N=1024):
    pn=near[:,None]*reg.frequency;pf=far[:,None]*reg.frequency
    cn,sn=pn.cos().mean(0),pn.sin().mean(0)
    cf,sf=pf.cos().mean(0),pf.sin().mean(0)
    dc,ds=cn-cf,sn-sf;g=reg.target;w=reg.weight
    a=((N-1)*(dc.square()+ds.square())*w).sum()
    b=((2*(N-1)*(cf*dc+sf*ds)-2*N*g*dc)*w).sum()
    c=(((N-1)*(cf.square()+sf.square())-2*N*g*cf+N*g.square()+1)*w).sum()
    return np.array([float(a),float(b),float(c)])


def checks():
    reg=B.SIGReg1D().double();values=torch.tensor([-.7,1.4],dtype=torch.float64)
    probs=[.3,.7];N=2;actual=0.
    for a,b in itertools.product(range(2),repeat=2):actual+=probs[a]*probs[b]*float(reg(values[[a,b]]))
    coeff=risk_coefficients(values[:1],values[1:],reg,N=2)
    expected=float(np.polyval(coeff,.3));assert abs(actual-expected)<1e-12
    raw=torch.tensor([[-1.],[1.]],dtype=torch.float64);code=torch.zeros(2,dtype=torch.float64)
    identity=torch.nn.Identity()
    collision=float(D.N.objective(raw,code,identity));assert abs(collision-np.log(2))<1e-12
    eps=.01;right=torch.tensor([-eps,eps],dtype=torch.float64);wrong=-right
    right_loss=float(D.N.objective(raw,right,identity));wrong_loss=float(D.N.objective(raw,wrong,identity))
    assert right_loss<collision<wrong_loss
    assert float(reg(right))==float(reg(wrong))
    return dict(finite_batch_expected_risk_enumeration=dict(exact=actual,formula=expected),
        two_point_collision_loss=collision,separated_correct=right_loss,separated_reversed=wrong_loss,
        sigreg_same_for_permuted_codes=True,
        collision_excess_curve=[dict(alpha=float(a),cross_pair_probability=float(2*a*(1-a)),
             ideal_two_candidate_excess_cost=float(2*a*(1-a)*np.log(2))) for a in D.MASSES])


def main():
    torch.set_num_threads(1);check=checks();ds,c,_,_,_,_,_,_=D.setup()
    reg=B.SIGReg1D().double();results=[]
    for n,f in itertools.product([130,136],repeat=2):
        candidates=[]
        for alpha in [.5,.75,.9375]:
            path=D.OUT/'runs'/f'n{n}-f{f}-a{alpha:g}-sigreg.pt'
            saved=R.trials.load(path);m=B.Encoder().double();m.load_state_dict(saved['model'])
            with torch.no_grad():
                yn=m(ds[n]['near'].double());yf=m(ds[f]['far'].double())
                data=ds[131]['data'];ya=m(data['x'].double());yp=m(data['xp'].double())
                alignment=float((ya-yp).square().mean())
            coeff=.1*risk_coefficients(yn,yf,reg);coeff[-1]+=.9*alignment
            candidates.append(dict(trained_mass=alpha,trained_geometry=saved['geometry'],
                objective_quadratic=coeff.tolist(),alignment=alignment,
                near_output=dict(mean=float(yn.mean()),std=float(yn.std()),min=float(yn.min()),max=float(yn.max())),
                far_output=dict(mean=float(yf.mean()),std=float(yf.std()),min=float(yf.min()),max=float(yf.max()),
                                abs_above3=float((yf.abs()>3).double().mean()),
                                within_near_range=float(((yf>=yn.min())&(yf<=yn.max())).double().mean())),
                checkpoint_hash=R.digest(path.read_bytes())))
        crossings=[]
        for a,b in itertools.combinations(candidates,2):
            roots=np.roots(np.array(a['objective_quadratic'])-np.array(b['objective_quadratic']))
            crossings.append(dict(a=a['trained_mass'],b=b['trained_mass'],
                roots=[float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<=r.real<=1]))
        grid=np.linspace(.2,.99,159)
        vals=np.array([np.polyval(x['objective_quadratic'],grid) for x in candidates])
        results.append(dict(near=n,far=f,candidates=candidates,crossings=crossings,
                            alpha_grid=grid.tolist(),risks=vals.tolist()))
    result=dict(checks=check,templates=results,
        code_hash=R.digest(Path(__file__).read_bytes()),protocol_hash=R.digest((S.HERE/'MECHANISM_PROTOCOL.md').read_bytes()),
        scope='Exact expected iid finite-batch risk for three fixed trained candidates per template; no global-optimum or full-training forecast claim.')
    R.trials.atomic_json(S.OUT/'mechanism.json',result)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,r in zip(axes.ravel(),results):
        for cand,risk in zip(r['candidates'],r['risks']):
            ax.plot(r['alpha_grid'],risk,label=f'Trained mass {cand["trained_mass"]:g}'+(' (fold)' if cand['trained_geometry']['folded'] else ' (ordered grid)'))
        ax.set(title=f'Near profile {r["near"]}, far profile {r["far"]}',xlabel='Target near mass',ylabel='Expected original objective')
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(S.OUT/'mechanism_risks.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(checks=check,crossings=[dict(near=r['near'],far=r['far'],crossings=r['crossings']) for r in results])))


if __name__=='__main__':main()

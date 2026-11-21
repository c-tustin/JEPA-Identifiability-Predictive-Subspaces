"""Shared SIGReg training for explicitly specified endpoint sampling plans."""
from pathlib import Path
import time
import dose as D

S,R,torch,B=D.S,D.R,D.torch,D.B


def fit(path,x,ids,identity,ds,c,initial,ix):
    path=Path(path)
    identity=dict(**identity,training_helper_hash=R.digest(Path(__file__).read_bytes()),dependencies=D.dependencies())
    if path.exists():
        s=R.trials.load(path);assert s['identity']==identity;return s
    m=B.Encoder();m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
    data=ds[131]['data'];frames=[dict(step=0,**R.geometry(m,c,1025))];start=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        code=m(x[ids[step]])
        ya=m(torch.cat([data['x'][ix[step]],data['xp'][ix[step]]]))
        alignment=(ya[:512]-ya[512:]).square().mean()
        loss=.9*alignment+.1*reg(code)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,**R.geometry(m,c,1025)))
    d=1/x[:,1];lo,hi=float(d.min()),float(d.max())
    saved=dict(identity=identity,initial=initial,model=m.state_dict(),optimizer=opt.state_dict(),config=vars(c),
        frames=frames,support=[lo,hi],seconds=time.monotonic()-start)
    R.trials.save_torch(path,saved);check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001)
    saved['geometry']['roots_in_source_support']=[r for r in saved['geometry']['roots'] if lo<=r<=hi]
    R.trials.save_torch(path,saved)
    return saved

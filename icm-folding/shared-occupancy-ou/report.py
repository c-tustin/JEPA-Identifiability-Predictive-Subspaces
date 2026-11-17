"""Reload all fits, verify paired inputs and geometry, render measured occupancy."""
from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = { .01:'#2166ac', .1:'#c44e52' }

def verify(path):
    r = old.trials.load(path)
    assert r['code_sha256']==sha(HERE/'run.py')
    assert r['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    assert r['reference_sha256']==sha(base.__file__)
    assert r['sensor_sha256']==sha(old.trials.behavior.__file__)
    c = old.trials.behavior.Config(**r['config'])
    tr = generate(r['seed'],r['target'],r['dt'],c.train_episodes,c.horizon)
    for key in ('z','q','noise','initial'):
        np.testing.assert_array_equal(tr[key],r['trajectory'][key])
    x,xp = observations(tr,c)
    assert torch.equal(x,r['x']) and torch.equal(xp,r['xp'])
    expected = torch.randint(len(x),(c.jepa_steps,c.batch),generator=torch.Generator().manual_seed(2400000+r['seed']))
    assert torch.equal(expected,r['indices'])
    torch.manual_seed(40000+r['seed'])
    initial = old.ScaledEncoder(4)
    assert all(torch.equal(v,r['initial'][k]) for k,v in initial.state_dict().items())
    model = old.ScaledEncoder(4)
    model.load_state_dict(r['encoder'])
    _,y,s = base.curve(model,c)
    np.testing.assert_array_equal(y,r['history'][-1]['y'])
    np.testing.assert_array_equal(s,r['history'][-1]['slope'])
    model = model.double()
    q = torch.linspace(0,np.log(16),20001,dtype=torch.float64,requires_grad=True)
    y = model(old.trials.behavior.sensor(q.exp(),c))
    slope = torch.autograd.grad(y.sum(),q)[0].detach().numpy()
    grid = q.detach().numpy()
    y = y.detach().numpy()
    roots = base.roots(grid,slope)
    mass = masses(r['target'])
    a,b = tr['q'][:-1].reshape(-1),tr['q'][1:].reshape(-1)
    ha,_ = np.histogram(a,EDGES)
    hb,_ = np.histogram(b,EDGES)
    hist = (ha+hb)/(len(a)+len(b))
    peaks = base.PEAKS[r['target']]
    cuts = (np.log(peaks[:-1])+np.log(peaks[1:]))/2 if peaks else []
    ia,ib = np.digitize(a,cuts),np.digitize(b,cuts)
    k = len(cuts)+1
    matrix = np.bincount(k*ia+ib,minlength=k*k).reshape(k,k)
    delta = abs(np.exp(b)-np.exp(a))
    metrics = dict(seed=r['seed'],target=r['target'],dt=r['dt'],roots=roots,folded=bool(roots),
        r2=float(np.corrcoef(grid,y)[0,1]**2), encoder_std=float(y.std()),
        target_mass=mass.tolist(),endpoint_histogram=hist.tolist(),
        tv=float(abs(hist-mass).sum()/2),x_tv=float(abs(ha/len(a)-mass).sum()/2),
        xp_tv=float(abs(hb/len(b)-mass).sum()/2),
        latent_lag_correlation=float(np.corrcoef(tr['z'][:-1].reshape(-1),tr['z'][1:].reshape(-1))[0,1]),
        mode_switch_fraction=float(np.mean(ia!=ib)), transition_counts=matrix.tolist(),
        absolute_distance_step_quantiles=dict(zip(('p50','p90','p99','max'),np.quantile(delta,[.5,.9,.99,1]).tolist())),
        fraction_exceeding_original_action_bound=float(np.mean(delta>c.move)), checkpoint_sha256=sha(path))
    return r,metrics,grid,y

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def main():
    torch.set_num_threads(1)
    records = {}
    for name in TARGETS:
        for dt in DTS:
            for seed in SEEDS:
                path = OUT/f'{name}-dt{dt:g}-{seed}.pt'
                records[name,dt,seed] = verify(path)
    # Explicit cross-condition pairing checks, independent of the trainer.
    for seed in SEEDS:
        runs = [r for (name,dt,s), (r,*_) in records.items() if s==seed]
        for r in runs[1:]:
            assert torch.equal(r['indices'],runs[0]['indices'])
            assert all(torch.equal(v,runs[0]['initial'][k]) for k,v in r['initial'].items())
            np.testing.assert_array_equal(r['trajectory']['noise'],runs[0]['trajectory']['noise'])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(3,3,figsize=(15,11))
    for row,name in enumerate(TARGETS):
        axes[row,0].stairs(masses(name),np.exp(EDGES),color='black',ls='--',label='Target')
        for dt in DTS:
            hist = np.mean([records[name,dt,s][1]['endpoint_histogram'] for s in SEEDS],axis=0)
            axes[row,0].stairs(hist,np.exp(EDGES),color=COLORS[dt],label=f'dt={dt}')
            r = records[name,dt,131][0]
            axes[row,1].plot(np.exp(r['trajectory']['q'][:,0]),color=COLORS[dt],label=f'dt={dt}')
            for seed,style in zip(SEEDS,('-','--',':')):
                _,m,q,y = records[name,dt,seed]
                sign = 1 if np.cov(q,y)[0,1]>=0 else -1
                axes[row,2].plot(np.exp(q),sign*(y-y.mean())/y.std(),style,color=COLORS[dt],label=f'dt={dt}, seed {seed}')
        axes[row,0].set(title=f'{name}: actual shared endpoint occupancy',xlabel='Distance',ylabel='Probability per log bin',xscale='log')
        axes[row,0].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        axes[row,1].set(title='One trajectory, seed 131',xlabel='Observation step',ylabel='Distance',ylim=(1,16))
        axes[row,2].set(title='Ordinary alignment + SIGReg',xlabel='Distance',ylabel='Standardized encoder')
    for ax in axes[0]:
        ax.legend(fontsize=7)
    fig.suptitle('Both losses use the same temporal pairs and the same encoded endpoint tensor',y=1.01)
    save(fig,'shared_occupancy_encoders')
    fig,axes = plt.subplots(2,3,figsize=(12,7))
    for row,dt in enumerate(DTS):
        for col,name in enumerate(TARGETS):
            counts = np.sum([records[name,dt,s][1]['transition_counts'] for s in SEEDS],axis=0)
            prob = counts/counts.sum(1,keepdims=True)
            ax = axes[row,col]
            ax.imshow(prob,vmin=0,vmax=1,cmap='Blues')
            for (i,j),p in np.ndenumerate(prob):
                ax.text(j,i,f'{p:.3f}',ha='center',va='center',color='white' if p>.55 else 'black')
            ax.set(title=f'{name}, dt={dt}',xlabel='Next basin',ylabel='Current basin',xticks=range(len(prob)),yticks=range(len(prob)))
    save(fig,'transition_matrices')
    all_metrics = [v[1] for v in records.values()]
    old.trials.atomic_json(OUT/'summary.json',dict(runs=all_metrics))
    old.trials.atomic_json(OUT/'verification.json',dict(checkpoints=len(records), exact_data_regeneration=True,
        adjacent_pair_sensor_reconstruction=True, exact_checkpoint_curve_reproduction=True,
        float64_slope_grid_points=20001, paired_initializations=True, paired_indices=True, paired_innovations=True,
        hashes_verified=True))
    root = HERE.parents[1]
    sources = {HERE/'run.py', HERE/'report.py', HERE/'PROTOCOL.md', HERE/'requirements.txt'}
    for module in tuple(sys.modules.values()):
        file = getattr(module,'__file__',None)
        if file:
            path = Path(file).resolve()
            if path.suffix=='.py' and path.is_relative_to(root):
                sources.add(path)
    import scipy
    old.trials.atomic_json(OUT/'provenance.json',dict(
        sources={str(p.relative_to(root)):sha(p) for p in sorted(sources)},
        versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__)))
    lines = ['# Shared-input multimodal OU results','','November 16, 2026. All 18 fits completed: three targets × two OU time steps × three diagnostic seeds, 4,000 updates each. Both losses use identical paired observations and a single encoder evaluation. No occupancy weights or ICM transition pairs.','',
        '| Target | dt | Final folds | Mean endpoint TV to target | Mean basin-switch rate | Mean fraction of steps > 0.25 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in TARGETS:
        for dt in DTS:
            group = [m for m in all_metrics if m['target']==name and m['dt']==dt]
            lines.append(f"| {name} | {dt} | {sum(m['folded'] for m in group)}/3 | {np.mean([m['tv'] for m in group]):.4f} | {np.mean([m['mode_switch_fraction'] for m in group]):.4f} | {np.mean([m['fraction_exceeding_original_action_bound'] for m in group]):.4f} |")
    lines += ['', f"Final encoder standard deviations on the uniform-log evaluation grid range from {min(m['encoder_std'] for m in all_metrics):.4f} to {max(m['encoder_std'] for m in all_metrics):.4f}; affine log-distance R² ranges from {min(m['r2'] for m in all_metrics):.4f} to {max(m['r2'] for m in all_metrics):.4f}. Standardized plots alone do not show these output scales.",
        '', 'TV is total variation of the measured 64-bin endpoint histogram from the prescribed mass; endpoints are temporally correlated. Basin boundaries are log-midpoints between peaks. Switching counts all adjacent basin changes and can include repeated boundary crossings. The uniform control has one basin by convention.',
        '', 'Checkpoint reload reproduces stored final float32 curves/slopes exactly. Fold roots are independently recomputed on a 20,001-point float64 grid; these are empirical sign-reversal checks, not injectivity guarantees. All generated data, shared pair-index streams, initializations and normal innovations were regenerated and checked.',
        '', 'This passive transformed OU process permits distance increments beyond the original three-action environment. Occupancy is stationary in the population by construction, but finite correlated trajectories fluctuate. The plots display actual data occupancy. Time-step comparisons change temporal coupling; cross-target comparisons change both occupancy and pair geometry. No causal occupancy-only or bounded-action actor claim follows.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    main()

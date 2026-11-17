"""Verify new fits and render occupancy, encoder and derivative comparisons."""
from run import *
import sys
import scipy
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'native':'#222222','ou-small':'#2166ac','ou-large':'#c44e52'}
LABELS = {'native':'Original ICM pairs','ou-small':'ICM-matched OU, dt=.01','ou-large':'ICM-matched OU, dt=.1'}
EDGES = np.geomspace(1,16,65)

def geometry(model,c):
    model = model.double()
    q = torch.linspace(0,np.log(16),20001,dtype=torch.float64,requires_grad=True)
    y = model(old.trials.behavior.sensor(q.exp(),c))
    slope = torch.autograd.grad(y.sum(),q)[0]
    q,y,slope = (t.detach().numpy() for t in (q,y,slope))
    return q,y,slope,base.roots(q,slope)

def histogram(d,bins=64):
    counts,_ = np.histogram(d,np.geomspace(1,16,bins+1))
    return counts/counts.sum()

def verify(seed,arm):
    path = OUT/f'{arm}-{seed}.pt'
    r = old.trials.load(path)
    source_path,data,c = source(seed)
    assert r['source_sha256']==sha(source_path)
    assert r['code_sha256']==sha(HERE/'run.py')
    assert r['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    assert r['shared_code_sha256']==sha(shared.__file__)
    pool,target = sorted_pool(data)
    if arm=='native':
        x,xp = data['x'],data['xp']
        assert r['trajectory'] is None
    else:
        tr,x,xp = generate(seed,ARMS[arm],pool,target,c.train_episodes,c.horizon)
        for key,value in tr.items():
            np.testing.assert_array_equal(value,r['trajectory'][key])
        # Reconstruct endpoint selection from saved ranks, independently of generator output.
        rr = r['trajectory']['ranks']
        torch.testing.assert_close(x,pool[torch.from_numpy(rr[:-1].reshape(-1))],rtol=0,atol=0)
        torch.testing.assert_close(xp,pool[torch.from_numpy(rr[1:].reshape(-1))],rtol=0,atol=0)
        assert torch.equal(xp[:-c.train_episodes],x[c.train_episodes:])
    expected = torch.randint(len(x),(c.jepa_steps,c.batch),generator=torch.Generator().manual_seed(2400000+seed))
    assert torch.equal(expected,r['indices'])
    torch.manual_seed(40000+seed)
    initial = old.ScaledEncoder(4)
    assert all(torch.equal(v,r['initial'][k]) for k,v in initial.state_dict().items())
    model = old.ScaledEncoder(4)
    model.load_state_dict(r['encoder'])
    _,y,slope = base.curve(model,c)
    np.testing.assert_array_equal(y,r['history'][-1]['y'])
    np.testing.assert_array_equal(slope,r['history'][-1]['slope'])
    q,y,slope,roots = geometry(model,c)
    d = old.trials.distances(x).numpy().astype(np.float64)
    dp = old.trials.distances(xp).numpy().astype(np.float64)
    endpoints = np.r_[d,dp]
    delta = dp-d
    counts = histogram(endpoints)
    target_counts = histogram(target)
    assert np.all(counts[target_counts==0]==0)
    def inside(rs,lo,hi):
        return [v for v in rs if lo<=v<=hi]
    metrics = dict(seed=seed,arm=arm,dt=ARMS[arm],roots=roots,folded=bool(roots),
        roots_inside_source_support=inside(roots,target.min(),target.max()),
        roots_inside_training_support=inside(roots,endpoints.min(),endpoints.max()),
        any_sampled_history_fold=any(h['roots'] for h in r['history']),
        source_support=[float(target.min()),float(target.max())],training_support=[float(endpoints.min()),float(endpoints.max())],
        encoder_std=float(y.std()),r2=float(np.corrcoef(q,y)[0,1]**2),
        target_histogram=target_counts.tolist(),endpoint_histogram=counts.tolist(),
        tv64=float(abs(counts-target_counts).sum()/2),
        tv32=float(abs(histogram(endpoints,32)-histogram(target,32)).sum()/2),
        x_tv64=float(abs(histogram(d)-target_counts).sum()/2),xp_tv64=float(abs(histogram(dp)-target_counts).sum()/2),
        ks_distance=float(ks_2samp(endpoints,target).statistic),
        fraction_below2=float(np.mean(endpoints<2)),target_fraction_below2=float(np.mean(target<2)),
        absolute_step_quantiles=dict(zip(('p50','p90','p99','max'),np.quantile(abs(delta),[.5,.9,.99,1]).tolist())),
        fraction_step_over025=float(np.mean(abs(delta)>.25001)),mean_signed_step=float(delta.mean()),
        checkpoint_sha256=sha(path),source_sha256=sha(source_path))
    return r,metrics,q,y,slope

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def gallery(records):
    fig,axes = plt.subplots(3,4,figsize=(17,10))
    for col,seed in enumerate(SEEDS):
        for arm in ARMS:
            r,m,q,y,slope = records[seed,arm]
            color = COLORS[arm]
            axes[0,col].stairs(m['endpoint_histogram'],EDGES,color=color,label=LABELS[arm],lw=1.6)
            sign = 1 if np.cov(q,y)[0,1]>=0 else -1
            axes[1,col].plot(np.exp(q),sign*(y-y.mean())/y.std(),color=color,lw=1.6)
            axes[2,col].plot(np.exp(q),sign*slope/y.std(),color=color,lw=1.6)
            for root in m['roots']:
                axes[1,col].axvline(root,color=color,ls=':',alpha=.6)
                axes[2,col].plot(root,0,'o',color=color,ms=4)
        m = records[seed,'native'][1]
        for row in (1,2):
            axes[row,col].axvspan(1,m['source_support'][0],color='gray',alpha=.12)
            axes[row,col].axvspan(m['source_support'][1],16,color='gray',alpha=.12)
            axes[row,col].set(xlim=(1,16),xlabel='Distance')
        counts = '/'.join('F' if records[seed,a][1]['folded'] else 'ordered' for a in ARMS)
        axes[0,col].set(title=f'Seed {seed}',xlabel='Distance',ylabel='Probability per log bin',xscale='log')
        axes[0,col].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        axes[1,col].set(title=counts,ylabel='Standardized encoder')
        axes[2,col].axhline(0,color='gray',lw=1,ls='--')
        axes[2,col].set(ylabel='Slope versus log-distance')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('ICM occupancy matched by OU: both losses share all encoded endpoints\nCurve titles: original ICM / OU dt=.01 / OU dt=.1; F = detected fold. Gray bands: outside source support.',y=1.03)
    save(fig,'icm_matched_histograms_encoders')
    fig,axes = plt.subplots(2,4,figsize=(17,7))
    for col,seed in enumerate(SEEDS):
        for arm in ARMS:
            r,m,*_ = records[seed,arm]
            history = r['history']
            steps = [h['step'] for h in history]
            axes[0,col].plot(steps,[len(h['roots']) for h in history],color=COLORS[arm],label=LABELS[arm])
            for h in history:
                if h['roots']:
                    axes[1,col].scatter([h['step']]*len(h['roots']),h['roots'],s=8,color=COLORS[arm],alpha=.7)
        axes[0,col].set(title=f'Seed {seed}',xlabel='Update',ylabel='Detected roots')
        axes[1,col].set(xlabel='Update',ylabel='Turn distance',ylim=(1,16))
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Sampled training history (every 100 updates; missed transient/inter-grid roots remain possible)',y=1.01)
    save(fig,'fold_histories')

def multimodal_gallery():
    fig,axes = plt.subplots(2,2,figsize=(11,7))
    for col,name in enumerate(('bimodal','trimodal')):
        axes[0,col].stairs(shared.masses(name),np.exp(shared.EDGES),color='black',ls='--',label='Target')
        for dt,color in ((.01,'#2166ac'),(.1,'#c44e52')):
            measured = []
            for seed,style in zip(shared.SEEDS,('-','--',':')):
                path = shared.OUT/f'{name}-dt{dt:g}-{seed}.pt'
                r = old.trials.load(path)
                assert r['code_sha256']==sha(shared.__file__)
                q = r['trajectory']['q']
                counts,_ = np.histogram(np.r_[q[:-1].reshape(-1),q[1:].reshape(-1)],shared.EDGES)
                measured.append(counts/counts.sum())
                c = old.trials.behavior.Config(**r['config'])
                model = old.ScaledEncoder(4)
                model.load_state_dict(r['encoder'])
                grid,y,s,roots = geometry(model,c)
                assert not roots
                sign = 1 if np.cov(grid,y)[0,1]>=0 else -1
                axes[1,col].plot(np.exp(grid),sign*(y-y.mean())/y.std(),style,color=color,label=f'dt={dt}, seed {seed}')
            axes[0,col].stairs(np.mean(measured,axis=0),np.exp(shared.EDGES),color=color,label=f'Measured dt={dt}')
        axes[0,col].set(title=name.capitalize()+' shared endpoint occupancy',xlabel='Distance',ylabel='Probability per log bin',xscale='log')
        axes[0,col].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        axes[1,col].set(title='Final encoder: no detected folds in these runs',xlabel='Distance',ylabel='Standardized encoder')
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Bimodal/trimodal OU: alignment and SIGReg receive identical pairs\nThree seeds × two time steps per target; 4,000 updates',y=1.03)
    save(fig,'bimodal_trimodal_shared_inputs')

def transformation_figure():
    _,data,_ = source(131)
    _,d = sorted_pool(data)
    z = np.linspace(-3,3,1001)
    u = ndtr(z)
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    axes[0].plot(z,u,color='#222222')
    axes[0].set(title='1. Gaussian state → percentile',xlabel='Latent OU state z',ylabel='u = Φ(z)')
    for name,color in (('bimodal','#2166ac'),('trimodal','#147b79')):
        mapped = np.exp(shared.quantile(u,shared.masses(name)))
        axes[1].plot(u,mapped,label=name,color=color)
        axes[2].plot(z,mapped,label=name,color=color)
    ranks = np.minimum((len(d)*u).astype(np.int64),len(d)-1)
    axes[1].plot(u,d[ranks],label='ICM seed 131',color='#c44e52')
    axes[2].plot(z,d[ranks],label='ICM seed 131',color='#c44e52')
    axes[1].set(title='2. Percentile → target distance',xlabel='Percentile u',ylabel='d = exp(F⁻¹(u))')
    axes[2].set(title='Composed increasing transform',xlabel='Latent OU state z',ylabel='Distance d')
    for ax in axes[1:]:
        ax.legend(fontsize=8)
    fig.suptitle('The generator transforms observations, before either training loss is evaluated',y=1.03)
    save(fig,'ou_quantile_transform')

def main():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    records = {(seed,arm):verify(seed,arm) for seed in SEEDS for arm in ARMS}
    for seed in SEEDS:
        first = records[seed,'native'][0]
        for arm in ('ou-small','ou-large'):
            r = records[seed,arm][0]
            assert torch.equal(first['indices'],r['indices'])
            assert all(torch.equal(v,r['initial'][k]) for k,v in first['initial'].items())
        np.testing.assert_array_equal(records[seed,'ou-small'][0]['trajectory']['noise'],records[seed,'ou-large'][0]['trajectory']['noise'])
    gallery(records)
    multimodal_gallery()
    transformation_figure()
    metrics = [v[1] for v in records.values()]
    old.trials.atomic_json(OUT/'summary.json',dict(runs=metrics))
    old.trials.atomic_json(OUT/'verification.json',dict(checkpoints=len(records),
        generator_exact_regeneration=True,observations_exactly_selected_from_native_pool=True,empty_bins_preserved=True,
        checkpoint_float32_curves_and_slopes_exact=True,float64_slope_points=20001,
        paired_initializations=True,paired_indices=True,paired_ou_innovations=True,source_and_training_hashes_verified=True))
    root = HERE.parents[1]
    files = {HERE/'run.py',HERE/'report.py',HERE/'PROTOCOL.md',Path(shared.__file__).resolve(),Path(base.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        f = getattr(module,'__file__',None)
        if f:
            p = Path(f).resolve()
            if p.suffix=='.py' and p.is_relative_to(root):
                files.add(p)
    old.trials.atomic_json(OUT/'provenance.json',dict(sources={str(p.relative_to(root)):sha(p) for p in sorted(files)},
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__)))
    lines = ['# ICM-occupancy-matched OU results','','November 16, 2026. Twelve completed 4,000-update fits. Both losses use the same minibatch and encoder outputs in all arms.','',
        '| Seed | Data | Final folds (root distances) | Roots inside source support | 64-bin TV | KS distance | Mass below 2 |',
        '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for m in metrics:
        roots = ', '.join(f'{r:.3f}' for r in m['roots']) or 'none'
        lines.append(f"| {m['seed']} | {m['arm']} | {roots} | {len(m['roots_inside_source_support'])} | {m['tv64']:.4f} | {m['ks_distance']:.4f} | {m['fraction_below2']:.4f} |")
    lines += ['','| Arm | Final folded / 4 | Any sampled training fold / 4 | Mean 64-bin TV | Mean per-seed median absolute step |',
        '| --- | ---: | ---: | ---: | --- |']
    for arm in ARMS:
        group = [m for m in metrics if m['arm']==arm]
        lines.append(f"| {arm} | {sum(m['folded'] for m in group)}/4 | {sum(m['any_sampled_history_fold'] for m in group)}/4 | {np.mean([m['tv64'] for m in group]):.4f} | {np.mean([m['absolute_step_quantiles']['p50'] for m in group]):.4f} |")
    lines += ['', 'All saved final float32 curves/slopes reproduce exactly after checkpoint reload. Final roots were independently recomputed on 20,001 float64 log-distance points. Roots inside source support are distinguished from extrapolation; the gray bands in the gallery mark unobserved distances. History checks use a coarser 1,025-point grid every 100 updates.',
        '', 'The OU rank map targets the entire empirical endpoint distribution without smoothing, including exact empty bins and source multiplicities. Actual finite correlated rollouts only approximate that distribution; TV and KS above measure this approximation, not independent-sample significance. No fresh seeds, learned OU actor, bounded-action equivalence or general injectivity guarantee.',
        '', 'Matching occupancy does not match temporal pairs: OU targets the pooled stationary marginal, whereas ICM retains its recorded transitions. Both new losses share inputs in the native control too, so its fold outcomes need not equal the historical separate-regularizer-input control.',
        '', 'See [histograms/encoders/slopes](icm_matched_histograms_encoders.png), [fold histories](fold_histories.png), [multimodal gallery](bimodal_trimodal_shared_inputs.png), [mapping diagram](ou_quantile_transform.png), [full metrics](summary.json), and [verification](verification.json).','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    main()

"""Reload the invariant-alignment experiment and summarize causal contrasts."""
from run import *
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'native':'#c44e52','balanced':'#2166ac'}
LABELS={'native':'Native-like stays','balanced':'Spread-out stays'}
STYLES={.5:'-',.25:'--'}

def geometry(model,c):
    model=model.double()
    q=torch.linspace(0,np.log(16),20001,dtype=torch.float64,requires_grad=True)
    y=model(old.trials.behavior.sensor(q.exp(),c))
    slope=torch.autograd.grad(y.sum(),q)[0]
    q,y,slope=(t.detach().numpy() for t in (q,y,slope))
    dy=np.diff(y)
    total=abs(dy).sum()
    backward=min(dy.clip(0).sum(),(-dy).clip(0).sum())/total if total else 0.
    roots=base.roots(q,slope)
    return q,y,slope,roots,float(backward)

def verify(seed,fraction,allocation,s):
    path=OUT/f'{seed}-f{fraction:g}-{allocation}.pt'
    r=old.trials.load(path)
    assert r['code_sha256']==sha(HERE/'run.py')
    assert r['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    assert r['shared_code_sha256']==sha(shared.__file__)
    assert r['source_sha256']==sha(s['path'])
    assert r['source_plan_sha256']==s['source_plan_sha256']
    assert r['hold_plan_sha256']==s['hold_plan_sha256'][allocation]
    assert r['exposure']==s['exposure'][allocation]
    torch.manual_seed(40000+seed)
    model=old.ScaledEncoder(4)
    assert all(torch.equal(v,r['initial'][k]) for k,v in model.state_dict().items())
    model.load_state_dict(r['encoder'])
    _,y,slope=base.curve(model,s['c'])
    np.testing.assert_array_equal(y,r['history'][-1]['y'])
    np.testing.assert_array_equal(slope,r['history'][-1]['slope'])
    q,y,slope,roots,backward=geometry(model,s['c'])
    lo,hi=r['source_support']
    inside=[root for root in roots if lo<=root<=hi]
    realized=np.array(r['exposure']['realized'])
    expected=np.array(r['exposure']['expected'])
    metrics=dict(seed=seed,fraction=fraction,allocation=allocation,roots=roots,folded=bool(roots),
        roots_inside_source_range=inside,folded_inside_source_range=bool(inside),
        backward_variation=backward,std=float(y.std()),r2=float(np.corrcoef(q,y)[0,1]**2),
        realized_mass_below2=float(realized[:8].sum()),expected_mass_below2=float(expected[:8].sum()),
        occupancy_tv=float(abs(realized-expected).sum()/2),source_support=r['source_support'],
        expected_histogram=expected.tolist(),realized_histogram=realized.tolist(),checkpoint_sha256=sha(path))
    return r,metrics,q,y,slope

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def same_histogram_example():
    u=(np.arange(100000)+.5)/100000
    ordered=-np.cos(np.pi*u)
    folded=np.cos(2*np.pi*u)
    bins=np.linspace(-1,1,41)
    a,_=np.histogram(ordered,bins)
    b,_=np.histogram(folded,bins)
    difference=float(abs(a-b).sum()/(2*len(u)))
    assert difference<.001
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(u,ordered,color='#222222',label='Ordered: −cos(πu)')
    axes[0].plot(u,folded,color='#9467bd',label='Folded: cos(2πu)')
    axes[0].set(xlabel='Input percentile u',ylabel='Encoder output',title='Same input occupancy, different ordering')
    axes[0].legend(fontsize=8)
    axes[1].stairs(a/len(u),bins,color='#222222',label='Ordered output histogram',lw=2)
    axes[1].stairs(b/len(u),bins,color='#9467bd',ls='--',label='Folded output histogram',lw=2)
    axes[1].set(xlabel='Output',ylabel='Probability per bin',title='Identical population output distributions')
    axes[1].legend(fontsize=8)
    fig.suptitle('Analytic example: a marginal histogram cannot certify encoder ordering',y=1.03)
    save(fig,'same_histogram_different_order')
    old.trials.atomic_json(OUT/'analytic-example.json',dict(points=len(u),
        numerical_histogram_tv=difference,common_cdf='1-arccos(y)/pi on [-1,1]',
        ordered='-cos(pi*u)',folded='cos(2*pi*u)',trained_mlp=False))

def figures(records):
    fig,axes=plt.subplots(3,2,figsize=(11,11))
    for col,fraction in enumerate(FRACTIONS):
        for allocation in ALLOCATIONS:
            group=[records[s,fraction,allocation][1] for s in SEEDS]
            hist=np.mean([m['realized_histogram'] for m in group],axis=0)
            axes[0,col].stairs(hist,EDGES,color=COLORS[allocation],label=LABELS[allocation],lw=2)
        axes[0,col].set(title=f'{int(100*fraction)}% source pairs, {int(100*(1-fraction))}% added stays',
            xlabel='Distance',ylabel='Measured probability per log bin',xscale='log')
        axes[0,col].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        axes[0,col].legend(fontsize=8)
        counts=[sum(records[s,fraction,a][1]['folded_inside_source_range'] for s in SEEDS) for a in ALLOCATIONS]
        bars=axes[1,col].bar([LABELS[a] for a in ALLOCATIONS],counts,color=[COLORS[a] for a in ALLOCATIONS],width=.6)
        for bar,count in zip(bars,counts):
            axes[1,col].text(bar.get_x()+bar.get_width()/2,count+.15,f'{count}/10',ha='center')
        axes[1,col].set(ylabel='Final folded inside source range',ylim=(0,11),yticks=[0,2,4,6,8,10])
        for seed in SEEDS:
            yy=[records[seed,fraction,a][1]['backward_variation'] for a in ALLOCATIONS]
            axes[2,col].plot([0,1],yy,color='gray',alpha=.6,lw=1)
            for i,a in enumerate(ALLOCATIONS):
                axes[2,col].scatter(i,yy[i],color=COLORS[a],s=25)
        axes[2,col].set(xticks=[0,1],xticklabels=[LABELS[a] for a in ALLOCATIONS],ylabel='Normalized backward variation',ylim=(-.015,.52))
    fig.suptitle('Occupancy intervention: alignment function is identical between colors within each column\nSame source pairs, stay count, initial weights, optimizer and encoded inputs to both losses',y=1.02)
    save(fig,'causal_summary')
    gallery=(130,131,134,136)
    fig,axes=plt.subplots(3,4,figsize=(17,10))
    for col,seed in enumerate(gallery):
        for fraction in FRACTIONS:
            for allocation in ALLOCATIONS:
                r,m,q,y,slope=records[seed,fraction,allocation]
                color,style=COLORS[allocation],STYLES[fraction]
                label=f'{LABELS[allocation]}, source {fraction:g}'
                axes[0,col].stairs(m['realized_histogram'],EDGES,color=color,ls=style,label=label)
                sign=1 if np.cov(q,y)[0,1]>=0 else -1
                axes[1,col].plot(np.exp(q),sign*(y-y.mean())/y.std(),style,color=color,lw=1.5)
                axes[2,col].plot(np.exp(q),sign*slope/y.std(),style,color=color,lw=1.5)
                for root in m['roots']:
                    axes[2,col].plot(root,0,'o',color=color,ms=3)
        axes[0,col].set(title=f'Seed {seed}',xlabel='Distance',ylabel='Measured log-bin probability',xscale='log')
        axes[0,col].set_xticks([1,2,4,8,16],labels=['1','2','4','8','16'])
        axes[1,col].set(xlabel='Distance',ylabel='Standardized encoder',xlim=(1,16))
        axes[2,col].axhline(0,color='gray',ls=':',lw=1)
        axes[2,col].set(xlabel='Distance',ylabel='Slope versus log-distance',xlim=(1,16))
        for row in (1,2):
            lo,hi=records[seed,.5,'native'][1]['source_support']
            axes[row,col].axvspan(1,lo,color='gray',alpha=.12)
            axes[row,col].axvspan(hi,16,color='gray',alpha=.12)
    axes[0,0].legend(fontsize=6)
    fig.suptitle('Native-like vs spread-out stay occupancy: compare colors at the same line style\nSolid: 50% source pairs; dashed: 25%. Gray: outside stored distance range.',y=1.03)
    save(fig,'occupancy_encoders_slopes')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,fraction in enumerate(FRACTIONS):
        for seed in SEEDS:
            r=records[seed,fraction,'native'][0]
            for fork in r['forks']:
                before=np.array(fork['slopes_before'])
                diff=np.array(fork['slopes_after']['balanced'])-np.array(fork['slopes_after']['native'])
                axes[row,0].scatter(fork['alignment_gradient_max_error'],fork['regularizer_gradient_difference_norm'],color='#147b79',s=12,alpha=.6)
                axes[row,1].plot(PROBES,diff,'o-',alpha=.25,lw=.8,ms=3)
        axes[row,0].set(title=f'Source fraction {fraction:g}',xlabel='Max alignment-gradient difference',ylabel='SIGReg gradient difference norm')
        axes[row,1].axhline(0,color='gray',ls=':',lw=1)
        axes[row,1].set(title='Spread-out minus native-like: next-step slopes',xlabel='Query distance',ylabel='Difference after actual Adam step')
    fig.suptitle('Same-state update forks: same weights, Adam moments and source pairs\nDifferences enter through SIGReg; slope response need not have one sign across states.',y=1.03)
    save(fig,'same_state_forks')

def main():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    records={}
    for seed in SEEDS:
        for fraction in FRACTIONS:
            s=setup(seed,fraction)
            for allocation in ALLOCATIONS:
                records[seed,fraction,allocation]=verify(seed,fraction,allocation,s)
            a,b=(records[seed,fraction,k][0] for k in ALLOCATIONS)
            assert a['source_plan_sha256']==b['source_plan_sha256']
            assert all(torch.equal(v,b['initial'][k]) for k,v in a['initial'].items())
    figures(records)
    same_histogram_example()
    metrics=[r[1] for r in records.values()]
    forks=[dict(seed=seed,fraction=fraction,**fork) for (seed,fraction,allocation),(r,*_) in records.items() if allocation=='native' for fork in r['forks']]
    old.trials.atomic_json(OUT/'summary.json',dict(runs=metrics))
    old.trials.atomic_json(OUT/'forks.json',dict(probe_distances=PROBES,checks=forks))
    old.trials.atomic_json(OUT/'verification.json',dict(checkpoints=len(records),
        source_and_code_hashes_verified=True,plans_and_exposure_regenerated=True,paired_initializations=True,
        paired_source_plans=True,float32_curve_and_slope_reload_exact=True,float64_slope_points=20001,
        same_state_forks=len(forks),max_alignment_gradient_difference=max(f['alignment_gradient_max_error'] for f in forks),
        max_gradient_channel_error=max(f['gradient_channel_max_error'] for f in forks)))
    root=HERE.parents[1]
    files={HERE/'run.py',HERE/'report.py',HERE/'PROTOCOL.md',Path(reference.__file__).resolve(),Path(shared.__file__).resolve(),Path(base.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.suffix=='.py' and p.is_relative_to(root):files.add(p)
    old.trials.atomic_json(OUT/'provenance.json',dict(sources={str(p.relative_to(root)):sha(p) for p in sorted(files)},
        versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__)))
    lines=['# Occupancy intervention with invariant alignment','','November 17, 2026. Forty completed 4,000-update fits on ten historical sources. Both losses use the same encoded endpoint tensor. Added stay activations are reused in both endpoint positions; their alignment value and gradient are exactly zero.','',
        '| Source-pair fraction | Stay allocation | Final folds in [1,16] | Final folds in source range | Mean mass below 2 | Mean occupancy TV to target |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for fraction in FRACTIONS:
        for allocation in ALLOCATIONS:
            group=[m for m in metrics if m['fraction']==fraction and m['allocation']==allocation]
            lines.append(f"| {fraction} | {allocation} | {sum(m['folded'] for m in group)}/10 | {sum(m['folded_inside_source_range'] for m in group)}/10 | {np.mean([m['realized_mass_below2'] for m in group]):.4f} | {np.mean([m['occupancy_tv'] for m in group]):.5f} |")
    lines+=['','Only comparisons between allocations at the same source fraction have an identical alignment function. Changing source fraction changes alignment strength and is a separate stratum. Original ICM source pairs can already contain stays.',
        '',f"All 40 final float32 curves/slopes reproduce exactly after checkpoint reload; final geometry was checked independently on 20,001 float64 log-distance points. Output standard deviations on that grid range from {min(m['std'] for m in metrics):.5f} to {max(m['std'] for m in metrics):.5f}.",
        '',f"Sixty same-state forks at updates 100/200/400 have maximum alignment-gradient difference {max(f['alignment_gradient_max_error'] for f in forks):.8g}; maximum residual after attributing the combined-gradient contrast to 0.1 times the SIGReg contrast is {max(f['gradient_channel_max_error'] for f in forks):.8g}. Forks share weights, optimizer state and the next source batch.",
        '', 'The sampled regularizer occupancy is the manipulated training variable, with an analytically invariant alignment function. Changes in final geometry identify its causal effect in this controlled setting; they need not be caused exclusively by the population objective because finite-batch moments and gradient noise also change. No universal occupancy threshold, fresh-policy validation, real-trajectory stationary-distribution claim or continuous injectivity guarantee.',
        '', 'Normalized backward variation is min(total upward variation,total downward variation)/total variation on the evaluation grid. It is sign/scale invariant, zero for a monotone sampled curve, and at most 0.5. A source-range root means within recorded min/max distances, not proof that every intermediate state was observed.',
        '', 'See [summary](causal_summary.png), [histograms/encoders/slopes](occupancy_encoders_slopes.png), [same-state forks](same_state_forks.png), [full metrics](summary.json), [verification](verification.json), [input/alignment audit](audit.json), and [protocol](../PROTOCOL.md).','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()

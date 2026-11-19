"""Plot audited replay histories and explicitly selected representation examples."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/icm-gradient-cache')
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import run as original

HERE=original.HERE
OUT=original.OUT
HISTORY=OUT/'convergence'
TEAL='#087f8c'
ORANGE='#d16b32'
PURPLE='#7b59a3'
COLORS={'sigreg':'#ba4450','vicreg':'#8264a8','infonce':'#cc8a2e','raw-own-code':TEAL}
LABELS={'sigreg':'SIGReg','vicreg':'VICReg (scalar)','infonce':'Temporal InfoNCE','raw-own-code':'Raw / own-code InfoNCE (latest)'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(ax,x,values,color,label,style='-'):
    low,mid,high=np.quantile(values,[.1,.5,.9],axis=0)
    ax.fill_between(x,low,high,color=color,alpha=.13,linewidth=0)
    ax.plot(x,mid,color=color,lw=2.3,label=label,ls=style)


def finish(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,facecolor='white')
    svg=OUT/(name+'.svg')
    fig.savefig(svg,facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def normalize_curve(g):
    y=np.asarray(g['y'])
    # Display only: remove the arbitrary affine code scale and global sign.
    sign=1 if np.corrcoef(g['q'],y)[0,1]>=0 else -1
    return sign*(y-y.mean())/max(y.std(),1e-12)


def main():
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'semibold','axes.labelcolor':'#333b45',
                         'text.color':'#25303b','axes.edgecolor':'#aeb7bf',
                         'grid.alpha':.2,'grid.color':'#aeb7bf','legend.frameon':False})
    arrays=[]; audits=[]; records={}; scorer_diagnostics=[]
    for seed in range(130,145):
        path=HISTORY/f'{seed}.npz'
        audit=json.loads((HISTORY/f'{seed}.json').read_text())
        assert audit['array_sha256']==digest(path)
        assert audit['replay_code_sha256']==digest(HERE/'convergence.py')
        assert all(audit[k] for k in ('encoder_bit_exact','critic_bit_exact','decoder_bit_exact',
                                      'original_saved_losses_exact','final_full_dataset_errors_exact'))
        arrays.append(dict(np.load(path)))
        audits.append(audit)
        records[seed]=original.base.trials.load(OUT/'runs'/f'{seed}-n4000.pt')
        # The renderer lies on a one-dimensional affine line in raw space.
        # Removing a critic component perpendicular to that line changes all
        # candidate distances by the same row constant, so softmax ignores it.
        c,train,test,_,encoder,scorer,_,mean,std=original.setup(seed)
        encoder.load_state_dict(records[seed]['model'])
        scorer.load_state_dict(records[seed]['critic'])
        gen=original.torch.Generator().manual_seed(7400000+10*seed+1)
        ix=original.torch.randperm(len(test['x']),generator=gen)[:128]
        raw=original.torch.cat([test['x'][ix],test['xp'][ix]])
        fixed=((raw-mean)/std).double()
        with original.torch.no_grad():
            predicted=scorer(encoder(raw)[:,None]).double()
            origin=fixed.mean(0)
            direction=original.torch.linalg.svd(fixed-origin,full_matrices=False).Vh[0]
            error=predicted-fixed
            parallel=(error@direction).square().mean()/fixed.shape[1]
            orthogonal=(error-(error@direction)[:,None]*direction).square().mean()
            projected=origin+((predicted-origin)@direction)[:,None]*direction
            def log_prob(p):
                return original.torch.log_softmax(-(p[:,None,:]-fixed[None,:,:]).square().mean(-1)/.2,dim=1)
            logprob_shift=(log_prob(predicted)-log_prob(projected)).abs().max().item()
            assert logprob_shift<.003, (seed,logprob_shift)
        scorer_diagnostics.append(dict(seed=seed,raw_mse=error.square().mean().item(),
            parallel_mse=parallel.item(),orthogonal_mse=orthogonal.item(),
            orthogonal_fraction=(orthogonal/(parallel+orthogonal)).item(),
            max_abs_log_probability_shift_after_projection=logprob_shift))
    stack=lambda key:np.stack([a[key] for a in arrays])
    updates=arrays[0]['encoder_updates']
    decoder_updates=arrays[0]['decoder_updates']
    for a in arrays:
        np.testing.assert_array_equal(updates,a['encoder_updates'])
        np.testing.assert_array_equal(decoder_updates,a['decoder_updates'])
        assert all(np.isfinite(v).all() for v in a.values())

    fig,axes=plt.subplots(1,3,figsize=(15.2,4.6),layout='constrained')
    # Non-overlapping 50-update means; no interpolation masquerading as measurements.
    losses=stack('update_loss').reshape(15,80,50).mean(2)
    band(axes[0],np.arange(50,4001,50),losses,TEAL,'Optimizer minibatches')
    axes[0].axhline(np.log(256),color='#87919c',ls=':',lw=1.4,label='Uniform: log(256)')
    axes[0].set(title='Training objective',ylabel='InfoNCE cross-entropy')
    axes[0].legend(fontsize=9)
    band(axes[1],updates,stack('train_nce'),TEAL,'Fixed training objects')
    band(axes[1],updates,stack('heldout_nce'),ORANGE,'Fixed held-out objects','--')
    axes[1].axhline(np.log(256),color='#87919c',ls=':',lw=1.2)
    axes[1].set(title='Same candidates at every evaluation',ylabel='InfoNCE cross-entropy')
    axes[1].legend(fontsize=9)
    band(axes[2],updates,stack('train_raw_mse'),TEAL,'Fixed training objects')
    band(axes[2],updates,stack('heldout_raw_mse'),ORANGE,'Fixed held-out objects','--')
    axes[2].set(title='Scorer drift (not optimized as MSE)',ylabel='Standardized raw MSE: g(f(x)) vs x',yscale='log')
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.set(xlabel='Encoder + critic updates',xlim=(0,4000))
        ax.grid(True,which='major')
        ax.axvline(2000,color='#89939d',ls=':',lw=.8)
    fig.suptitle('Raw observation / own embedding — encoder training\n'
                 '15 sources · median and 10–90% band · fixed evaluations every 25 updates',fontsize=14)
    finish(fig,'training_convergence')

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    band(axes[0],decoder_updates,stack('decoder_train_mse'),TEAL,'Whole training trajectories')
    band(axes[0],decoder_updates,stack('decoder_heldout_mse'),ORANGE,'Whole held-out trajectories','--')
    axes[0].set(title='Native trajectory distribution',ylabel='Whitened next-observation MSE')
    band(axes[1],decoder_updates,stack('decoder_coverage_mse'),PURPLE,'Uniform-distance coverage set')
    axes[1].set(title='Coverage / extrapolation stress test',ylabel='Whitened next-observation MSE')
    for ax in axes:
        ax.set(xlabel='Separate decoder updates (encoder frozen)',yscale='log',xlim=(0,1500))
        ax.grid(True,which='major')
        ax.legend(fontsize=9)
    fig.suptitle('Prediction error convergence — after encoder training\n'
                 '15 sources · median and 10–90% band · full datasets every 50 decoder updates',fontsize=14)
    finish(fig,'prediction_convergence')

    ranked=sorted(records,key=lambda s:(records[s]['prediction']['native_mse'],s))
    selected=[ranked[0],ranked[len(ranked)//2],ranked[-1]]
    roles=['Lowest final held-out MSE','Median final held-out MSE','Highest final held-out MSE']
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    colors=plt.get_cmap('viridis')(np.linspace(.1,.95,5))
    for j,(seed,role) in enumerate(zip(selected,roles)):
        r=records[seed]
        frames={f['step']:f for f in r['frames']}
        for color,step in zip(colors,[0,100,500,2000,4000]):
            g=frames[step]
            axes[0,j].plot(np.exp(g['q']),g['y'],color=color,lw=2 if step==4000 else 1.3,label=f'{step:,}')
        axes[0,j].set(title=f"Seed {seed} · {role}\nfinal prediction MSE {r['prediction']['native_mse']:.6f}",
                       ylabel='Distance representation z\n(arbitrary code units)')
        for method in ('sigreg','vicreg','infonce','raw-own-code'):
            g=r['geometry'] if method=='raw-own-code' else original.base.trials.load(
                original.base.OUT/'runs'/f'{seed}-{method}-c0-n4000.pt')['geometry']
            axes[1,j].plot(np.exp(g['q']),normalize_curve(g),color=COLORS[method],lw=2,
                           label=LABELS[method])
        axes[1,j].set(ylabel='Distance representation z\n(centered, unit std, sign oriented)')
    for ax in axes.flat:
        ax.set(xlabel='Physical distance (evaluation only)',xlim=(1,16))
        ax.grid(True)
    axes[0,0].legend(title='Encoder updates',fontsize=8,title_fontsize=9,ncol=2)
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Learned representations of physical distance\n'
                 'Columns chosen by raw/code final held-out MSE · top: learning over updates · bottom: final objective comparison',fontsize=14)
    finish(fig,'representative_representations')

    # The source discussed earlier is retained as an explicitly identified case study.
    seed=131
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    for method in ('sigreg','vicreg','infonce','raw-own-code'):
        g=records[seed]['geometry'] if method=='raw-own-code' else original.base.trials.load(
            original.base.OUT/'runs'/f'{seed}-{method}-c0-n4000.pt')['geometry']
        y=normalize_curve(g)
        d=np.exp(g['q'])
        axes[0].plot(d,y,color=COLORS[method],label=LABELS[method],lw=2)
        axes[1].plot(d,np.gradient(y,d),color=COLORS[method],lw=2)
    axes[0].set(title='Learned distance representations',ylabel='Distance representation z\n(centered, unit std, sign oriented)')
    axes[0].legend(fontsize=9)
    axes[1].axhline(0,color='#55616c',lw=1)
    axes[1].set(title='Representation slope vs distance',ylabel='Change in displayed code per unit distance')
    for ax in axes:
        ax.set(xlabel='Physical distance (evaluation only)',xlim=(1,16))
        ax.grid(True)
    fig.suptitle('Physical distance → learned scalar code · seed 131\nTeal: latest raw / own-code InfoNCE · code values are not physical distances',fontsize=14)
    finish(fig,'seed131_comparison')

    summary={'date':'2026-11-19','count':15,'selected_seeds':selected,'selection':'min, median, max final held-out MSE',
             'all_encoder_critic_decoder_replays_bit_exact':True,
             'scorer_diagnostics':scorer_diagnostics,
             'encoder_evaluations':len(updates),'decoder_evaluations':len(decoder_updates),
             'encoder_final_median':{k:float(np.median(stack(k)[:,-1])) for k in
                 ('train_nce','heldout_nce','train_raw_mse','heldout_raw_mse')},
             'decoder_final_mean':{k:float(np.mean(stack(k)[:,-1])) for k in
                 ('decoder_train_mse','decoder_heldout_mse','decoder_coverage_mse')},
             'decoder_final_median':{k:float(np.median(stack(k)[:,-1])) for k in
                 ('decoder_train_mse','decoder_heldout_mse','decoder_coverage_mse')},
             'code_hashes':{p.name:digest(p) for p in (HERE/'convergence.py',HERE/'plot_convergence.py')}}
    original.base.trials.atomic_json(HISTORY/'summary.json',summary)
    original.base.trials.atomic_json(HISTORY/'verification.json',dict(count=15,runs=audits,plot_code_sha256=digest(HERE/'plot_convergence.py')))
    lines=['# Dense replay convergence measurements','',
           'All 15 final encoders, critics and downstream decoders reproduce the saved models bit-for-bit.',
           'Shading in figures is the 10th–90th percentile across sources, not a confidence interval.',
           'Encoder fixed-batch raw MSE measures the scorer output against the observation; decoder MSE measures next-observation prediction. These have different targets and units.',
           'The scorer raw MSE initially falls and then rises strongly. The scorer is trained for contrastive classification, not calibrated reconstruction. In this renderer the observations occupy a one-dimensional affine line: a scorer component perpendicular to that line adds the same squared-distance constant to every candidate in a row and therefore cancels from the softmax.',
           f"At the final checkpoint, at least {100*min(r['orthogonal_fraction'] for r in scorer_diagnostics):.3f}% of held-out scorer raw MSE is perpendicular to the observed line in every source. Removing that component changes candidate log probabilities by at most {max(r['max_abs_log_probability_shift_after_projection'] for r in scorer_diagnostics):.3g} in the numerical audit. This identifies unpenalized scorer drift; raw MSE is not a reconstruction convergence claim.",'',
           '## Encoder + critic updates','',
           '| Update | Median training InfoNCE | Median held-out InfoNCE | Median training raw MSE | Median held-out raw MSE |',
           '|---|---:|---:|---:|---:|']
    for step in (0,25,100,250,500,1000,2000,4000):
        ix=np.flatnonzero(updates==step)[0]
        vals=[np.median(stack(k)[:,ix]) for k in ('train_nce','heldout_nce','train_raw_mse','heldout_raw_mse')]
        lines.append('| '+str(step)+' | '+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines+=['','## Separate frozen-encoder decoder updates','',
            '| Update | Median training MSE | Median held-out MSE | Median coverage MSE |',
            '|---|---:|---:|---:|']
    for step in (0,50,100,250,500,1000,1500):
        ix=np.flatnonzero(decoder_updates==step)[0]
        vals=[np.median(stack(k)[:,ix]) for k in ('decoder_train_mse','decoder_heldout_mse','decoder_coverage_mse')]
        lines.append('| '+str(step)+' | '+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines+=['',f"Representative sources (lowest, median, highest final held-out error): {selected}.",
            'Cross-objective curves are centered, scaled to unit standard deviation on the evaluation grid, and globally sign oriented for display. This does not remove folds. Training snapshots retain their original units.',
            'The seed-131 comparison is an explicit earlier case study, not a randomly chosen example. Physical coordinates enter evaluation only. These plots are empirical; there is no injectivity guarantee.','',
            '![Encoder training](../training_convergence.png)','',
            '![Prediction error](../prediction_convergence.png)','',
            '![Representative curves](../representative_representations.png)','',
            '![Seed 131](../seed131_comparison.png)']
    (HISTORY/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    original.torch.set_num_threads(1)
    main()

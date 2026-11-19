"""Independent scalar-area reload audit, reports and distance-representation figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/icm-gradient-cache')
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run

TEAL='#087f8c'


def save(fig,name):
    for extension in ('png','svg'):
        fig.savefig(run.OUT/f'{name}.{extension}',dpi=160,facecolor='white')
    svg = run.OUT / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    records={}; rows=[]; audits=[]
    for seed in range(130,145):
        path=run.OUT/f'{seed}-n4000.pt';r=run.base.trials.load(path)
        c,train,test,source_hash,model,scorer,plan,mean,std=run.setup(seed,4000)
        assert r['source_hash']==source_hash and r['plan_hash']==run.base.sha(plan)
        assert r['code_hash']==run.sha(run.HERE/'run.py') and r['protocol_hash']==run.sha(run.HERE/'PROTOCOL.md')
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,r['initial'][key],rtol=0,atol=0)
        torch.testing.assert_close(mean,r['raw_mean'],rtol=0,atol=0)
        torch.testing.assert_close(std,r['raw_std'],rtol=0,atol=0)
        assert train['x'].shape[1]==test['x'].shape[1]==1
        ref=run.base.trials.load(run.reference.OUT/'runs'/f'{seed}-n4000.pt')
        assert r['plan_hash']==ref['plan_hash'] and r['source_hash']==ref['source_hash']
        model.load_state_dict(r['model']);scorer.load_state_dict(r['critic'])
        # Independent geometry through the analytic inverse-square area renderer.
        # Reuse neither geometry() nor its determinant expression here.
        vertices=run.base.B.sensor(torch.ones(1,dtype=torch.float64),c).reshape(3,2)
        area0=.5*abs(sum(vertices[i,0]*vertices[(i+1)%3,1]-vertices[(i+1)%3,0]*vertices[i,1] for i in range(3)))
        d=torch.exp(torch.linspace(0,np.log(16),20001,dtype=torch.float64)).requires_grad_()
        m=run.Encoder().double();m.load_state_dict(r['model'])
        a=area0/d.square();y=m(a[:,None]);slope=torch.autograd.grad(y.sum(),d)[0]
        torch.testing.assert_close(y.detach(),torch.from_numpy(r['geometry']['code']),rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(slope.detach(),torch.from_numpy(r['geometry']['slope']),rtol=1e-8,atol=1e-10)
        folded=bool((slope[:-1]*slope[1:]<0).any())
        assert folded==r['geometry']['folded']
        np.testing.assert_allclose(slope.detach().numpy(),np.gradient(y.detach().numpy(),d.detach().numpy(),edge_order=2),rtol=.003,atol=2e-6)
        assert r['geometry']['std']>1e-5,'Collapsed representation'
        pairs,stats,common=run.prediction_features(model,train,test,c)
        for key in stats:torch.testing.assert_close(stats[key],r['prediction']['stats'][key],rtol=0,atol=0)
        decoder=run.decoder();decoder.load_state_dict(r['prediction']['model'])
        with torch.no_grad():
            errors={name:torch.nn.functional.mse_loss(decoder(x),target).item() for name,(x,target) in pairs.items()}
            for name,value in errors.items():assert abs(value-r['prediction']['history'][-1][name])<1e-8
            gen=torch.Generator().manual_seed(7400000+10*seed+1)
            ix=torch.randperm(len(test['x']),generator=gen)[:128]
            fixed=torch.cat([test['x'][ix],test['xp'][ix]])
            loss=run.reference.objective((fixed-mean)/std,model(fixed),scorer).item()
            assert loss==r['history'][-1]['infonce']['heldout']
        row=dict(seed=seed,folded=folded,std=r['geometry']['std'],
                 log_distance_r2=r['geometry']['log_distance_r2'],heldout_infonce=loss,
                 **{f'{k}_mse':v for k,v in errors.items()},
                 action_only_heldout_mse=r['prediction']['action_only']['heldout'])
        rows.append(row);records[seed]=r
        audits.append(dict(seed=seed,checkpoint_sha256=run.sha(path),source_and_plan_matched=True,
                           scalar_observations=True,initialization_and_statistics=True,
                           independent_inverse_square_geometry=True,dense_float64_points=20001,
                           physical_derivative_finite_difference=True,decoder_errors_reloaded=True,
                           heldout_infonce_reloaded=True))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,5,figsize=(16,9),layout='constrained')
    for ax,(seed,r) in zip(axes.flat,records.items()):
        g=r['geometry'];ax.plot(g['distance'],g['code'],color=TEAL,lw=2)
        ax.set(title=f"Seed {seed} · {'fold detected' if g['folded'] else 'no detected fold'}",
               xlabel='Physical distance',ylabel='Scalar code z\n(arbitrary units)',xlim=(1,16))
        ax.grid(alpha=.2)
    fig.suptitle('Scalar-area InfoNCE: learned distance representations\n'
                 'Input is one projected-area value · final update 4,000 · original code units',fontsize=15)
    save(fig,'distance_representations')
    r=records[131];g=r['geometry']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    axes[0].plot(g['distance'],g['area'],color=TEAL,lw=2)
    axes[0].set(title='Observation: one scalar area',xlabel='Physical distance',ylabel='Projected area A')
    ids=np.argsort(g['area'])
    axes[1].plot(g['area'][ids],g['code'][ids],color=TEAL,lw=2)
    axes[1].set(title='Encoder: area → scalar code',xlabel='Projected area A',ylabel='Learned code z (arbitrary units)')
    axes[2].plot(g['distance'],g['code'],color=TEAL,lw=2)
    axes[2].set(title='Learned distance representation',xlabel='Physical distance',ylabel='Learned code z (arbitrary units)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Scalar-area InfoNCE only · seed 131\n'
                 'Coordinates are reduced to area before the encoder; distance is evaluation only',fontsize=14)
    save(fig,'scalar_observation_and_representation')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
    updates=[h['step'] for h in r['history']]
    for key,label,style in [('train','Fixed training objects','-'),('heldout','Fixed held-out objects','--')]:
        values=np.array([[h['infonce'][key] for h in rec['history']] for rec in records.values()])
        lo,mid,hi=np.quantile(values,[.1,.5,.9],axis=0)
        axes[0].plot(updates,mid,color=TEAL if key=='train' else '#c27630',ls=style,label=label)
        axes[0].fill_between(updates,lo,hi,color=TEAL if key=='train' else '#c27630',alpha=.12)
    axes[0].axhline(np.log(256),color='gray',ls=':',label='Uniform: log(256)')
    axes[0].set(title='Encoder + scalar critic training',xlabel='Encoder updates',ylabel='InfoNCE cross-entropy');axes[0].legend(fontsize=8)
    for key,label,color in [('heldout','Held-out scalar-area prediction',TEAL),('coverage','Uniform-distance coverage','#8264a8')]:
        values=np.array([[h[key] for h in rec['prediction']['history']] for rec in records.values()])
        lo,mid,hi=np.quantile(values,[.1,.5,.9],axis=0)
        t=[h['step'] for h in r['prediction']['history']]
        axes[1].plot(t,mid,color=color,label=label);axes[1].fill_between(t,lo,hi,color=color,alpha=.12)
    axes[1].set(title='Separate decoder: next-area prediction',xlabel='Decoder updates (encoder frozen)',ylabel='Whitened next-area MSE',yscale='log');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Scalar-area InfoNCE only · 15 reused source datasets\nMedian and 10–90% across sources; different training stages on separate axes',fontsize=14)
    save(fig,'training_and_prediction')
    summary=dict(count=15,folded=sum(row['folded'] for row in rows),
                 minimum_output_std=min(row['std'] for row in rows),
                 mean_heldout_mse=float(np.mean([row['heldout_mse'] for row in rows])),
                 mean_coverage_mse=float(np.mean([row['coverage_mse'] for row in rows])),runs=rows)
    run.base.trials.atomic_json(run.OUT/'summary.json',summary)
    run.base.trials.atomic_json(run.OUT/'verification.json',dict(runs=audits,report_sha256=run.sha(__file__),
        formula_checks=json.loads((run.OUT/'formula-checks.json').read_text())))
    lines=['# Scalar projected-area InfoNCE results','','Completed November 19, 2026. All 15 runs: 4,000 encoder/critic updates and 1,500 separate decoder updates.',
           '',f"Detected final folds: **{summary['folded']}/15**. Minimum output standard deviation: {summary['minimum_output_std']:.6f}.",
           f"Mean held-out next-area MSE: {summary['mean_heldout_mse']:.6f}; uniform-distance coverage MSE: {summary['mean_coverage_mse']:.6f}.",
           '', 'These target scalar area whitened using training-successor statistics. They are not directly comparable with the earlier six-coordinate target MSEs.',
           'The observation is area computed from projected vertices, not the coordinates themselves. Sources/actors are reused from coordinate-trained ICM; no scalar-observation actor retraining was performed.',
           'Independent checkpoint reloads verify source/batch plans, scalar inputs, initialization/statistics, analytic inverse-square area geometry, finite-difference derivatives, InfoNCE and decoder errors.',
           'The fixed-shape area is strictly monotone in distance, but the unconstrained learned encoder has no injectivity guarantee. Finite-grid no-fold checks do not certify the entire interval.',
           '', '| Seed | Fold | Output std | Log-distance R² | Held-out InfoNCE | Held-out next-area MSE | Coverage MSE | Action-only held-out MSE |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {int(row['folded'])} | {row['std']:.4f} | {row['log_distance_r2']:.4f} | {row['heldout_infonce']:.4f} | {row['heldout_mse']:.6f} | {row['coverage_mse']:.6f} | {row['action_only_heldout_mse']:.6f} |")
    lines+=['','![Scalar input and distance representation](scalar_observation_and_representation.png)','',
            '![All final distance representations](distance_representations.png)','',
            '![Training and separate prediction](training_and_prediction.png)']
    (run.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))


if __name__=='__main__':main()

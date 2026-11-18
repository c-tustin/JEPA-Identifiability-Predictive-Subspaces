"""One plainly labeled training history; no new fitting or inferred exact onset."""
from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    torch.set_num_threads(1)
    seed,fraction=131,.25
    _,_,c=reference.source(seed)
    torch.manual_seed(40000+seed)
    initial_model=old.ScaledEncoder(4)
    q,_,initial_slope=base.curve(initial_model,c)
    assert not base.roots(q,initial_slope)
    fig,axes=plt.subplots(2,3,figsize=(12,6.5))
    records={}
    for row,(allocation,color,label) in enumerate((
        ('native','#c44e52','Stays concentrated near the object'),
        ('balanced','#2166ac','Stays spread across distances'))):
        r=old.trials.load(OUT/f'{seed}-f{fraction:g}-{allocation}.pt')
        history=r['history']
        first=next((h['step'] for h in history if h['roots']),None)
        records[allocation]=dict(first_saved_turn=first,
            first_saved_turn_bracket=None if first is None else [first-100,first],
            final_roots=history[-1]['roots'])
        for col,step in enumerate((200,300,4000)):
            h=next(h for h in history if h['step']==step)
            y=h['y']; scaled=(y-y.mean())/y.std()
            ax=axes[row,col]
            ax.plot(np.exp(q),scaled,color=color,lw=2)
            for root in h['roots']:ax.axvline(root,color=color,ls=':',alpha=.8)
            ax.set(title=f"Update {step:,}\n"+('Turn detected' if h['roots'] else 'No turn detected'),
                xlabel='Physical distance',ylabel='Encoder output (scaled)',xlim=(1,16))
            ax.spines[['top','right']].set_visible(False)
        axes[row,0].text(0,1.32,label,transform=axes[row,0].transAxes,color=color,fontsize=11,weight='bold')
    fig.suptitle('When a fold develops: one example, run 131\nEvery batch has 128 original pairs + 384 added stay pairs; only stay locations differ.',y=1.08)
    fig.tight_layout(h_pad=3)
    fig.savefig(OUT/'development_plain.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'development_plain.pdf',bbox_inches='tight')
    plt.close(fig)
    assert records['native']['first_saved_turn']==300
    assert records['balanced']['first_saved_turn'] is None
    old.trials.atomic_json(OUT/'development_plain.json',dict(seed=seed,source_fraction=fraction,
        initial_turn_detected=False,history_interval=100,conditions=records,
        caveat='Snapshot sampling does not exclude brief or inter-grid turns between saved checks.',
        code_sha256=sha(__file__)))
    print(records)

if __name__=='__main__':main()

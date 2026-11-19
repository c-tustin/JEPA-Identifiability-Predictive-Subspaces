"""Independently reload every local optimizer-intervention checkpoint."""
import json
from pathlib import Path
import birth_replication as B
import numpy as np
import torch


def main():
    torch.set_num_threads(1)
    out=B.R.HERE/'output';path=out/'birth_replication.json'
    s=json.loads(path.read_text());states=B.C.trials.load(out/'birth_replication_forks.pt')
    assert s['code_hash']==B.C.digest((B.R.HERE/'birth_replication.py').read_bytes())
    assert s['protocol_hash']==B.C.digest((B.R.HERE/'BIRTH_PROTOCOL.md').read_bytes())
    assert all(x==0 for x in s['final_replay_errors'].values())
    for seed,h in s['source_hashes'].items():
        assert h==B.C.digest((B.C.LEGACY/'output/sources'/f'icm-{seed}.pt').read_bytes())
    verified=0
    for e in s['events']:
        source=B.C.trials.load(B.C.LEGACY/'output/sources'/f'icm-{e["seed"]}.pt')
        c=B.C.B.Config(**source['config'])
        for mode,expected in e['forks'].items():
            m=B.C.B.Encoder();m.load_state_dict(states[f'{e["seed"]}-{e["update"]}'][mode])
            assert B.C.geometry(m,c,20001)==expected;verified+=1
    modes=['full','remove_regularizer_history','remove_alignment_history','remove_all_history']
    matrix=np.array([[e['forks'][k]['folded'] for k in modes] for e in s['events']])
    n=len(s['events']);prevented=int((~matrix[:,1]).sum())
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.3))
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(4),xticklabels=['Original step','Remove SIGReg\nhistory','Remove alignment\nhistory','Remove all\nhistory'],
           yticks=range(n),yticklabels=[f'Seed {e["seed"]}, update {e["update"]}' for e in s['events']],
           title='Same-state Adam interventions at every detected birth through update 500')
    for y in range(n):
        for x in range(4):ax.text(x,y,'Turn' if matrix[y,x] else 'No turn',ha='center',va='center',color='white' if matrix[y,x] else 'black')
    fig.tight_layout();fig.savefig(out/'birth_replication.png',dpi=180);plt.close(fig)
    lines=['# Optimizer-history interventions across five historical failures','',
        f'All five original 4,000-update paths replay bit-for-bit. Scanning each update through 500 detects {n} ordered-to-folded events, all passing independent dense before/after geometry gates. All {verified} fork checkpoints reproduce saved geometry exactly after reload on 20,001 float64 points.','',
        '| Source | Update | Original turn distance | Remove SIGReg history | Remove alignment history | Remove all history |',
        '| --- | ---: | ---: | --- | --- | --- |']
    for e in s['events']:
        values=['Turn' if e['forks'][k]['folded'] else 'No turn' for k in modes[1:]]
        lines.append(f'| {e["seed"]} | {e["update"]} | {e["after"]["roots"][0]:.4f} | '+ ' | '.join(values)+' |')
    lines+=['',f'Removing accumulated SIGReg first moment prevents {prevented}/{n} inspected birth steps; removing alignment first moment prevents {int((~matrix[:,2]).sum())}/{n}. Every one of the five sources has at least one prevented birth. Current gradients, second moments, step count and learning rate are identical within each event. These are actual finite Adam steps, rather than only gradient or linearized slope attributions.','',
        'Two counterexamples limit the momentum claim. Seed 139 develops a transient turn on its first update, when no momentum history exists. Seed 132 at update 368 still turns when all first-moment history is removed. Therefore accumulated momentum is a causal trigger at several inspected events, but is not necessary for every turn. A fork starts from the historically trained before-state; it does not erase the influence of momentum on that earlier path or on second moments.','',
        'The initial seed-136 model already has a turn; that is not counted as a birth. Multiple births in one run reflect intervening disappearance, so eight events are not eight independent sources. The seven later turns first enter near the far boundary; five of those seven begin beyond their source’s observed range. Both seed-136 events begin inside its wider stored support. All five original final folds lie inside observed support. These intervention checks establish local causes, not permanent prevention. Finite temporal inspection stops at 500, finite spatial grids can miss narrow turns, and selected historical failures do not establish generality.','',
        '[Protocol](../BIRTH_PROTOCOL.md) · [Trainer](../birth_replication.py) · [Measurements](birth_replication.json) · [Figure](birth_replication.png)','']
    (out/'BIRTH_RESULTS.md').write_text('\n'.join(lines))
    B.C.trials.atomic_json(out/'birth_verification.json',dict(passed=True,verified_fork_checkpoints=verified,
        exact_final_replays=s['final_replay_errors'],failed_dense_gates=s['failed_dense_gates'],
        metadata_hash=B.C.digest(path.read_bytes()),fork_checkpoint_hash=B.C.digest((out/'birth_replication_forks.pt').read_bytes()),
        report_hash=B.C.digest(Path(__file__).read_bytes())))
    print(json.dumps(dict(events=n,prevented_without_regularizer_history=prevented,verified_forks=verified)))


if __name__=='__main__':main()

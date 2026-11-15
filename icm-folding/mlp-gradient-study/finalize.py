"""Requirement-level artifact and checkpoint audit; regenerate full gallery."""
import hashlib
import json
from pathlib import Path
import re
from run import HERE,OUT,trials


def main():
    audits=[json.loads((OUT/name).read_text()) for name in
        ('verification.json','mining-verification.json','fresh-verification.json','strength-verification.json')]
    assert [a['count'] for a in audits]==[80,30,25,10]
    files=[]
    for audit in audits:
        for r in audit['runs']:
            p=HERE/r['file']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            files.append(p)
    assert len(files)==len(set(files))==145
    fresh=json.loads((OUT/'fresh-summary.json').read_text())
    assert fresh['fresh_policy_seeds']==list(range(140,145)) and fresh['fixed_coefficient']==100
    assert fresh['paired']['prediction_improved']==5
    assert next(r for r in fresh['summary'] if r['label']=='SIGReg')['folded']==2
    assert next(r for r in fresh['summary'] if r['label']=='SIGReg + global mining 100')['folded']==0
    assert next(r for r in fresh['summary'] if r['label']=='VICReg (scalar)')['folded']==1
    assert next(r for r in fresh['summary'] if r['label']=='VICReg (scalar)')['material_folded']==0
    for seed in range(140,145):
        src=trials.load(OUT/'fresh-sources'/f'icm-{seed}.pt')
        assert src['history'][-1]['update']==599 and src['parameter_change']>0
    figures=sorted(OUT.glob('[0-9][0-9]_*.png'))
    assert len(figures)==10
    for p in figures:assert p.with_suffix('.pdf').exists()
    page=['<!doctype html><meta charset="utf-8"><title>Original MLP: gradients and regularizers</title>',
        '<style>body{font:18px system-ui;max-width:1450px;margin:40px auto;padding:0 20px}img{max-width:100%}section{margin:45px 0}</style>',
        '<h1>Original MLP: gradients and regularizers</h1>',
        '<p><a href="../README.md">Explanation and reproduction</a> · <a href="../PROTOCOL.md">Protocol</a> · <a href="RESULTS.md">Main tables</a> · <a href="MINING.md">Global mining</a> · <a href="FRESH.md">Fresh sources</a> · <a href="STRENGTH.md">Strength control</a></p>',
        '<p>145 original-MLP fits; scalar VICReg and distance-score InfoNCE adaptations. Five fresh ICM sources use the fixed mining coefficient. Empirical fold checks are not all-input guarantees.</p>']
    for p in figures:
        title=p.stem[3:].replace('_',' ')
        page.append(f'<section><h2>{title}</h2><a href="{p.with_suffix(".pdf").name}">PDF figure</a><br><img src="{p.name}"></section>')
    (OUT/'index.html').write_text('\n'.join(page)+'\n')
    for path in (HERE/'README.md',HERE/'PROTOCOL.md',HERE.parent/'README.md',HERE.parent/'PROGRESS.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not link.startswith(('http:','https:','#')):
                assert (path.parent/link.split('#')[0]).exists(),(path,link)
    for href in re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text()):
        assert (OUT/href).exists(),href
    relevant=list(HERE.glob('*.py'))+[HERE/'README.md',HERE/'PROTOCOL.md',HERE.parent/'README.md',HERE.parent/'PROGRESS.md']
    result=dict(requirements={
        'unchanged_MLP_from_scratch':'All 145 audited initializations and paired base batch hashes verified by cohort reports.',
        'transition_only_gradients':'Training input dictionaries contain x/xp/action only; loss formula and direction checks passed.',
        'alternative_regularizers':'Four objectives across ten development and five fresh sources; scalar adaptations stated explicitly.',
        'fresh_source_validation':'Five 600-update ICM sources; coefficient 100 fixed before generation; 25 encoder fits verified.',
        'controls':'Local coefficient 1, local coefficient 100, global coefficients 1/10/100; adaptive decisions retained.',
        'prediction':'Independent checkpoint recomputation of held-out and coverage-stress decoder errors for all 145 fits.',
        'geometry':'Independent 20001-point derivative checks for all 145 fits; any versus material folds separately reported.',
        'artifacts':'Ten PNG/PDF figure pairs; all ten PNGs visually reviewed; gallery and repository links checked.',
        'limitations':'No hard guarantee; one substantial mining failure; deterministic scalar simulator; compute and nuisance limits explicit.'},
        audited_encoder_fits=145,independent_final_grid_points=20001,
        newly_trained_policy_sources=5,figure_pairs=10,
        source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant})
    (OUT/'completion-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified 145 unique checkpoint audits, five fresh policies, ten figure pairs, and all document/gallery links.')


if __name__=='__main__':main()

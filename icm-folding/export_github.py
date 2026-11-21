"""Export a portable research snapshot; never traverse the original study symlinks twice.

Usage: python3 icm-folding/export_github.py /path/to/clean/repository
The destination must already contain behavior-policy-experiments from main.
"""
from pathlib import Path
import argparse, hashlib, json, os, re
HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('repository',type=Path);args=parser.parse_args()
    repo=args.repository.resolve();dest=repo/'icm-folding'
    assert (repo/'behavior-policy-experiments/behavior-policy-experiments/qualitative-study/run_trials.py').exists()
    dest.mkdir(exist_ok=True)
    records=[]
    roots=[(WORKSPACE/'explanations/occupancy-folding',Path('sources/explanations/occupancy-folding'))]
    roots += [(HERE/name,Path(name)) for name in ('inverse-occupancy-lr','mlp-gradient-study','no-fold-proof','weighted-moments-validation','occupancy-integral','ema-study','shared-occupancy-ou','icm-matched-ou','fold-development','occupancy-causal-dwell','raw-embedding-infonce','initialization-crossing','scalar-area-infonce','sampling-crossing','loss-source-crossing','occupancy-feature-isolation')]
    roots += [(HERE/name,Path(name)) for name in ('README.md','PROGRESS.md','export_github.py')]
    for source,relative in roots:
        files=[source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
        for p in files:
            local=p.relative_to(source) if source.is_dir() else Path('.')
            targetrel=relative/local if source.is_dir() else relative
            if '__pycache__' in p.parts or p.suffix in ('.pyc','.log','.aux','.out') or p.name=='.DS_Store':continue
            data=p.read_bytes()
            included=p.suffix!='.pt' or ('fresh-sources' in p.parts) or (relative.name=='occupancy-feature-isolation' and p.parent.name=='sources' and p.parent.parent.name=='fresh') or relative.name in ('inverse-occupancy-lr','weighted-moments-validation','ema-study','raw-embedding-infonce','scalar-area-infonce')
            entry=dict(source=str(p.relative_to(WORKSPACE)),destination=str(targetrel),bytes=len(data),source_sha256=sha(data),included=included)
            if included:
                original=data
                # The portable source tree supplies the original dependency layout.
                if targetrel==Path('sources/explanations/occupancy-folding/run_study.py'):
                    data=data.replace(b'ROOT = HERE.parents[1]\nMECH = ROOT /',b'ROOT = HERE.parents[3]  # repository root for provenance paths\nMECH = HERE.parents[1] /')
                if targetrel==Path('occupancy-integral/run.py'):
                    data=data.replace(b"MECH = ROOT/'predictive-closure-workspace/",b"MECH = HERE.parent/'sources/predictive-closure-workspace/")
                if targetrel in (Path('mlp-gradient-study/run.py'),Path('initialization-crossing/run.py')):
                    data=data.replace(b'ROOT = HERE.parents[1]',b'ROOT = HERE.parent / "sources"')
                if targetrel in (Path('README.md'),Path('PROGRESS.md')):
                    text=data.decode().replace('../explanations/','sources/explanations/').replace('../predictive-closure-workspace/','sources/predictive-closure-workspace/')
                    text=text.replace('[AGENTS.md](../AGENTS.md)','[workspace instructions](WORKSPACE_INSTRUCTIONS.md)')
                    text=text.replace('The outer workspace currently has no commits; the existing nested repositories are preserved.','The local source workspace and nested repositories were preserved during publication. See [publication details](PUBLISHING.md).')
                    data=text.encode()
                entry.update(published_sha256=sha(data),adapted=data!=original)
                target=dest/targetrel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records.append(entry)
    aliases={
        'sources/predictive-closure-workspace':'../../behavior-policy-experiments',
        'sources/explanations/tost-icm-sigreg':'../../../behavior-policy-experiments/explanations/tost-icm-sigreg',
        'occupancy-study':'sources/explanations/occupancy-folding',
        'fold-repair':'occupancy-study/gradient-followup',
        'mechanistic-audit':'occupancy-study/mech-interp',
        'matched-concentration':'occupancy-study/matched-concentration',
        'hard-order-guarantees':'occupancy-study/hypothesis-guarantee',
        'earlier-qualitative-study':'sources/predictive-closure-workspace/behavior-policy-experiments/qualitative-study',
        'exposure-graphs':'sources/explanations/tost-icm-sigreg',
    }
    for name,target in aliases.items():
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.is_symlink():p.symlink_to(target)
        else:assert os.readlink(p)==target
        assert p.exists(),(p,target)
    # GitHub document links use canonical directories; keep symlinks for code.
    for record in records:
        if not record['included'] or not record['destination'].endswith('.md'):continue
        path=dest/record['destination']
        def canonical(match):
            target=match.group(1)
            if '://' in target or target.startswith('#'):return match.group(0)
            base,sep,anchor=target.partition('#')
            resolved=(path.parent/base).resolve()
            if not resolved.exists() or not resolved.is_relative_to(repo):return match.group(0)
            relative=os.path.relpath(resolved,path.parent)
            return ']('+relative+(sep+anchor if sep else '')+')'
        data=re.sub(r'\]\(([^)]+)\)',canonical,path.read_text()).encode()
        path.write_bytes(data)
        record.update(published_sha256=sha(data),adapted=sha(data)!=record['source_sha256'])
    (dest/'WORKSPACE_INSTRUCTIONS.md').write_bytes((WORKSPACE/'AGENTS.md').read_bytes())
    (dest/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.DS_Store\n*.log\n*.aux\n*.out\n')
    (dest/'requirements.txt').write_text('numpy\ntorch\nmatplotlib\nscipy\n')
    (dest/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(date='2026-11-21',files=records,aliases=aliases),indent=2)+'\n')
    (dest/'PUBLISHING.md').write_text('''# Portable GitHub snapshot

Updated November 21, 2026 from the local ICM folding workspace. The occupancy-feature-isolation study is included with its code, protocols, prediction locks, cell measurements, audits, figures and the ten fresh policy source datasets (9100–9109); its derived encoder checkpoints stay local and are inventoried in the manifest. Dense raw/code convergence histories and representation plots were recovered by replay, matching all 15 encoder/scorer/decoder checkpoints bit-for-bit; scorer drift is distinguished from downstream prediction error. See raw-embedding-infonce/CONVERGENCE.md. The raw observation/own embedding InfoNCE study includes all 15 new checkpoints, its fixed protocol, independent verification, and final curves. Its report compares with earlier alternative-objective checkpoints, which must first be regenerated using that study's README; the raw/code trainer itself only requires the stored source datasets. The fold-development note derives sufficient slope-reversal conditions from occupancy, encoder response and optimizer state; its small verifier checks algebra and an analytic polynomial example, without claiming a trained-MLP certificate. All study source code, protocols, reports, PNG/PDF figures and JSON audits are included. Each study appears once; named entry points are relative source-folder links. `SOURCE_MANIFEST.json` records the original path and SHA-256, inclusion policy, and published SHA-256 for adapted files.

The existing original qualitative study and its baseline sources are reused from the repository's `behavior-policy-experiments` tree through a relative link. No unrelated branches were merged. The scalar-MLP study's root path and occupancy driver's provenance root were adapted to this layout; loss/training formulas were not changed. Original audit files describe their original code hashes and run paths; use the manifest to distinguish those from the portable paths.

Included raw artifacts: five recent ICM source datasets/policies (140–144), all inverse-occupancy learning-rate checkpoints, and all new weighted-moment validation and EMA checkpoints. Older occupancy/mechanistic/constrained and alternative-objective derived checkpoints remain in the source workspace rather than adding hundreds of megabytes to Git. The 30 derived OU checkpoints likewise remain local (about 515 MB); both OU studies' code, protocols, figures, numerical results, audits and source/checkpoint hashes are included. Their hashes, available reports and inclusion flags are retained in the manifest. These omitted checkpoints must be regenerated before running the corresponding checkpoint-dependent reports; figure and table availability does not imply they are bundled.

For the new studies, run from repository root:

```sh
python3 -m pip install -r icm-folding/requirements.txt
OPENBLAS_NUM_THREADS=1 python3 icm-folding/weighted-moments-validation/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/weighted-moments-validation/report.py
```

To recreate the new shared-input OU experiments and checkpoint-dependent figures, run these commands from repository root (the original source ICM data are already present in the repository):

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/shared-occupancy-ou/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/shared-occupancy-ou/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/shared-occupancy-ou/report.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/icm-matched-ou/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/icm-matched-ou/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/icm-matched-ou/report.py
```

The ICM-matched report also renders the earlier multimodal fits, so run the studies in the displayed order. The shared-loss input identity is audited explicitly; final fold counts are finite-grid empirical checks. The passive OU process targets occupancy and does not reproduce the original actor's action limits or transition dynamics.

The invariant-alignment occupancy experiment can be regenerated independently of those derived checkpoints:

```sh
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-causal-dwell/run.py --check
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-causal-dwell/run.py
OPENBLAS_NUM_THREADS=1 python3 icm-folding/occupancy-causal-dwell/report.py
```

Its derived checkpoints likewise stay local and are inventoried in the manifest. All protocols, figures, results and same-state fork diagnostics are included. The analytic statement concerns cancellation of alignment on stay pairs; final fold checks remain empirical.

To regenerate historical occupancy artifacts, run `icm-folding/sources/explanations/occupancy-folding/run_study.py`, followed by policy diagnostics and its report. The original README's `explanations/...` commands are relative to the source workspace; in this layout use the `icm-folding/sources/` prefix. For the gradient, matched-concentration, and hard-order studies, regenerate their prerequisites in that sequence before their checkpoint-based reports or the no-fold verifier. Existing original qualitative-study baseline data and settings are already present in the repository. The learning-rate report likewise needs its original-rate controls regenerated first. The alternative-objective study is regenerated by the ordered commands in its README, from the repository root.

This snapshot is evidence and reproducible research code. Weighted moments' standard-gain prevention, failure to repair existing folds, input-gain stress failures, and constrained models' analytic guarantees are reported separately.
''')
    print(json.dumps(dict(included=sum(r['included'] for r in records),omitted=sum(not r['included'] for r in records),included_bytes=sum(r['bytes'] for r in records if r['included']))))
if __name__=='__main__':main()

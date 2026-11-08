"""Post hoc sensitivity check: correct all 29 new tests as one family."""
import json
from pathlib import Path
import original
from analyze_matched import holm

output=Path(__file__).resolve().parent/'output'
first=json.loads((output/'statistics.json').read_text())
early=json.loads((output/'early_statistics.json').read_text())
rows=[]
for family,records in [
    ('alignment_removal',first['alignment_removal']),
    ('original_shared_vs_ou',first['planning']['shared_vs_ou']),
    ('original_native_vs_shared',first['planning']['native_vs_shared']),
    ('early_identifiability',early['identifiability']),
    ('early_shared_planning',early['planning']),
]:
    rows.extend([{'comparison':family,**r} for r in records])
for row,adjusted in zip(rows,holm([r['p'] for r in rows])):
    row['global_holm_p']=float(adjusted)
# Independently check the correction using its sorted-prefix definition.
sorted_p=sorted(r['p'] for r in rows)
for row in rows:
    index=max(i for i,p in enumerate(sorted_p) if p<=row['p'])
    expected=min(1.,max((len(sorted_p)-i)*p for i,p in enumerate(sorted_p[:index+1])))
    assert abs(expected-row['global_holm_p'])<1e-14
original.save_json(output/'GLOBAL_CORRECTION.json',{
    'scope':'Post hoc conservative sensitivity check: one Holm family over all 29 newly reported exact tests, in addition to the predeclared separate families. Does not remove exploratory/adaptive study-design limits.',
    'tests':rows,
})
print('Checked one global Holm family:',len(rows),'tests.')

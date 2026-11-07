"""Remove unused backing storage from saved batch-index views, preserving values.

PyTorch otherwise serializes the entire 4,000-update index schedule behind
one saved batch. This changes checkpoint storage only, never training.
"""
import json
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent / 'output'


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def main():
    before_bytes = after_bytes = count = 0
    for path in sorted((OUT / 'snapshots').glob('*.pt')):
        if path.name.endswith('.compact.pt'):
            continue
        saved = torch.load(path, map_location='cpu')
        compact = dict(saved)
        for key in ('alignment_indices', 'regularizer_indices'):
            compact[key] = saved[key].clone()
        temporary = path.with_suffix('.compact.pt')
        torch.save(compact, temporary)
        assert_equal(saved, torch.load(temporary, map_location='cpu'))
        before_bytes += path.stat().st_size
        after_bytes += temporary.stat().st_size
        temporary.replace(path)
        count += 1
    result = {'snapshots_checked': count, 'before_bytes': before_bytes,
              'after_bytes': after_bytes, 'all_saved_values_unchanged': True}
    (OUT / 'SNAPSHOT_STORAGE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()

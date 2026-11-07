"""Prespecified full RND and resumed ICM training gates on cohort seed 6000."""
import json
import multiprocessing as mp
from pathlib import Path
import torch
import run_confirmation as run


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def replay(policy):
    torch.set_num_threads(1)
    run.check_lock()
    seed = 6000
    source = torch.load(run.OUT / 'sources' / f'{policy}_seed{seed}.pt', map_location='cpu')
    output = run.OUT / 'gates' / ('fresh_' + policy)
    result_path = output / f'{policy}_matched_seed{seed}.json'
    if policy == 'icm' and not result_path.exists():
        partial = output / 'partials' / f'{policy}_matched_seed{seed}.pt'
        if not partial.exists():
            run.train_cell(policy, 'matched', seed, source, output, stop_at=500)
        saved = torch.load(partial, map_location='cpu')
        assert saved['step'] == 500
    actual = run.train_cell(policy, 'matched', seed, source, output)
    expected = json.loads((run.OUT / f'{policy}_matched_seed{seed}.json').read_text())
    assert actual == expected
    for name in (f'encoder_{policy}_matched_seed{seed}.pt', f'predictor_{policy}_matched_shared_seed{seed}.pt', f'predictor_{policy}_matched_native_seed{seed}.pt'):
        equal(torch.load(output / name, map_location='cpu'), torch.load(run.OUT / name, map_location='cpu'))
    return {'policy': policy, 'seed': seed, 'resumed_at': 500 if policy == 'icm' else 0, 'exact': True}


if __name__ == '__main__':
    with mp.get_context('spawn').Pool(2) as pool:
        checks = list(pool.imap_unordered(replay, ['icm', 'rnd']))
    run.save_json(run.OUT / 'FRESH_REPLAY_GATES.json', {'signature': run.signature(), 'checks': checks})
    print(json.dumps(checks))

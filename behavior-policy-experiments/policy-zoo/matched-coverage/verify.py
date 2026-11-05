"""Re-evaluate saved encoders, predictors and physical rollouts independently."""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
import matched as exp
import analyze_matched as analyze

HERE = Path(__file__).resolve().parent


def verify_one(path):
    torch.set_num_threads(1)
    record = json.loads(path.read_text())
    name, seed = record['name'], record['seed']
    design = exp.make_design()
    encoder = exp.behavior.Encoder()
    encoder.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cpu')['encoder'])
    encoder.eval()
    metrics, curve = exp.representation_metrics(encoder, design, name)
    for key, value in metrics.items():
        if not np.isclose(value, record['representation'][key], atol=1e-12, rtol=0):
            raise AssertionError((path, key, value))
    arrays = np.load(path.with_suffix('.npz'))
    np.testing.assert_array_equal(curve, arrays['representation'])
    for kind in ['shared', 'native']:
        saved = torch.load(path.with_name(path.stem + '_' + kind + '_predictor.pt'), map_location='cpu')
        predictor = exp.planning.Predictor()
        predictor.load_state_dict(saved['predictor'])
        predictor.eval()
        replay_metrics, replay_paths = exp.planning_evaluation(encoder, predictor, saved['mean'], saved['scale'], seed)
        for key, value in replay_metrics.items():
            if not np.isclose(value, record['planning'][kind][key], atol=1e-12, rtol=0):
                raise AssertionError((path, kind, key))
        for key, value in replay_paths.items():
            np.testing.assert_array_equal(value, arrays[kind + '_' + key])
    return path.name


def main():
    output = HERE / 'output'
    paths = sorted(output.glob('*_seed*.json'))
    with mp.get_context('spawn').Pool(6) as pool:
        verified = list(pool.imap_unordered(verify_one, paths))
    rows = [json.loads(p.read_text()) for p in paths]
    comparisons = analyze.inference(rows)
    for row in comparisons:
        n = row['wins'] + row['losses']
        expected = binomtest(row['wins'], n, .5).pvalue if n else 1.
        assert np.isclose(row['p'], expected, rtol=0, atol=1e-14)
    saved = json.loads((output / 'comparisons.json').read_text())
    assert comparisons == saved
    result = {'encoder_evaluations': len(verified), 'planning_replays': 2 * len(verified),
              'comparisons_checked': len(comparisons), 'errors': 0}
    (output / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

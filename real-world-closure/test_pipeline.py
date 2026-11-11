import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from data import prepare, read_rows, fetch
from models import JEPA, sigreg
from run import probe, ridge_fit, ridge_predict, weather_input


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values, cls.times = read_rows(fetch(Path(__file__).parent / 'data/beijing.csv'))

    def test_chronological_pairs_and_no_standardization_leakage(self):
        splits, meta = prepare(self.values, self.times, 6)
        for name, part in splits.items():
            self.assertTrue(np.all(part['end_indices'] - part['indices'] == 6))
            self.assertTrue(np.isfinite(part['x']).all())
            self.assertTrue(np.isfinite(part['y']).all())
            self.assertEqual(len(part['x']), len(part['times']))
        self.assertLess(splits['train']['end_indices'].max(), splits['val']['indices'].min())
        self.assertLess(splits['val']['end_indices'].max(), splits['test']['indices'].min())
        changed = self.values.copy()
        changed[self.times >= np.datetime64('2013-01-01'), :7] += 100
        after, after_meta = prepare(changed, self.times, 6)
        np.testing.assert_array_equal(splits['train']['x'], after['train']['x'])
        self.assertEqual(meta['train_mean'], after_meta['train_mean'])
        self.assertEqual(meta['train_scale'], after_meta['train_scale'])

    def test_missing_rows_never_compress_calendar(self):
        changed = self.values.copy()
        changed[100:110, 0] = np.nan
        split, _ = prepare(changed, self.times, 6)
        a, b = split['train']['indices'], split['train']['end_indices']
        self.assertFalse(np.isin(a, np.arange(100, 110)).any())
        self.assertFalse(np.isin(b, np.arange(100, 110)).any())
        self.assertTrue(np.all(self.times[b] - self.times[a] == np.timedelta64(6, 'h')))

    def test_sigreg_matches_pinned_upstream(self):
        path = Path(__file__).resolve().parents[2] / 'predictive-closure-upstream/jepa/losses.py'
        if not path.exists():
            self.skipTest('Optional pinned upstream checkout not present')
        spec = importlib.util.spec_from_file_location('reference_losses', path)
        reference = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        z = torch.randn(2, 32, 3, requires_grad=True)
        torch.manual_seed(17)
        ours = sigreg(z)
        torch.manual_seed(17)
        theirs = reference.SIGReg(num_projections=32, expected_time_steps=2)(z)
        torch.testing.assert_close(ours, theirs)
        ga, = torch.autograd.grad(ours, z, retain_graph=True)
        gb, = torch.autograd.grad(theirs, z)
        torch.testing.assert_close(ga, gb)

    def test_gradients_reach_both_endpoints(self):
        model = JEPA(11, 2, 10)
        x, y = torch.randn(32, 11, requires_grad=True), torch.randn(32, 11, requires_grad=True)
        z, target, prediction = model(x, y, x[:, 1:])
        (0.9 * (prediction - target).square().mean() + 0.1 * sigreg(torch.stack([z, target]))).backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(y.grad.abs().sum().item(), 0)

    def test_shuffle_preserves_weather_rows_and_only_changes_predictor_input(self):
        x = torch.arange(220, dtype=torch.float32).reshape(20, 11)
        generator = torch.Generator().manual_seed(1)
        shuffled = weather_input(x, 'shuffled_weather', generator)
        self.assertEqual(sorted(shuffled[:, 0].tolist()), sorted(x[:, 1].tolist()))
        self.assertFalse(torch.equal(shuffled, x[:, 1:]))
        self.assertEqual(weather_input(x, 'none', generator).abs().sum().item(), 0)

    def test_probe_selection_does_not_use_test_targets(self):
        rng = np.random.default_rng(15)
        features = {key: rng.normal(size=(100, 3)) for key in ['train', 'val', 'test']}
        targets = {key: value[:, 0] + rng.normal(size=100) for key, value in features.items()}
        first, _ = probe(features, targets)
        targets['test'] = rng.normal(size=100) * 100
        second, _ = probe(features, targets)
        self.assertEqual(first['alpha'], second['alpha'])

    def test_cross_feature_dependency_can_beat_self_predictability(self):
        # U is iid; V[t+1]=U[t]+noise; W is an unrelated AR(1).
        # Fixed two-coordinate sets: [U,V] has a useful off-diagonal link,
        # despite both coordinates having zero self-predictability.
        rng = np.random.default_rng(27)
        n = 100000
        u, v, w = rng.normal(size=(3, n))
        future = np.stack([rng.normal(size=n), u + .05*rng.normal(size=n),
                           .5*w + np.sqrt(.75)*rng.normal(size=n)], axis=1)
        current = np.stack([u, v, w], axis=1)
        self_correlation = np.corrcoef(u, future[:, 0])[0, 1]
        self.assertLess(abs(self_correlation), .02)
        errors = []
        for columns in [[0, 1], [0, 2]]:
            x, y = current[:, columns], future[:, columns]
            fit = ridge_fit(x[:50000], y[:50000], .001)
            errors.append(np.mean((ridge_predict(fit, x[50000:]) - y[50000:])**2))
        self.assertLess(errors[0], errors[1] - .25)


if __name__ == '__main__':
    unittest.main()

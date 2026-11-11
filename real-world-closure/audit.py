"""Frozen-model diagnostics closer to the set-level prediction in the theory.

No encoder training or test-driven hyperparameter selection. Linear readout gaps
are operational diagnostics; they are neither MI estimates nor causal effects.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import FEATURES, fetch, prepare, read_rows
from models import JEPA
from run import encode, evaluate_embeddings, probe, weather_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--data', type=Path, help='Override the data path recorded on the training machine')
    args = parser.parse_args()
    output = json.loads((args.output / 'metrics.json').read_text())
    config = output['config']
    torch.set_num_threads(config['threads'])
    data_path = args.data or Path(__file__).resolve().parent / 'data/beijing.csv'
    split, _ = prepare(*read_rows(fetch(data_path)), config['horizon'])
    results = {'interpretation': 'Linear forecast gaps with frozen encoders; no mutual-information or causal claim',
               'runs': [], 'self_predictability': {}, 'additional_baselines': {}}
    results['checkpoint_sha256'] = {}
    # The first run's exact source files are preserved beside the checkpoints.
    for name, expected in output['source_sha256'].items():
        snapshot = args.output / 'source' / name
        if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source snapshot hash mismatch: {name}')
    # A reference based only on individual feature predictability, selected on validation.
    val_scores = []
    for j, name in enumerate(FEATURES):
        x, y = split['train']['x'][:, j], split['train']['y'][:, j]
        from run import ridge_fit, ridge_predict
        fit = ridge_fit(x[:, None].astype(float), y.astype(float), 1.)
        target = split['val']['y'][:, j]
        error = np.mean((ridge_predict(fit, split['val']['x'][:, j:j+1]) - target) ** 2)
        score = float(1 - error / np.var(target))
        results['self_predictability'][name] = score
        val_scores.append(score)
    for latent in config['latent_dims']:
        indices = np.argsort(val_scores)[-latent:][::-1]
        selected = {k: v['x'][:, indices].astype(float) for k, v in split.items()}
        results['additional_baselines'][f'top_self_k{latent}'] = dict(
            selected_features=[FEATURES[j] for j in indices], **evaluate_embeddings(selected, split))
    # A raw-unit pollution baseline avoids favoring a log-transformed linear predictor.
    raw_pm = {k: v['pm_now'][:, None] for k, v in split.items()}
    raw_weather = {k: np.concatenate([v['pm_now'][:, None], v['x'][:, 1:]], axis=1)
                   for k, v in split.items()}
    targets = {k: v['pm_next'] for k, v in split.items()}
    for name, features in [('raw_pm_ridge', raw_pm), ('raw_pm_and_weather_ridge', raw_weather)]:
        results['additional_baselines'][name], _ = probe(features, targets)
    for row in output['runs']:
        checkpoint_path = args.output / (row['tag'] + '.pt')
        results['checkpoint_sha256'][checkpoint_path.name] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = JEPA(len(FEATURES), row['latent_dim'], len(FEATURES) - 1)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        z = encode(model, split)
        reproduced_forecast, _ = probe(z, targets)
        for name in ['r2', 'rmse', 'mae', 'alpha']:
            np.testing.assert_allclose(reproduced_forecast[name], row['forecast'][name], rtol=1e-7, atol=1e-8)
        future_split = {k: dict(x=v['y']) for k, v in split.items()}
        future_z = encode(model, future_split)
        weather = {k: np.concatenate([z[k], v['x'][:, 1:]], axis=1) for k, v in split.items()}
        full = {k: v['x'].astype(float) for k, v in split.items()}
        own, _ = probe(z, future_z)
        supplied, _ = probe(weather, future_z)
        all_observed, _ = probe(full, future_z)
        # Conditional-mean residual checks on the learned neural predictor.
        with torch.no_grad():
            x, y = [torch.from_numpy(split['test'][k]) for k in ['x', 'y']]
            _, target, pred = model(x, y, weather_input(x, row['mode'], torch.Generator().manual_seed(7701)))
            residual = (target - pred).numpy().astype(float)
        covariance = np.atleast_2d(np.cov(residual, rowvar=False))
        eigen = np.linalg.eigvalsh(covariance)
        centered = residual - residual.mean(0)
        kurtosis = np.mean(centered**4, axis=0) / np.maximum(np.mean(centered**2, axis=0)**2, 1e-12) - 3
        results['runs'].append(dict(tag=row['tag'],
            mean_future_latent_r2_from_z=float(np.mean(own['r2'])),
            mean_future_latent_r2_from_z_and_weather=float(np.mean(supplied['r2'])),
            mean_future_latent_r2_from_full_observation=float(np.mean(all_observed['r2'])),
            weather_gain_in_latent_r2=float(np.mean(supplied['r2']) - np.mean(own['r2'])),
            residual_covariance_eigenvalues=eigen.tolist(),
            residual_covariance_condition_number=float(eigen[-1] / max(eigen[0], 1e-12)),
            residual_excess_kurtosis=kurtosis.tolist()))
    (args.output / 'diagnostics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f"Verified frozen forecast readouts from {len(results['runs'])} saved checkpoints")
    print(json.dumps(results['additional_baselines'], indent=2))


if __name__ == '__main__':
    main()

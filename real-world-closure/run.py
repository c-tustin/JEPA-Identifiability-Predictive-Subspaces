"""Run an exploratory, CPU-sized real-data closure/side-information study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from data import FEATURES, fetch, prepare, read_rows
from models import JEPA, sigreg

ROOT = Path(__file__).resolve().parent
ALPHAS = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]


def ridge_fit(x, y, alpha):
    """Centered, train-standardized ridge; intercept is unpenalized."""
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    a = (x - mean) / scale
    ym = y.mean(0)
    weight = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y - ym))
    return mean, scale, ym, weight


def ridge_predict(fit, x):
    mean, scale, ym, weight = fit
    return ((x - mean) / scale) @ weight + ym


def probe(features, targets):
    """Choose each readout's alpha using validation only, then open test."""
    targets = {k: np.asarray(v, dtype=np.float64) for k, v in targets.items()}
    if targets['train'].ndim == 1:
        targets = {k: v[:, None] for k, v in targets.items()}
    predictions, selections = [], []
    for column in range(targets['train'].shape[1]):
        y = targets['train'][:, column]
        candidates = [ridge_fit(features['train'], y, alpha) for alpha in ALPHAS]
        errors = [np.mean((ridge_predict(fit, features['val']) - targets['val'][:, column]) ** 2)
                  for fit in candidates]
        best = int(np.argmin(errors))
        selections.append(ALPHAS[best])
        predictions.append(ridge_predict(candidates[best], features['test']))
    pred = np.stack(predictions, axis=1)
    truth = targets['test']
    residual = np.mean((pred - truth) ** 2, axis=0)
    variance = np.mean((truth - truth.mean(0)) ** 2, axis=0)
    return dict(r2=(1 - residual / np.maximum(variance, 1e-12)).tolist(),
                rmse=np.sqrt(residual).tolist(), mae=np.abs(pred - truth).mean(0).tolist(),
                alpha=selections), pred


@torch.no_grad()
def encode(model, split):
    result = {}
    for key, data in split.items():
        result[key] = torch.cat([model.encoder(part) for part in
                                torch.from_numpy(data['x']).split(2048)]).numpy().astype(np.float64)
    return result


def weather_input(x, mode, generator):
    if mode == 'weather':
        return x[:, 1:]
    if mode == 'shuffled_weather':
        return x[torch.randperm(len(x), generator=generator), 1:]
    if mode == 'none':
        return torch.zeros_like(x[:, 1:])
    raise ValueError(mode)


def train(split, latent, mode, seed, args, out):
    torch.manual_seed(seed)
    model = JEPA(len(FEATURES), latent, len(FEATURES) - 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Separate generators make source pairs identical across the three conditions.
    pair_rng = torch.Generator().manual_seed(seed + 10000)
    shuffle_rng = torch.Generator().manual_seed(seed + 20000)
    x, y = [torch.from_numpy(split['train'][k]) for k in ['x', 'y']]
    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        indices = torch.randint(len(x), (args.batch_size,), generator=pair_rng)
        a, b = x[indices], y[indices]
        z, target, prediction = model(a, b, weather_input(a, mode, shuffle_rng))
        prediction_loss = (prediction - target).square().mean()
        regularizer = sigreg(torch.stack([z, target]), args.projections)
        loss = 0.9 * prediction_loss + 0.1 * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == args.steps:
            history.append(dict(step=step + 1, prediction=float(prediction_loss.detach()),
                                sigreg=float(regularizer.detach())))
    tag = f'k{latent}_{mode}_seed{seed}'
    torch.save(dict(model=model.state_dict(), config=vars(args), latent=latent, mode=mode,
                    seed=seed, preprocessing=out['data'], history=history),
               Path(args.output) / f'{tag}.pt')
    model.eval()
    embeddings = encode(model, split)
    result = evaluate_embeddings(embeddings, split)
    with torch.no_grad():
        test = split['test']
        a, b = [torch.from_numpy(test[k]) for k in ['x', 'y']]
        z, target, prediction = model(a, b, weather_input(a, mode, torch.Generator().manual_seed(7701)))
        mse = (prediction - target).square().mean().item()
        future_var = target.var(dim=0, unbiased=False).mean().item()
        result['latent_prediction_mse'] = mse
        result['latent_prediction_mse_over_future_variance'] = mse / max(future_var, 1e-12)
        result['test_latent_variances'] = z.var(dim=0, unbiased=False).tolist()
        result['test_covariance_eigenvalues'] = torch.linalg.eigvalsh(torch.cov(z.T)).tolist()
    result.update(tag=tag, mode=mode, latent_dim=latent, seed=seed, history=history,
                  elapsed_seconds=time.perf_counter() - started)
    print(json.dumps({k: result[k] for k in ['tag', 'forecast', 'elapsed_seconds']}), flush=True)
    return result


def evaluate_embeddings(features, split):
    readout, _ = probe(features, {k: v['x'] for k, v in split.items()})
    forecast, pred = probe(features, {k: v['pm_next'] for k, v in split.items()})
    # Frozen-representation leakage diagnostic: how much can omitted raw weather
    # improve a linear forecast? This is an MSE diagnostic, not an MI estimator.
    augmented = {k: np.concatenate([features[k], v['x'][:, 1:]], axis=1) for k, v in split.items()}
    augmented_forecast, _ = probe(augmented, {k: v['pm_next'] for k, v in split.items()})
    # Block summaries expose temporal heterogeneity; they are not independent samples.
    dates = split['test']['times'].astype('datetime64[h]').astype('datetime64[M]').astype(str)
    error = pred[:, 0] - split['test']['pm_next']
    monthly = {month: float(np.sqrt(np.mean(error[dates == month] ** 2))) for month in np.unique(dates)}
    return dict(readout_r2=dict(zip(FEATURES, readout['r2'])), readout_alpha=readout['alpha'],
                forecast=forecast, forecast_with_weather=augmented_forecast, monthly_rmse=monthly)


def baselines(split, latent_dims, seeds):
    targets = {k: v['pm_next'] for k, v in split.items()}
    raw = {k: v['x'].astype(np.float64) for k, v in split.items()}
    result = {}
    result['raw_ridge'] = evaluate_embeddings(raw, split)
    result['pm_only_ridge'], _ = probe({k: v[:, :1] for k, v in raw.items()}, targets)
    y, now = split['test']['pm_next'], split['test']['pm_now']
    result['persistence'] = dict(rmse=[float(np.sqrt(np.mean((y - now) ** 2)))],
                                 mae=[float(np.mean(np.abs(y - now)))],
                                 r2=[float(1 - np.mean((y - now) ** 2) / np.var(y))])
    mean = raw['train'].mean(0)
    _, _, v = np.linalg.svd(raw['train'] - mean, full_matrices=False)
    for latent in latent_dims:
        pca = {k: (a - mean) @ v[:latent].T for k, a in raw.items()}
        result[f'pca_k{latent}'] = evaluate_embeddings(pca, split)
        for seed in seeds:
            torch.manual_seed(seed)
            model = JEPA(len(FEATURES), latent, len(FEATURES) - 1)
            result[f'random_k{latent}_seed{seed}'] = evaluate_embeddings(encode(model, split), split)
    return result


def summarize(output, directory):
    horizon = output['config']['horizon']
    lines = ['# Beijing predictive-closure pilot', '',
             'Exploratory real-data extension, not an original-paper replication or a causal test.', '',
             f'| Latent | Condition | {horizon}-hour PM2.5 RMSE (seed mean ± SD) | PM2.5 readout R² |',
             '| --- | --- | ---: | ---: |']
    for latent in output['config']['latent_dims']:
        for mode in output['config']['modes']:
            rows = [r for r in output['runs'] if r['latent_dim'] == latent and r['mode'] == mode]
            if not rows:
                continue
            values = np.array([r['forecast']['rmse'][0] for r in rows])
            recovery = np.mean([r['readout_r2']['log_pm25'] for r in rows])
            lines.append(f'| {latent} | {mode} | {values.mean():.2f} ± {values.std(ddof=1) if len(values)>1 else 0:.2f} | {recovery:.3f} |')
    lines += ['', 'Units: μg/m³. SD measures initialization/training variability on one fixed test year; '
              'it does not measure uncertainty across datasets or years.', '', 'Baselines:']
    for name in ['persistence', 'pm_only_ridge', 'raw_ridge']:
        row = output['baselines'][name]
        if 'forecast' in row:
            row = row['forecast']
        lines.append(f"- {name}: RMSE {row['rmse'][0]:.2f}; R² {row['r2'][0]:.3f}.")
    lines += ['', 'See metrics.json for all seeds, validation-selected ridge penalties, per-feature '
              'readouts, PCA/random baselines, collapse diagnostics and month-by-month errors.',
              '', 'The held-out comparisons describe this fixed pilot. They do not establish '
              'cross-subsidization, mutual-information maximization, or causal weather effects.']
    (directory / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def plot(output, directory):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/closure-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dims, modes = output['config']['latent_dims'], output['config']['modes']
    horizon = output['config']['horizon']
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 4.2), squeeze=False)
    for ax, latent in zip(axes[0], dims):
        for j, mode in enumerate(modes):
            values = [r['forecast']['rmse'][0] for r in output['runs']
                      if r['latent_dim'] == latent and r['mode'] == mode]
            ax.scatter(np.full(len(values), j), values, s=26, color=f'C{j}', alpha=.7)
            ax.plot([j-.18, j+.18], [np.mean(values)]*2, color=f'C{j}', lw=3)
        ax.axhline(output['baselines']['persistence']['rmse'][0], color='.5', ls='--', label='Persistence')
        ax.axhline(output['baselines']['raw_ridge']['forecast']['rmse'][0], color='k', ls=':', label='Raw ridge')
        ax.set(xticks=range(len(modes)), xticklabels=[m.replace('_', '\n') for m in modes],
               ylabel=f'{horizon}-hour PM2.5 RMSE (μg/m³); lower is better', title=f'{latent}-dimensional latent')
        ax.legend(fontsize=8)
    fig.suptitle('2014 holdout • points = training seeds • bars = seed means')
    fig.tight_layout()
    fig.savefig(directory / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data/beijing.csv'))
    parser.add_argument('--output', default=str(ROOT / 'output-pilot'))
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--latent-dims', type=int, nargs='+', default=[2, 8])
    parser.add_argument('--modes', nargs='+', choices=['none', 'weather', 'shuffled_weather'],
                        default=['none', 'weather', 'shuffled_weather'])
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--projections', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.projections < 1 or args.threads < 1 or args.lr <= 0:
        parser.error('Invalid training parameters')
    if any(k < 1 or k > len(FEATURES) for k in args.latent_dims):
        parser.error('Invalid latent dimension')
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    split, metadata = prepare(*read_rows(fetch(args.data)), args.horizon)
    output = dict(config=vars(args), data=metadata, runs=[],
                  runtime=dict(python=platform.python_version(), torch=torch.__version__,
                               numpy=np.__version__, platform=platform.platform(), device='cpu'),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in ROOT.glob('*.py')})
    snapshot = directory / 'source'
    snapshot.mkdir()
    for name in output['source_sha256']:
        (snapshot / name).write_bytes((ROOT / name).read_bytes())
    output['baselines'] = baselines(split, args.latent_dims, args.seeds)
    for latent in args.latent_dims:
        for seed in args.seeds:
            for mode in args.modes:
                output['runs'].append(train(split, latent, mode, seed, args, output))
                (directory / 'metrics.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    summarize(output, directory)
    plot(output, directory)
    print(directory / 'RESULTS.md')


if __name__ == '__main__':
    main()

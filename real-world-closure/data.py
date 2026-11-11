"""Chronological, endpoint-complete hourly Beijing pairs; no imputation."""
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = 'https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip'
ARCHIVE_SHA256 = '40cdd29c24918e855133dce280ff59779bb0b5852898e05bafcf422a4cc0a175'
CSV_SHA256 = '4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c'
NUMERIC = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
WINDS = ['NE', 'NW', 'SE', 'cv']
FEATURES = ['log_pm25', 'dewpoint', 'temperature', 'pressure', 'log_cum_wind',
            'log_snow', 'log_rain'] + ['wind_' + w for w in WINDS]


def fetch(path):
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            blob = response.read()
        if hashlib.sha256(blob).hexdigest() != ARCHIVE_SHA256:
            raise ValueError('UCI archive hash differs from the pinned download')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            blob = archive.read('PRSA_data_2010.1.1-2014.12.31.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError('CSV hash differs from pinned UCI data')
    return path


def read_rows(path):
    values, times = [], []
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            times.append(np.datetime64(f"{int(row['year']):04}-{int(row['month']):02}-"
                                      f"{int(row['day']):02}T{int(row['hour']):02}", 'h'))
            numeric = [float(row[k]) if row[k] != 'NA' else np.nan for k in NUMERIC]
            wind = row['cbwd']
            if wind not in WINDS:
                raise ValueError(f'Unknown wind category: {wind}')
            values.append(numeric + [float(wind == w) for w in WINDS])
    values, times = np.asarray(values, dtype=np.float64), np.asarray(times)
    if not np.all(np.diff(times) == np.timedelta64(1, 'h')):
        raise ValueError('Expected a unique, continuous hourly calendar')
    return values, times


def prepare(values, times, horizon):
    if not isinstance(horizon, int) or horizon < 1 or horizon >= len(times):
        raise ValueError('horizon must be a positive integer smaller than the series')
    transformed = values.copy()
    transformed[:, [0, 4, 5, 6]] = np.log1p(transformed[:, [0, 4, 5, 6]])
    valid = np.isfinite(transformed).all(axis=1)
    train_rows = valid & (times < np.datetime64('2013-01-01'))
    mean = transformed[train_rows].mean(0)
    scale = transformed[train_rows].std(0)
    scale = np.maximum(scale, 1e-6)
    x = ((transformed - mean) / scale).astype(np.float32)
    starts = np.arange(len(times) - horizon)
    ends = starts + horizon
    complete = valid[starts] & valid[ends]
    split = {}
    boundaries = [('train', '2010-01-01', '2013-01-01'),
                  ('val', '2013-01-01', '2014-01-01'),
                  ('test', '2014-01-01', '2015-01-01')]
    for name, low, high in boundaries:
        keep = complete & (times[starts] >= np.datetime64(low)) & (times[ends] < np.datetime64(high))
        a, b = starts[keep], ends[keep]
        if len(a) < 100:
            raise ValueError(f'Too few {name} pairs')
        split[name] = dict(x=x[a], y=x[b], pm_now=values[a, 0], pm_next=values[b, 0],
                           times=times[a].astype(str), indices=a, end_indices=b)
    metadata = dict(horizon_hours=horizon, features=FEATURES, train_mean=mean.tolist(),
                    train_scale=scale.tolist(), rows=len(times), complete_rows=int(valid.sum()),
                    pairs={k: len(v['x']) for k, v in split.items()}, csv_sha256=CSV_SHA256,
                    url=URL, split='2010–2012 train; 2013 validation; 2014 test',
                    missing='Drop pairs with missing endpoints; preserve calendar offsets; no imputation')
    return split, metadata


if __name__ == '__main__':
    path = fetch(Path(__file__).parent / 'data/beijing.csv')
    _, metadata = prepare(*read_rows(path), horizon=6)
    print(json.dumps(metadata, indent=2))

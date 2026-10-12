"""
Walk-Forward Validation for the BUY Detector

binary_search.py splits train/test with train_test_split(shuffle=False) on rows
concatenated symbol-by-symbol, so its "test set" is the last ~20% of *symbols*
over the same dates as training — market-wide moves leak across the split.
This script instead splits by date:

  - Expanding-window folds: train on everything before a test window, test on
    the next `--test-months` months, step forward, repeat.
  - Purge: a training row is dropped if its label window (the next `horizon`
    bars) reaches into the test window, so no training label sees test prices.

For comparison it also re-runs the legacy symbol-ordered split on the same data,
so the size of the leak is visible directly.

Per fold it reports ROC-AUC, PR-AUC (and lift over the base rate), precision at
the champion's frozen threshold, and top-k-per-bar precision — the closest
offline analogue of the live bot buying the top 5 scored names per run.

Usage:
    poetry run python ml/walk_forward.py                       # latest champion params, 400 random symbols
    poetry run python ml/walk_forward.py --max-files 0         # all symbols (training PC)
    poetry run python ml/walk_forward.py --params models/binary_search_results/champion_params_<ts>.json
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.binary_feature_builder import BinaryFeatureBuilder
from ml.binary_search import BinaryHyperparameterSearch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RESULTS_DIR = "models/walk_forward_results"
CACHE_DIR = "saved_data/feature_cache"


def latest_champion_params(results_dir="models/binary_search_results"):
    files = sorted(glob.glob(f"{results_dir}/champion_params_*.json"))
    if not files:
        raise FileNotFoundError(f"No champion_params_*.json in {results_dir}")
    return files[-1]


def _cache_key(p):
    return (f"w{p['window_size']}_h{p['horizon']}_"
            f"tp{p['take_profit']}_sl{p['stop_loss']}_combined")


def build_symbol(args):
    """Build features for one parquet file. Returns (symbol, X, y, times, label_end) or None.

    `times` is each sample's label-bar timestamp; `label_end` is the timestamp of
    the last bar its triple-barrier label looked at (used for purging).
    """
    filepath, params, cache_dir = args
    symbol = os.path.splitext(os.path.basename(filepath))[0]
    cache_path = os.path.join(cache_dir, _cache_key(params), f"{symbol}.npz") if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        return symbol, d['X'], d['y'], d['times'], d['label_end']

    try:
        df = pd.read_parquet(filepath)
        if len(df) < 100:
            return None
        fb = BinaryFeatureBuilder(
            window_size=params['window_size'],
            horizon=params['horizon'],
            take_profit=params['take_profit'],
            stop_loss=params['stop_loss'],
            feature_mode='combined',
        )
        X, y = fb.build_features(df)
        if len(X) == 0:
            return None

        idx = pd.DatetimeIndex(df.index)
        times = pd.DatetimeIndex(fb.sample_times)
        pos = idx.get_indexer(times)
        end_pos = np.minimum(pos + params['horizon'], len(idx) - 1)

        out = (
            symbol,
            X.reshape(len(X), -1).astype(np.float32),
            y.astype(np.int8),
            times.asi8,
            idx[end_pos].asi8,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, X=out[1], y=out[2], times=out[3], label_end=out[4])
        return out
    except Exception as e:
        logging.warning(f"{symbol}: {e}")
        return None


def topk_per_bar_precision(times, y_true, y_proba, k):
    """At each bar timestamp, take the k highest-scored symbols; return their hit
    rate and the all-symbol base rate over the same bars."""
    df = pd.DataFrame({'t': times, 'y': y_true, 'p': y_proba})
    df = df[df.groupby('t')['t'].transform('size') > k]   # only bars with a real choice
    if df.empty:
        return np.nan, np.nan, 0
    top = df.sort_values('p', ascending=False).groupby('t').head(k)
    return top['y'].mean(), df['y'].mean(), len(top)


def evaluate(searcher, params, threshold, X_tr, y_tr, X_te, y_te, t_te, top_k):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    model = searcher.get_classifier(params['classifier'], params['n_estimators'], params['max_depth'])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    base = y_te.mean()
    pr_auc = average_precision_score(y_te, proba)
    fired = proba >= threshold
    topk_prec, topk_base, topk_n = topk_per_bar_precision(t_te, y_te, proba, top_k)
    return {
        'n_train': len(y_tr),
        'n_test': len(y_te),
        'base_rate': base,
        'roc_auc': roc_auc_score(y_te, proba),
        'pr_auc': pr_auc,
        'pr_lift': pr_auc / base if base > 0 else np.nan,
        'signals_at_thresh': int(fired.sum()),
        'precision_at_thresh': y_te[fired].mean() if fired.any() else np.nan,
        f'top{top_k}_precision': topk_prec,
        f'top{top_k}_base_rate': topk_base,
        f'top{top_k}_lift': topk_prec - topk_base,
        f'top{top_k}_n': topk_n,
    }


def make_folds(t_min, t_max, min_train_months, test_months):
    folds = []
    start = (t_min + pd.DateOffset(months=min_train_months)).normalize()
    while start < t_max:
        end = start + pd.DateOffset(months=test_months)
        folds.append((start, min(end, t_max + pd.Timedelta(seconds=1))))
        start = end
    return folds


def main():
    ap = argparse.ArgumentParser(description="Walk-forward validation for the BUY detector")
    ap.add_argument('--params', default=None, help='champion_params_*.json (default: latest)')
    ap.add_argument('--data-dir', default='saved_data/historical_4h')
    ap.add_argument('--max-files', type=int, default=400, help='Random sample of symbols; 0 = all')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-train-months', type=int, default=12)
    ap.add_argument('--test-months', type=int, default=3)
    ap.add_argument('--top-k', type=int, default=5, help='Names bought per bar in the top-k metric (live: 5)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--skip-legacy', action='store_true', help='Skip the legacy symbol-split comparison')
    args = ap.parse_args()

    params_path = args.params or latest_champion_params()
    with open(params_path) as f:
        champ = json.load(f)
    params, threshold = champ['params'], champ['threshold']
    logging.info(f"Params: {params_path}  threshold={threshold:.4f}")

    files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    if args.max_files and args.max_files < len(files):
        files = sorted(random.Random(args.seed).sample(files, args.max_files))
    logging.info(f"Building features for {len(files)} symbols ({args.workers} workers)...")

    cache_dir = None if args.no_cache else CACHE_DIR
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        built = [r for r in ex.map(build_symbol, [(f, params, cache_dir) for f in files], chunksize=4) if r]

    X = np.concatenate([b[1] for b in built])
    y = np.concatenate([b[2] for b in built])
    times = np.concatenate([b[3] for b in built])
    label_end = np.concatenate([b[4] for b in built])
    t_idx = pd.to_datetime(times, utc=True)
    end_idx = pd.to_datetime(label_end, utc=True)
    logging.info(f"{len(built)} symbols, {len(y):,} samples, base rate {y.mean():.1%}, "
                 f"{t_idx.min().date()} → {t_idx.max().date()}")

    searcher = BinaryHyperparameterSearch.__new__(BinaryHyperparameterSearch)
    rows = []

    if not args.skip_legacy:
        # Legacy split: symbol-ordered concatenation, last 20% of rows as test
        cut = int(len(y) * 0.8)
        logging.info("Fold legacy (symbol-ordered 80/20, same dates in train and test)...")
        r = evaluate(searcher, params, threshold, X[:cut], y[:cut], X[cut:], y[cut:], times[cut:], args.top_k)
        rows.append({'fold': 'legacy_symbol_split', 'test_start': t_idx[cut:].min().date(),
                     'test_end': t_idx[cut:].max().date(), 'purged': 0, **r})
        logging.info(f"  ROC-AUC {r['roc_auc']:.3f}  PR-lift {r['pr_lift']:.2f}x")

    for i, (start, end) in enumerate(make_folds(t_idx.min(), t_idx.max(),
                                                args.min_train_months, args.test_months)):
        test = (t_idx >= start) & (t_idx < end)
        before = t_idx < start
        train = before & (end_idx < start)     # purge labels that peek into the test window
        if test.sum() < 1000 or y[test].min() == y[test].max():
            logging.info(f"Fold {i}: {start.date()} – {end.date()} skipped ({test.sum()} test rows)")
            continue
        logging.info(f"Fold {i}: test {start.date()} – {end.date()}, "
                     f"train {train.sum():,} (purged {(before & ~train).sum():,}), test {test.sum():,}")
        r = evaluate(searcher, params, threshold, X[train], y[train], X[test], y[test], times[test], args.top_k)
        rows.append({'fold': i, 'test_start': start.date(), 'test_end': end.date(),
                     'purged': int((before & ~train).sum()), **r})
        logging.info(f"  ROC-AUC {r['roc_auc']:.3f}  PR-lift {r['pr_lift']:.2f}x  "
                     f"top{args.top_k} {r[f'top{args.top_k}_precision']:.1%} vs base "
                     f"{r[f'top{args.top_k}_base_rate']:.1%}")

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"{RESULTS_DIR}/walk_forward_{ts}.csv"
    res.to_csv(out_csv, index=False)

    wf = res[res['fold'] != 'legacy_symbol_split']
    k = args.top_k
    summary = {
        'params_file': params_path,
        'params': params,
        'threshold': threshold,
        'n_symbols': len(built),
        'n_samples': int(len(y)),
        'date_range': [str(t_idx.min().date()), str(t_idx.max().date())],
        'n_folds': int(len(wf)),
        'roc_auc_mean': float(wf['roc_auc'].mean()),
        'roc_auc_min': float(wf['roc_auc'].min()),
        'roc_auc_max': float(wf['roc_auc'].max()),
        'pr_lift_mean': float(wf['pr_lift'].mean()),
        f'top{k}_lift_mean': float(wf[f'top{k}_lift'].mean()),
        f'top{k}_lift_positive_folds': int((wf[f'top{k}_lift'] > 0).sum()),
    }
    if not args.skip_legacy:
        summary['legacy_roc_auc'] = float(res.loc[res['fold'] == 'legacy_symbol_split', 'roc_auc'].iloc[0])
    with open(f"{RESULTS_DIR}/walk_forward_{ts}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    pd.set_option('display.width', 200)
    cols = ['fold', 'test_start', 'test_end', 'n_test', 'base_rate', 'roc_auc', 'pr_lift',
            'signals_at_thresh', 'precision_at_thresh', f'top{k}_precision', f'top{k}_base_rate']
    print("\n" + res[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWalk-forward ROC-AUC: mean {summary['roc_auc_mean']:.3f} "
          f"(min {summary['roc_auc_min']:.3f}, max {summary['roc_auc_max']:.3f}) over {len(wf)} folds")
    if 'legacy_roc_auc' in summary:
        print(f"Legacy symbol-split ROC-AUC: {summary['legacy_roc_auc']:.3f}")
    print(f"Top-{k}-per-bar lift: mean {summary[f'top{k}_lift_mean']*100:+.1f} pts, "
          f"positive in {summary[f'top{k}_lift_positive_folds']}/{len(wf)} folds")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

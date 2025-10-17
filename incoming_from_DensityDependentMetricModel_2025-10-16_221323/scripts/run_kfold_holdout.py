#!/usr/bin/env python3
"""
run_kfold_holdout.py

K-fold hold-out validation for the hierarchical cluster lensing model.

For each fold:
- Train on (N - N/K) clusters using the NUTS grid runner
- Predict held-out clusters with posterior predictive and record coverage/errors

Produces per-fold artifacts and an aggregated summary JSON.
"""
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Local imports
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_mass_scaled_hierarchical_inference import load_cluster_catalog


def _partition_folds(names: List[str], k: int) -> List[List[str]]:
    idx = np.arange(len(names))
    folds = np.array_split(idx, k)
    return [[names[i] for i in fold] for fold in folds]


def _run_training(
    runner: Path,
    catalog: Path,
    tiers: List[int],
    exclude: List[str],
    grids_dir: Path,
    chains: int,
    draws: int,
    tune: int,
    target_accept: float,
    outdir: Path
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    # Map CLI between runner variants
    runner_args = [
        sys.executable, str(runner),
        '--catalog', str(catalog),
        '--tiers', ','.join(str(t) for t in tiers),
        '--exclude', ','.join(exclude) if exclude else '',
        '--grids', str(grids_dir),
        '--chains', str(chains),
        '--draws', str(draws),
        '--tune', str(tune),
        '--out', str(outdir)
    ]
    # Remove empty value arguments
    runner_args = [arg for arg in runner_args if arg != '']
    subprocess.run(runner_args, check=True)
    return outdir / 'trace.netcdf'


def _run_holdout_eval(posterior_path: Path, holdouts: List[str], outdir: Path) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'run_holdout_validation.py'),
        '--posterior', str(posterior_path),
        '--holdout', ','.join(holdouts),
        '--out', str(outdir)
    ]
    # Run validation; do not fail on non-zero exit (we still want metrics JSON)
    subprocess.run(args, check=False)
    with open(outdir / 'holdout_validation.json', 'r') as f:
        return json.load(f)


def aggregate_metrics(per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_preds = []
    for fold in per_fold:
        all_preds.extend(fold.get('per_cluster', []))
    n = len(all_preds)
    inside_68 = sum(1 for r in all_preds if r.get('inside_68', False))
    # z-score using predictive width approx (p84-p16)/2
    z_scores = []
    for r in all_preds:
        s_pred = max((r['theta_E_pred_84'] - r['theta_E_pred_16']) / 2.0, 1e-6)
        z = (r['theta_E_pred_median'] - r['theta_E_obs']) / s_pred
        z_scores.append(abs(z))
    frac_gt2 = float(sum(1 for z in z_scores if z > 2.0)) / n if n else 0.0
    med_frac_err = float(np.median([r['frac_error'] for r in all_preds])) if n else 0.0
    return {
        'n_predictions': n,
        'coverage_68': float(inside_68) / n if n else 0.0,
        'frac_gt_2sigma': frac_gt2,
        'median_fractional_error': med_frac_err
    }


def main():
    ap = argparse.ArgumentParser(description='K-fold hold-out validation')
    ap.add_argument('--catalog', required=True, help='Cluster catalog CSV')
    ap.add_argument('--tiers', default='1,2', help='Comma-separated tiers')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--grids', required=True, help='Path to precomputed thetaE grids dir')
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--draws', type=int, default=4000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--target_accept', type=float, default=0.9)
    ap.add_argument('--runner', default=str(REPO_ROOT / 'scripts' / 'run_mass_scaled_nuts_grid.py'),
                    help='Path to training runner (NUTS grid)')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(',')]
    catalog_path = Path(args.catalog)
    grids_dir = Path(args.grids)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load catalog and names
    df = load_cluster_catalog(tiers, exclude=None, data_path=catalog_path, include=None)
    names = list(df['cluster_name'].values)
    folds = _partition_folds(names, args.folds)

    fold_summaries = []
    for i, holdout_names in enumerate(folds, start=1):
        fold_dir = outdir / f'fold_{i}'
        train_dir = fold_dir / 'train'
        eval_dir = fold_dir / 'eval'
        exclude = holdout_names
        # Train
        posterior_path = _run_training(
            runner=Path(args.runner),
            catalog=catalog_path,
            tiers=tiers,
            exclude=exclude,
            grids_dir=grids_dir,
            chains=args.chains,
            draws=args.draws,
            tune=args.tune,
            target_accept=args.target_accept,
            outdir=train_dir
        )
        # Evaluate
        metrics = _run_holdout_eval(posterior_path, holdout_names, eval_dir)
        fold_summaries.append(metrics)

    agg = aggregate_metrics(fold_summaries)
    summary = {
        'folds': args.folds,
        'per_fold': fold_summaries,
        'aggregate': agg
    }
    with open(outdir / 'kfold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary['aggregate'], indent=2))


if __name__ == '__main__':
    main()
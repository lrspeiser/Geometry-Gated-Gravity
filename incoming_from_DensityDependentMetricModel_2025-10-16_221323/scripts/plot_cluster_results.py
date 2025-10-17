#!/usr/bin/env python3
"""
plot_cluster_results.py

Generate plots for cluster hold-out validation and k-fold coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def load_holdout_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_holdouts(holdout_data: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = holdout_data.get('per_cluster', [])
    if not rows:
        return
    obs = np.array([r['theta_E_obs'] for r in rows], dtype=float)
    pred = np.array([r['theta_E_pred_median'] for r in rows], dtype=float)
    p16 = np.array([r['theta_E_pred_16'] for r in rows], dtype=float)
    p84 = np.array([r['theta_E_pred_84'] for r in rows], dtype=float)
    yerr = np.vstack([pred - p16, p84 - pred])

    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    ax.errorbar(obs, pred, yerr=yerr, fmt='o', color='C0', ecolor='C0', capsize=3)
    lim = [0.8*obs.min(), 1.2*obs.max()]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'Observed $\theta_E$ (arcsec)')
    ax.set_ylabel(r'Predicted $\theta_E$ (arcsec)')
    ax.set_title('Hold-out predicted vs observed')
    for r in rows:
        ax.annotate(r['cluster'], (r['theta_E_obs'], r['theta_E_pred_median']), xytext=(3,3), textcoords='offset points', fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'holdouts_pred_vs_obs.png')
    plt.close(fig)


def scan_kfold(kfold_dir: Path):
    eval_files = list(kfold_dir.glob('fold_*/eval/holdout_validation.json'))
    all_rows = []
    for f in eval_files:
        try:
            with open(f, 'r') as jf:
                d = json.load(jf)
                all_rows.extend(d.get('per_cluster', []))
        except Exception:
            pass
    return all_rows


def plot_kfold(rows: list[dict], outdir: Path):
    if not rows:
        return
    obs = np.array([r['theta_E_obs'] for r in rows], dtype=float)
    pred = np.array([r['theta_E_pred_median'] for r in rows], dtype=float)
    p16 = np.array([r['theta_E_pred_16'] for r in rows], dtype=float)
    p84 = np.array([r['theta_E_pred_84'] for r in rows], dtype=float)
    yerr = np.vstack([pred - p16, p84 - pred])

    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    ax.errorbar(obs, pred, yerr=yerr, fmt='o', color='C1', ecolor='C1', alpha=0.8, capsize=2)
    lim = [0.8*obs.min(), 1.2*obs.max()]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'Observed $\theta_E$ (arcsec)')
    ax.set_ylabel(r'Predicted $\theta_E$ (arcsec)')
    ax.set_title('K-fold hold-out: predicted vs observed')
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'kfold_pred_vs_obs.png')
    plt.close(fig)

    # coverage histogram
    inside = np.array([(p16[i] <= obs[i] <= p84[i]) for i in range(len(obs))], dtype=bool)
    fig, ax = plt.subplots(figsize=(4,3), dpi=150)
    ax.bar([0,1], [inside.size - inside.sum(), inside.sum()], color=['#ccc','#4caf50'])
    ax.set_xticks([0,1]); ax.set_xticklabels(['Outside 68%','Inside 68%'])
    ax.set_ylabel('Count')
    ax.set_title(f'Coverage: {inside.sum()}/{inside.size} = {inside.sum()/max(1,inside.size):.2%}')
    fig.tight_layout()
    fig.savefig(outdir / 'kfold_coverage.png')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Plot cluster hold-out and k-fold results')
    ap.add_argument('--holdout-json', required=False)
    ap.add_argument('--kfold-dir', required=False)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)

    if args.holdout_json:
        d = load_holdout_json(Path(args.holdout_json))
        plot_holdouts(d, outdir)

    if args.kfold_dir:
        rows = scan_kfold(Path(args.kfold_dir))
        plot_kfold(rows, outdir)

if __name__ == '__main__':
    main()

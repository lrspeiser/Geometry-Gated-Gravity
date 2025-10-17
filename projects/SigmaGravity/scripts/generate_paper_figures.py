#!/usr/bin/env python3
"""
Generate paper figures from calibration/validation outputs.
- Holdouts predicted vs observed (scatter with 68% CI)
- K-fold coverage (if kfold results present)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SG = ROOT / 'projects' / 'SigmaGravity'

HOLDOUT_DIRS = [ROOT / 'output' / 'holdout_validation_mass_scaled', SG / 'output' / 'holdout_paper']
FIGS = SG / 'figures'


def load_holdout_results():
    for d in HOLDOUT_DIRS:
        p = d / 'holdout_results.json'
        if p.exists():
            return json.loads(p.read_text())
    return None


def plot_holdouts(results):
    names = [k for k in results.keys() if k != 'summary']
    obs = [results[k]['theta_E_obs'] for k in names]
    obs_err = [results[k]['theta_E_err'] for k in names]
    pred = [results[k]['theta_E_pred_med'] for k in names]
    pred_lo = [results[k]['theta_E_pred_16'] for k in names]
    pred_hi = [results[k]['theta_E_pred_84'] for k in names]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(obs, pred, xerr=obs_err, yerr=[np.array(pred)-np.array(pred_lo), np.array(pred_hi)-np.array(pred)], fmt='o', color='C0')
    lim = [0, max(max(obs), max(pred)) * 1.1]
    ax.plot(lim, lim, 'k--', alpha=0.5)
    ax.set_xlabel('Observed θ_E (arcsec)')
    ax.set_ylabel('Predicted θ_E (arcsec)')
    ax.set_title('Holdouts: predicted vs observed (68% CI)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / 'holdouts_pred_vs_obs.png'
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f'Saved {out}')


def main():
    res = load_holdout_results()
    if res:
        plot_holdouts(res)
    else:
        print('WARN: No holdout_results.json found')


if __name__ == '__main__':
    main()

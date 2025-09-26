"""
A+B grid search runner for G³ cluster tests
Sweeps geometry-aware amplitude (A) and cluster-aware screening softening (B)
Outputs CSV + JSON summaries and a small dashboard figure.
"""

import json
import csv
import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from comprehensive_analysis import ClusterTestFramework


def run_grid_search(output_dir: str = 'g3_cluster_tests/outputs',
                    gamma_As=(0.5, 1.0, 1.5),
                    beta_As=(0.3, 0.5, 0.7),
                    f_maxs=(2.0, 3.0, 4.0),
                    eta_sigmas=(0.5, 1.0),
                    eta_alphas=(0.0, 0.5),
                    use_curv_gate=False,
                    p_in=2.0, p_out=1.0):
    os.makedirs(output_dir, exist_ok=True)

    # Baseline (no A/B)
    base = ClusterTestFramework()
    base.create_cluster_model()
    res_base = base.run_comprehensive_test()['Standard G³']
    base_kappa_max = float(res_base['kappa_max'])

    results = []
    best = {'kappa_max': -1, 'config': None, 'record': None}

    total = len(gamma_As) * len(beta_As) * len(f_maxs) * len(eta_sigmas) * len(eta_alphas)
    idx = 0

    for gamma_A, beta_A, f_max, eta_sigma, eta_alpha in product(gamma_As, beta_As, f_maxs, eta_sigmas, eta_alphas):
        idx += 1
        # Build framework with these knobs
        fw = ClusterTestFramework(
            gamma_A=gamma_A, beta_A=beta_A, f_max=f_max,
            eta_sigma=eta_sigma, eta_alpha=eta_alpha,
            use_curv_gate=use_curv_gate, p_in=p_in, p_out=p_out
        )
        fw.create_cluster_model()
        res = fw.run_comprehensive_test()['Standard G³']  # uses these internal knobs when building tail

        record = {
            'gamma_A': gamma_A,
            'beta_A': beta_A,
            'f_max': f_max,
            'eta_sigma': eta_sigma,
            'eta_alpha': eta_alpha,
            'use_curv_gate': use_curv_gate,
            'p_in': p_in,
            'p_out': p_out,
            'kappa_max': float(res['kappa_max']),
            'kappa_50': float(res['kappa_50']),
            'kappa_100': float(res['kappa_100']),
            'improvement_vs_baseline': float(res['kappa_max'])/base_kappa_max if base_kappa_max>0 else np.nan
        }
        results.append(record)

        if record['kappa_max'] > best['kappa_max']:
            best['kappa_max'] = record['kappa_max']
            best['config'] = (gamma_A, beta_A, f_max, eta_sigma, eta_alpha)
            best['record'] = record

        if idx % 10 == 0:
            print(f"[{idx}/{total}] κ_max={record['kappa_max']:.3f} best={best['kappa_max']:.3f}")

    # Write CSV
    csv_path = os.path.join(output_dir, 'grid_search_ab_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved grid results CSV to {csv_path}")

    # Write JSON summary
    summary = {
        'baseline_kappa_max': base_kappa_max,
        'best': best['record'],
        'count': len(results)
    }
    json_path = os.path.join(output_dir, 'grid_search_ab_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved grid summary JSON to {json_path}")

    # Small dashboard figure for top-N
    topN = sorted(results, key=lambda r: r['kappa_max'], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(topN)), [r['kappa_max'] for r in topN], color='teal')
    ax.set_xticks(range(len(topN)))
    ax.set_xticklabels([f"γA={r['gamma_A']}, βA={r['beta_A']}, f={r['f_max']}\nησ={r['eta_sigma']}, ηα={r['eta_alpha']}" for r in topN], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('κ̄_max')
    ax.set_title('Top-10 A+B configs (κ̄_max)')
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'grid_search_ab_top10.png')
    fig.savefig(fig_path, dpi=150)
    print(f"Saved grid top-10 plot to {fig_path}")

    return results, summary


if __name__ == '__main__':
    # Default: A+B sweep only, curvature gate off
    run_grid_search()

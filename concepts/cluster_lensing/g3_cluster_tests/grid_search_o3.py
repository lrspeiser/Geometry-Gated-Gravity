"""
Grid search for O3 (third-order lensing-only) parameters.
Sweeps environment kernel scale and test-mass scaling knobs with basic guardrails.
Outputs CSV, JSON, and a top-10 plot like other runners.
"""
import os
import csv
import json
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from comprehensive_analysis import ClusterTestFramework
from o3_lensing import apply_o3_lensing


def run_grid_search_o3(output_dir: str = 'g3_cluster_tests/outputs',
                       ell3_list=(300.0, 400.0, 600.0),
                       Sigma_star3_list=(30.0, 50.0, 80.0),
                       beta3_list=(1.0, 1.5, 1.8),
                       r3_list=(60.0, 80.0, 120.0),
                       w3_list=(0.7, 1.0, 1.5),
                       xi3_list=(0.5, 0.8, 1.2),
                       A3_list=(1e-6, 1e-5, 1e-4, 5e-4),
                       chi_list=(0.5, 0.8, 1.0)):
    os.makedirs(output_dir, exist_ok=True)

    fw = ClusterTestFramework()
    fw.create_cluster_model()
    g_N = fw.compute_newtonian()
    # Baseline dynamics include the G3 tail to keep consistency with your pipeline
    g_dyn = g_N + fw.compute_g3_tail(eta_boost=0.0)

    # Project baryon Sigma for O3 environment (Abel of rho_nfw)
    try:
        Sigma_bary_kpc2 = fw._sigma_from_rho_abel(fw.r, fw.rho_nfw, fw.r)
        Sigma_bary_pc2 = Sigma_bary_kpc2 / 1e6
    except Exception:
        Sigma_bary_pc2 = fw.rho_nfw * fw.r * 1e-6

    # Helper to compute metrics
    def compute_metrics(params):
        g_lens = apply_o3_lensing(g_dyn, fw.r, Sigma_bary_pc2, params, m_test_Msun=0.0)
        kappa, kbar, _ = fw.compute_convergence(g_lens)
        return {
            'kappa_max': float(np.max(kbar)),
            'kappa_30': float(np.interp(30, fw.r, kbar)),
            'kappa_50': float(np.interp(50, fw.r, kbar)),
            'kappa_100': float(np.interp(100, fw.r, kbar)),
        }

    results = []
    combos = list(product(ell3_list, Sigma_star3_list, beta3_list, r3_list, w3_list, xi3_list, A3_list, chi_list))
    for (ell3, Sstar3, beta3, r3, w3, xi3, A3, chi) in combos:
        params = {
            'ell3_kpc': float(ell3),
            'Sigma_star3_Msun_pc2': float(Sstar3),
            'beta3': float(beta3),
            'r3_kpc': float(r3),
            'w3_decades': float(w3),
            'xi3': float(xi3),
            'A3': float(A3),
            'chi': float(chi),
            'm_ref_Msun': 1.0,
'm_floor_Msun': 1e-6,
        }
        met = compute_metrics(params)
        rec = {
            'ell3_kpc': float(ell3), 'Sigma_star3': float(Sstar3), 'beta3': float(beta3),
            'r3_kpc': float(r3), 'w3_decades': float(w3), 'xi3': float(xi3),
            'A3': float(A3), 'chi': float(chi),
            **met,
        }
        results.append(rec)

    # Save CSV
    csv_path = os.path.join(output_dir, 'grid_search_o3_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Top-10 plot
    topN = sorted(results, key=lambda r: r['kappa_max'], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(topN)), [r['kappa_max'] for r in topN], color='darkorange')
    ax.set_xticks(range(len(topN)))
    ax.set_xticklabels([
        f"ell={r['ell3_kpc']}, S*={r['Sigma_star3']}, β3={r['beta3']}\nr3={r['r3_kpc']}, w3={r['w3_decades']}, ξ3={r['xi3']}, A3={r['A3']}, χ={r['chi']}"
        for r in topN
    ], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('κ̄_max')
    ax.set_title('Top-10 O3 configs (κ̄_max)')
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'grid_search_o3_top10.png')
    fig.savefig(fig_path, dpi=150)

    # JSON summary
    summary = {'best': topN[0], 'count': len(results)}
    with open(os.path.join(output_dir, 'grid_search_o3_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved O3 grid results to {csv_path} and plot to {fig_path}")
    return results, summary


if __name__ == '__main__':
    run_grid_search_o3()

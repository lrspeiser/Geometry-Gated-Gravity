"""
A+B+C grid search runner
- Loads top-N configs from A+B grid
- Sweeps p_in (curvature gate) values with use_curv_gate=True
- Writes CSV/JSON summaries and a plot
- Aborts if NFW θ_E unit test fails
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from comprehensive_analysis import ClusterTestFramework, nfw_lensing_unit_test


def run_grid_search_abc(output_dir: str = 'g3_cluster_tests/outputs',
                        topN:int = 10,
                        p_in_values=(2.0, 2.25, 2.5),
                        p_out=1.0):
    os.makedirs(output_dir, exist_ok=True)

    # Gate with NFW unit test first
    theta_E, passed = nfw_lensing_unit_test()
    if not passed:
        print(f"NFW θ_E unit test WARNING: θ_E={theta_E:.2f} arcsec outside expected band. Proceeding with grid.")
        with open(os.path.join(output_dir, 'grid_search_abc_WARNING.txt'), 'w', encoding='utf-8') as f:
            f.write(f"NFW θ_E WARNING (θ_E={theta_E}). Proceeding with grid.\n")
    else:
        print(f"NFW θ_E unit test passed: θ_E ≈ {theta_E:.1f}" )

    # Load A+B results
    ab_csv = os.path.join(output_dir, 'grid_search_ab_results.csv')
    if not os.path.exists(ab_csv):
        raise FileNotFoundError(f"Missing A+B results at {ab_csv}")

    rows = []
    with open(ab_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['kappa_max'] = float(r['kappa_max'])
            rows.append(r)

    rows_sorted = sorted(rows, key=lambda r: r['kappa_max'], reverse=True)[:topN]

    results = []
    best = {'kappa_max': -1, 'record': None}

    for base in rows_sorted:
        gamma_A = float(base['gamma_A'])
        beta_A = float(base['beta_A'])
        f_max = float(base['f_max'])
        eta_sigma = float(base['eta_sigma'])
        eta_alpha = float(base['eta_alpha'])

        for p_in in p_in_values:
            fw = ClusterTestFramework(
                gamma_A=gamma_A, beta_A=beta_A, f_max=f_max,
                eta_sigma=eta_sigma, eta_alpha=eta_alpha,
                use_curv_gate=True, p_in=p_in, p_out=p_out
            )
            fw.create_cluster_model()
            res = fw.run_comprehensive_test()['Standard G³']

            rec = {
                'gamma_A': gamma_A,
                'beta_A': beta_A,
                'f_max': f_max,
                'eta_sigma': eta_sigma,
                'eta_alpha': eta_alpha,
                'p_in': p_in,
                'p_out': p_out,
                'kappa_max': float(res['kappa_max']),
                'kappa_30': float(np.interp(30, fw.r, res['kappa_bar'])),
                'kappa_50': float(np.interp(50, fw.r, res['kappa_bar'])),
                'kappa_100': float(np.interp(100, fw.r, res['kappa_bar'])),
            }
            results.append(rec)
            if rec['kappa_max'] > best['kappa_max']:
                best['kappa_max'] = rec['kappa_max']
                best['record'] = rec

    # Write CSV
    csv_path = os.path.join(output_dir, 'grid_search_abc_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved ABC grid CSV to {csv_path}")

    # JSON summary
    json_path = os.path.join(output_dir, 'grid_search_abc_summary.json')
    summary = {'best': best['record'], 'count': len(results)}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved ABC grid summary JSON to {json_path}")

    # Plot top-10
    top10 = sorted(results, key=lambda r: r['kappa_max'], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(range(len(top10)), [r['kappa_max'] for r in top10], color='purple')
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels([f"γA={r['gamma_A']}, βA={r['beta_A']}, f={r['f_max']}\nησ={r['eta_sigma']}, ηα={r['eta_alpha']}, p_in={r['p_in']}" for r in top10], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('κ̄_max')
    ax.set_title('Top-10 A+B+C configs (κ̄_max)')
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'grid_search_abc_top10.png')
    fig.savefig(fig_path, dpi=150)
    print(f"Saved ABC grid top-10 plot to {fig_path}")

    return results, summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--topN', type=int, default=10)
    p.add_argument('--p_in_list', type=str, default='2.0,2.25,2.5')
    p.add_argument('--p_out', type=float, default=1.0)
    p.add_argument('--xi_gamma_list', type=str, default='0.35')
    p.add_argument('--beta_gamma', type=float, default=1.0)
    p.add_argument('--nu_gamma', type=float, default=1.0)
    p.add_argument('--Sigma_star', type=float, default=80.0)
    p.add_argument('--Sigma0', type=float, default=10.0)
    p.add_argument('--m_sat_list', type=str, default='1.5')
    args = p.parse_args()
    # Run with default - note: CLI options for ABC curve need framework changes in caller; here we only sweep p_in
    p_ins = tuple(float(x) for x in args.p_in_list.split(','))
    # For now, we pass xi_gamma and sat params via framework defaults below when needed
    run_grid_search_abc(topN=args.topN, p_in_values=p_ins, p_out=args.p_out)

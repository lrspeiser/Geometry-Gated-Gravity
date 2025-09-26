"""
Grid search for PMG (photon mass-gated) lensing parameters on toy clusters.
Writes top results to JSON.
"""
import json, os
import numpy as np
from itertools import product
from comprehensive_analysis import ClusterTestFramework

def run_once(pmg):
    fw = ClusterTestFramework(use_pmg=True, pmg_params=pmg)
    fw.create_cluster_model()
    res = fw.run_comprehensive_test()
    # Use PMG result if available
    key = 'PMG - Photons' if 'PMG - Photons' in res else 'Standard G³'
    r = fw.r
    kbar = res[key]['kappa_bar']
    return {
        'pmg': pmg,
        'kbar_max': float(np.max(kbar)),
        'kbar_50': float(np.interp(50.0, r, kbar)),
        'kbar_100': float(np.interp(100.0, r, kbar)),
    }

def main():
    out_dir = 'g3_cluster_tests/outputs'
    os.makedirs(out_dir, exist_ok=True)
    # Conservative ranges to avoid blow-ups; aim κ̄ ~ 0.8–1.2 at 30–100 kpc
    grid = {
        'chi': [1.0, 1.5],
        'mfloor_Msun': [1e-8, 1e-6],
        'A0': [5e-4, 1e-3, 5e-3],
        'Sigma_star': [30.0, 50.0],
        'Sigma0': [10.0, 20.0],
        'beta': [0.5, 0.7],
        'Rboost_kpc': [400.0, 600.0],
        'q': [2.0],
        'eta': [1.0],
        'c_curv': [0.0, 0.3],
        'nu': [1.0],
        'mref_Msun': [1.0],
    }
    keys = list(grid.keys())
    results = []
    count = 0
    for vals in product(*[grid[k] for k in keys]):
        pmg = {k:v for k,v in zip(keys, vals)}
        r = run_once(pmg)
        # basic guardrails
        # Solar gate proxy: E(r=1 kpc, Sigma=1000) should be tiny
        E_solar = 1.0 / (1.0 + (1000.0 / pmg['Sigma0'])**pmg['beta']) * ((1.0/pmg['Rboost_kpc'])**pmg['q'] / (1.0 + (1.0/pmg['Rboost_kpc'])**pmg['q']))
        solar_ok = (E_solar < 1e-8)
        r['solar_ok'] = bool(solar_ok)
        # Dynamics proxy: star mass ~1 Msun should not be amplified much
        # Effective mass factor for tracer ~1 Msun
        mref = pmg['mref_Msun']; mfloor = pmg['mfloor_Msun']; chi = pmg['chi']
        mass_fac_star = (mref/(1.0 + mfloor))**chi
        r['dyn_ok'] = bool(abs(mass_fac_star - 1.0) < 1e-3)
        results.append(r)
        count += 1
        if count % 25 == 0:
            print(f"{count} configs, best so far: {max(results, key=lambda x: x['kbar_max'])['kbar_max']:.3f}")
    results.sort(key=lambda x: x['kbar_max'], reverse=True)
    with open(os.path.join(out_dir, 'pmg_grid_results.json'), 'w') as f:
        json.dump(results[:50], f, indent=2)
    print(f"Saved PMG grid results (top 50) to {out_dir}/pmg_grid_results.json")

if __name__ == '__main__':
    main()

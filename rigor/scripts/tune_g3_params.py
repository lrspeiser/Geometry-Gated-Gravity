#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import subprocess as sp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_CLUSTER = ROOT / 'root-m' / 'pde' / 'run_cluster_pde.py'
RUN_SPARC   = ROOT / 'root-m' / 'pde' / 'run_sparc_pde.py'
DATA        = ROOT / 'data'
OUT_BASE    = ROOT / 'root-m' / 'out'

# Default grids (broader for clusters under full-baryon gN)
S0_GRID = [5.0e-5, 6.0e-5, 7.0e-5, 8.0e-5, 1.0e-4, 1.2e-4]
RC_GRID = [18.0, 22.0, 28.0, 35.0, 45.0]
RC_GAMMA_GRID = [0.3, 0.4, 0.5, 0.6]
SIGMA_BETA_GRID = [0.05, 0.08, 0.10, 0.12]

CLUSTERS = {
    'ABELL_0426': {
        'Rmax': 600, 'Zmax': 600,
        'clump_profile_csv': DATA / 'clusters' / 'ABELL_0426' / 'clump_profile.csv',
        'stars_csv':         DATA / 'clusters' / 'ABELL_0426' / 'stars_profile.csv',
    },
    'ABELL_1689': {
        'Rmax': 900, 'Zmax': 900,
        'clump_profile_csv': DATA / 'clusters' / 'ABELL_1689' / 'clump_profile.csv',
        'stars_csv':         DATA / 'clusters' / 'ABELL_1689' / 'stars_profile.csv',
    },
}

# Fixed globals for this tuning
G0 = 1200
RC_REF   = 30
SIGMA0     = 150
# Use slightly coarser grid for tuning speed (clusters)
NR = 96
NZ = 96
# Evaluation resolution for SPARC overlays
E_NR = 128
E_NZ = 128

# SPARC config
SINGLE_GAL_OVERLAYS = True
SPARC_IN   = DATA / 'sparc_predictions_by_radius.csv'
ROTMOD_PAR = DATA / 'sparc_rotmod_ltg.parquet'
ALL_TBL    = DATA / 'sparc_all_tables.parquet'
GAL_LIST_JSON = ROOT / 'out' / 'analysis' / 'type_breakdown' / 'sparc_example_galaxies.json'


def run(cmd: list[str]) -> None:
    print('[RUN]', ' '.join(str(x) for x in cmd), flush=True)
    sp.run([str(x) for x in cmd], check=True)


def tune_clusters() -> dict:
    results = []
    for S0 in S0_GRID:
        for rc in RC_GRID:
            for rc_gamma in RC_GAMMA_GRID:
                for sigma_beta in SIGMA_BETA_GRID:
                    errs = []
                    tag = f'tune_S0_{S0:.3e}_rc_{rc:.1f}_g_{rc_gamma:.2f}_b_{sigma_beta:.2f}'
                    outdir = OUT_BASE / 'pde_clusters_tune' / tag
                    for cl, cfg in CLUSTERS.items():
                        od = outdir / cl
                        od.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            'py', '-u', str(RUN_CLUSTER),
                            '--cluster', cl,
                            '--S0', f'{S0}', '--rc_kpc', f'{rc}',
                            '--rc_gamma', f'{rc_gamma}', '--rc_ref_kpc', f'{RC_REF}',
                            '--sigma_beta', f'{sigma_beta}', '--sigma0_Msun_pc2', f'{SIGMA0}',
                            '--g0_kms2_per_kpc', f'{G0}', '--NR', f'{NR}', '--NZ', f'{NZ}',
                            '--Rmax', f"{cfg['Rmax']}", '--Zmax', f"{cfg['Zmax']}",
                            '--clump_profile_csv', str(cfg['clump_profile_csv']),
                            '--stars_csv', str(cfg['stars_csv']),
                            '--gN_from_total_baryons',
                            '--outdir', str(outdir)
                        ]
                        run(cmd)
                        # read metrics
                        mpath = od / 'metrics.json'
                        if not mpath.exists():
                            raise RuntimeError(f'Missing metrics: {mpath}')
                        with open(mpath, 'r') as f:
                            m = json.load(f)
                        errs.append(float(m.get('temp_median_frac_err', 1e9)))
                    mean_err = sum(errs) / len(errs)
                    results.append({'S0': S0, 'rc_kpc': rc, 'rc_gamma': rc_gamma, 'sigma_beta': sigma_beta,
                                    'mean_cluster_frac_err': mean_err, 'per_cluster': dict(zip(CLUSTERS.keys(), errs))})
                    print(f"[TUNE] S0={S0:.3e} rc={rc:.1f} rc_gamma={rc_gamma:.2f} sigma_beta={sigma_beta:.2f} -> mean |dT|/T = {mean_err:.3f}")
    # pick best
    best = min(results, key=lambda r: r['mean_cluster_frac_err'])
    out_json = OUT_BASE / 'pde_clusters_tune' / 'tune_summary.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'grid': results, 'best': best}, f, indent=2)
    print('[TUNE] Best:', best)
    return best


def eval_sparc(best: dict) -> dict:
    S0 = best['S0']; rc = best['rc_kpc']
    rc_gamma = best.get('rc_gamma', 0.5)
    sigma_beta = best.get('sigma_beta', 0.1)
    # Try to get ~50 galaxies: prefer an eval JSON, else sample from rotmod parquet
    eval_json = ROOT / 'out' / 'analysis' / 'type_breakdown' / 'sparc_eval_galaxies.json'
    gals: list[str] = []
    if eval_json.exists():
        try:
            sel = json.loads(eval_json.read_text())
            gals = [s for s in sel.get('galaxies', [])][:50]
        except Exception:
            gals = []
    if not gals:
        try:
            import pandas as _pd
            rot = _pd.read_parquet(ROTMOD_PAR)
            gals = sorted(list(dict.fromkeys(rot['galaxy'].dropna().astype(str).tolist())))[:50]
        except Exception:
            if GAL_LIST_JSON.exists():
                sel = json.loads(GAL_LIST_JSON.read_text())
                gals = sel.get('galaxies', [])
                gals = gals[:50]
    if not gals:
        print('[WARN] No galaxies found for SPARC eval')
        return {}
    outdir = OUT_BASE / 'pde_sparc_eval'
    outdir.mkdir(parents=True, exist_ok=True)
    medians = []
    for g in gals:
        cmd = [
            'py', '-u', str(RUN_SPARC),
            '--in', str(SPARC_IN), '--axisym_maps', '--galaxy', g,
            '--rotmod_parquet', str(ROTMOD_PAR), '--all_tables_parquet', str(ALL_TBL),
            '--hz_kpc', '0.3', '--NR', f'{E_NR}', '--NZ', f'{E_NZ}',
            '--S0', f'{S0}', '--rc_kpc', f'{rc}',
            '--rc_gamma', f'{rc_gamma}', '--rc_ref_kpc', f'{RC_REF}',
            '--sigma_beta', f'{sigma_beta}', '--sigma0_Msun_pc2', f'{SIGMA0}',
            '--g0_kms2_per_kpc', f'{G0}', '--outdir', str(outdir), '--tag', g,
        ]
        run(cmd)
        # read summary
        s_path = outdir / g / 'summary.json'
        if s_path.exists():
            with open(s_path, 'r') as f:
                s = json.load(f)
            med = float(s.get('median_percent_close', 0.0))
            medians.append(med)
    overall = sum(medians)/len(medians) if medians else 0.0
    with open(outdir / 'sparc_eval_summary.json', 'w') as f:
        json.dump({'tuple': {'S0': S0, 'rc_kpc': rc, 'rc_gamma': rc_gamma, 'sigma_beta': sigma_beta, 'g0': G0},
                   'per_gal_median_percent_close': dict(zip(gals, medians)),
                   'mean_median_percent_close': overall}, f, indent=2)
    print(f'[SPARC] Mean median-percent-closenness across {len(medians)} galaxies = {overall:.2f}%')
    return {'mean_median_percent_close': overall}


def main():
    best = tune_clusters()
    eval_sparc(best)

if __name__ == '__main__':
    main()
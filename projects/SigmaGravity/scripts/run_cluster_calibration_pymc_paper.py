#!/usr/bin/env python3
"""
Calibrate mass-scaled hierarchical model using PyMC (paper variant).
- Reads projects/SigmaGravity/config/paper_settings.yaml and cluster_selection_paper.yaml
- Saves ArviZ trace (trace.nc), posterior_summary.csv, and a manifest.json (for provenance).
- Optionally converts to NPZ for compatibility with validate_holdout_mass_scaled.py
"""
import sys, os, json, time, hashlib, argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import arviz as az
except ImportError:
    az = None

try:
    import yaml
except ImportError:
    yaml = None

# Repo roots
ROOT = Path(__file__).resolve().parents[3]
SG = ROOT / 'projects' / 'SigmaGravity'
CFG_MAIN = SG / 'config' / 'paper_settings.yaml'
CFG_SEL = SG / 'config' / 'cluster_selection_paper.yaml'

# Scripts and modules
sys.path.insert(0, str(ROOT))

from core.hierarchical_cluster_model_mass_scaled import (
    HierarchicalClusterModelMassScaled,
    GeometryPredictors,
)


def git_commit_hash_or_none() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(['git', '--no-pager', 'rev-parse', 'HEAD'], cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    if yaml is None:
        print('FATAL: PyYAML not installed'); sys.exit(2)
    with open(CFG_MAIN, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    with open(CFG_SEL, 'r', encoding='utf-8') as f:
        sel = yaml.safe_load(f)
    return cfg, sel


def select_clusters(catalog: pd.DataFrame, sel: dict):
    variant = sel.get('variant', 'current_catalog')
    selv = sel[variant]
    holdouts = set(selv.get('holdout_names', []))
    excludes = set(selv.get('exclude_names', []))
    trains = selv.get('train_names', [])

    if not trains:
        # Use tiers 1 and 2
        mask_tiers = catalog['tier'].astype(int).isin([1,2])
        df = catalog[mask_tiers].copy()
        df_train = df[~df['cluster_name'].isin(holdouts | excludes)].copy()
        df_hold = df[df['cluster_name'].isin(holdouts)].copy()
    else:
        df_train = catalog[catalog['cluster_name'].isin(trains)].copy()
        df_hold = catalog[catalog['cluster_name'].isin(list(holdouts))].copy()
    return df_train.reset_index(drop=True), df_hold.reset_index(drop=True)


def build_predictors(df: pd.DataFrame):
    predictors = []
    for _, row in df.iterrows():
        predictors.append(GeometryPredictors(
            R_500=row['R_500_kpc'],
            M_500=row['M_500_Msun'],
            z=row['z_lens'],
            cool_core=(row['dynamical_state'] == 'relaxed'),
            c_500=3.0,
            T_X=row.get('TX_central_keV', 8.0)
        ))
    return predictors


def main():
    parser = argparse.ArgumentParser(description='Paper calibration (PyMC)')
    parser.add_argument('--outdir', type=str, default=str(SG / 'output' / 'pymc_mass_scaled'))
    parser.add_argument('--convert-npz', type=int, default=1)
    args = parser.parse_args()

    cfg, sel = load_config()

    catalog_path = (ROOT / cfg['clusters']['catalog_path']).resolve()
    catalog = pd.read_csv(catalog_path)

    df_train, df_hold = select_clusters(catalog, sel)
    print(f'Train: {len(df_train)} → {", ".join(df_train["cluster_name"].values)}')
    print(f'Holdout: {len(df_hold)} → {", ".join(df_hold["cluster_name"].values)}')

    predictors = build_predictors(df_train)
    observations = {
        'theta_E': df_train['theta_E_obs_arcsec'].values,
        'theta_E_err': df_train['theta_E_err_arcsec'].values,
    }

    model = HierarchicalClusterModelMassScaled(
        predictors=predictors,
        observations=observations,
        use_pymc=True,
        include_secondary_effects=False,
    )

    # Sampling controls
    n_samples = 2000
    n_tune = 1000
    n_chains = 4

    trace = model.fit_mcmc(n_samples=n_samples, n_tune=n_tune, n_chains=n_chains, target_accept=0.95)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = outdir / 'trace.nc'
    print(f'Saving ArviZ trace → {trace_path}')
    trace.to_netcdf(trace_path)

    # Summary
    if az is not None:
        summary = az.summary(trace, var_names=['ell0_star_pop','gamma_pop','mu_A_pop','sigma_A_pop'])
        summary.to_csv(outdir / 'posterior_summary.csv')

    # Manifest
    manifest = {
        'run_id': time.strftime('%Y%m%d_%H%M%S'),
        'train_clusters': sorted(df_train['cluster_name'].tolist()),
        'tiers': [1,2],
        'cosmology': {'H0': cfg['cosmology']['H0'], 'Om0': cfg['cosmology']['Om0']},
        'physics': {'bcg': True, 'triaxial': True, 'pzsource': cfg['source_redshift']['mode']},
        'kernel': {'norm': cfg['kernel']['norm'], 'mass_scaling': True, 'gamma_prior': cfg['kernel']['gamma_prior']},
        'catalog_md5': md5sum(catalog_path),
        'catalog_path': str(catalog_path),
        'code_commit': git_commit_hash_or_none(),
    }
    with open(outdir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Optional: convert to NPZ for validator compatibility
    if args.convert_npz:
        try:
            ell0 = np.ravel(trace.posterior['ell0_star_pop'].values)
            gamma = np.ravel(trace.posterior['gamma_pop'].values)
            muA = np.ravel(trace.posterior['mu_A_pop'].values)
            sigA = np.ravel(trace.posterior['sigma_A_pop'].values)
            n = min(len(ell0), len(gamma), len(muA), len(sigA))
            samples = np.vstack([ell0[:n], gamma[:n], muA[:n], sigA[:n]]).T
            np.savez(outdir / 'flat_samples_from_pymc.npz', samples=samples, manifest=json.dumps(manifest))
            print(f'Saved NPZ for validator → {outdir / "flat_samples_from_pymc.npz"}')
        except Exception as e:
            print(f'WARN: Could not convert to NPZ: {e}')

    print('Calibration complete.')


if __name__ == '__main__':
    main()

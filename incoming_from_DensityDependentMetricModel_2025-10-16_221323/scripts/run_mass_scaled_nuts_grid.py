#!/usr/bin/env python3
"""
run_mass_scaled_nuts_grid.py

PyMC model using precomputed theta_E(A_c, ell0) grids with bilinear interpolation.
Attempts NUTS; falls back to DEMetropolisZ if gradients unavailable.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import extra_ops as xo

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_mass_scaled_hierarchical_inference import load_cluster_catalog
from scripts.cluster_overrides import normalize_cluster_name


def load_grids(index_path: Path, grids_dir: Path, clusters: list[str]):
    with open(index_path, 'r') as f:
        idx_rows = json.load(f)
        # support both old and new index formats
        idx = {row['cluster_name']: row.get('file') for row in idx_rows}
    A_list, L0_list, T_list, R500 = [], [], [], []
    for name in clusters:
        key = normalize_cluster_name(name)
        file = grids_dir / f'{key}.npz'
        if not file.exists():
            raise FileNotFoundError(f'Missing grid for {name}: {file}')
        dat = np.load(file, allow_pickle=True)
        A_list.append(dat['A_grid'])
        L0_list.append(dat['L0_grid'])
        # If a newer 5D grid exists, take the central slice in geometry/kappa to remain compatible
        theta = dat['thetaE']
        if theta.ndim == 5:
            iqpl = theta.shape[2] // 2
            iql = theta.shape[3] // 2
            ik = theta.shape[4] // 2
            theta = theta[:, :, iqpl, iql, ik]
        T_list.append(theta)
        meta = dat['meta'].item() if isinstance(dat['meta'], np.ndarray) else dat['meta']
        R500.append(float(meta.get('R500_Mpc', 1.0)))
    return A_list, L0_list, T_list, np.array(R500, dtype=float)


def make_bilinear(gA, gL, T):
    gA = pt.as_tensor_variable(np.asarray(gA, dtype=np.float64))
    gL = pt.as_tensor_variable(np.asarray(gL, dtype=np.float64))
    T  = pt.as_tensor_variable(np.asarray(T, dtype=np.float64))
    def interp(A, L0):
        # assume uniform grids
        A  = pt.clip(A, gA[0], gA[-1])
        L0 = pt.clip(L0, gL[0], gL[-1])
        dA = gA[1] - gA[0]
        dL = gL[1] - gL[0]
        ia0 = pt.cast(pt.floor((A - gA[0]) / dA), 'int64')
        il0 = pt.cast(pt.floor((L0 - gL[0]) / dL), 'int64')
        ia0 = pt.clip(ia0, 0, gA.shape[0]-2)
        il0 = pt.clip(il0, 0, gL.shape[0]-2)
        A0, A1 = gA[ia0], gA[ia0+1]
        L00, L01 = gL[il0], gL[il0+1]
        t = pt.switch(pt.neq(A1, A0), (A - A0) / (A1 - A0), 0.0)
        u = pt.switch(pt.neq(L01, L00), (L0 - L00) / (L01 - L00), 0.0)
        f00 = T[ia0    , il0    ]
        f10 = T[ia0 + 1, il0    ]
        f01 = T[ia0    , il0 + 1]
        f11 = T[ia0 + 1, il0 + 1]
        return (1-t)*(1-u)*f00 + t*(1-u)*f10 + (1-t)*u*f01 + t*u*f11
    return interp


def build_model(catalog: pd.DataFrame, grids_dir: Path, gamma_fixed: float | None = None):
    clusters = list(catalog['cluster_name'].values)
    A_list, L0_list, T_list, R500 = load_grids(grids_dir / 'index.json', grids_dir, clusters)

    N = len(clusters)
    interps = [make_bilinear(A_list[i], L0_list[i], T_list[i]) for i in range(N)]

    with pm.Model() as model:
        mu_A    = pm.Normal('mu_A', 16.5, 1.5)
        sigma_A = pm.HalfNormal('sigma_A', 1.0)
        ell0s   = pm.Lognormal('ell0_star_kpc', np.log(200.0), 0.5)
        if gamma_fixed is None:
            gamma   = pm.Uniform('gamma', 0.0, 1.0)
        else:
            gamma   = pm.Deterministic('gamma', pm.math.constant(float(gamma_fixed)))
        sigma_int = pm.HalfNormal('sigma_int', 5.0)

        A_c = pm.Normal('A_c', mu=mu_A, sigma=sigma_A, shape=N)
        ell0_i = ell0s * (pt.as_tensor_variable(R500) ** gamma)

        theta_list = [interps[i](A_c[i], ell0_i[i]) for i in range(N)]
        theta_E = pt.stack(theta_list)
        pm.Deterministic('theta_E_model', theta_E)

        sigma_obs = catalog['sigma_theta_E'].values.astype(float)
        obs = catalog['theta_E_obs'].values.astype(float)
        pm.Normal('theta_E_obs', mu=theta_E, sigma=pt.sqrt(sigma_int**2 + sigma_obs**2), observed=obs)
    return model


def main():
    ap = argparse.ArgumentParser(description='Run NUTS with precomputed theta_E grids')
    ap.add_argument('--catalog', default=None)
    ap.add_argument('--tiers', default='1,2')
    ap.add_argument('--exclude', default=None)
    ap.add_argument('--include', default=None)
    ap.add_argument('--grids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--draws', type=int, default=3000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--gamma-fixed', type=float, default=None)
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(',')]
    exclude = args.exclude.split(',') if args.exclude else None
    include = args.include.split(',') if args.include else None
    catalog_path = Path(args.catalog) if args.catalog else None

    catalog = load_cluster_catalog(tiers, exclude, catalog_path, include)

    model = build_model(catalog, Path(args.grids), gamma_fixed=args.gamma_fixed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    try:
        with model:
            trace = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, target_accept=0.9,
                              return_inferencedata=True, cores=1, idata_kwargs={"log_likelihood": True})
    except Exception as e:
        print(f"NUTS failed ({e}), falling back to DEMetropolisZ...")
        with model:
            trace = pm.sample(draws=args.draws, tune=args.tune, chains=min(args.chains,2), step=pm.DEMetropolisZ(),
                              return_inferencedata=True, cores=1, idata_kwargs={"log_likelihood": True})

    trace.to_netcdf(outdir / 'trace.netcdf')

    import arviz as az
    summary = az.summary(trace, hdi_prob=0.68)
    summary.to_csv(outdir / 'summary.csv')

    try:
        waic = az.waic(trace)
        loo  = az.loo(trace)
        with open(outdir / 'metrics.json', 'w') as f:
            json.dump({
                'waic': float(waic.elpd_waic), 'waic_se': float(waic.se),
                'loo': float(loo.elpd_loo), 'loo_se': float(loo.se)
            }, f, indent=2)
    except Exception as e:
        with open(outdir / 'metrics.json', 'w') as f:
            json.dump({'waic': None, 'loo': None, 'error': str(e)}, f, indent=2)


if __name__ == '__main__':
    main()

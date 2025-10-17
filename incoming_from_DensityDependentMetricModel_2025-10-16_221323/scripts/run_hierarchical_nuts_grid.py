#!/usr/bin/env python3
"""
run_hierarchical_nuts_grid.py

Wrapper to run the NUTS grid-based hierarchical model using precomputed theta_E grids.
Matches the CLI proposed in the execution plan; forwards to run_mass_scaled_nuts_grid.
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_mass_scaled_nuts_grid as nuts2d
from scripts.run_mass_scaled_hierarchical_inference import load_cluster_catalog


def main():
    ap = argparse.ArgumentParser(description='Hierarchical NUTS with grid-interpolated theta_E')
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--tiers', default='1,2')
    ap.add_argument('--exclude', default=None)
    ap.add_argument('--include', default=None)
    ap.add_argument('--grids', required=True, help='Dir with index.json and per-cluster NPZ grids')
    ap.add_argument('--pzs', default='mode:mixture', help='Source redshift handling (accepted but not used here)')
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--draws', type=int, default=4000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--target_accept', type=float, default=0.9)
    ap.add_argument('--gamma-fixed', type=float, default=None, help='If set, request gamma=0 (not yet supported)')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(',')]
    exclude = args.exclude.split(',') if args.exclude else None
    include = args.include.split(',') if args.include else None

    # Build catalog (for consistency checks)
    catalog = load_cluster_catalog(tiers, exclude, Path(args.catalog), include)
    model = nuts2d.build_model(catalog, Path(args.grids))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Sample using nuts2d runner logic
    import pymc as pm
    try:
        with model:
            trace = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept,
                              return_inferencedata=True, cores=1, idata_kwargs={"log_likelihood": True})
    except Exception as e:
        print(f"NUTS failed ({e}), falling back to DEMetropolisZ...")
        with model:
            trace = pm.sample(draws=args.draws, tune=args.tune, chains=min(args.chains,2), step=pm.DEMetropolisZ(),
                              return_inferencedata=True, cores=1, idata_kwargs={"log_likelihood": True})

    trace.to_netcdf(outdir / 'trace.netcdf')

    # Save diagnostics as in nuts2d
    import arviz as az, json
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
#!/usr/bin/env python3
"""
Convert ArviZ NetCDF trace from PyMC calibration to NPZ format expected by validate_holdout_mass_scaled.py.
- Inputs: --trace path/to/trace.nc, --manifest path/to/manifest.json
- Output: flat_samples_from_pymc.npz with fields: samples (N x 4) and manifest (JSON string)
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import arviz as az


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    trace = az.from_netcdf(args.trace)
    ell0 = np.ravel(trace.posterior['ell0_star_pop'].values)
    gamma = np.ravel(trace.posterior['gamma_pop'].values)
    muA = np.ravel(trace.posterior['mu_A_pop'].values)
    sigA = np.ravel(trace.posterior['sigma_A_pop'].values)
    n = min(len(ell0), len(gamma), len(muA), len(sigA))
    samples = np.vstack([ell0[:n], gamma[:n], muA[:n], sigA[:n]]).T

    manifest = json.loads(Path(args.manifest).read_text())
    out_path = Path(args.out) if args.out else Path(args.trace).with_name('flat_samples_from_pymc.npz')
    np.savez(out_path, samples=samples, manifest=json.dumps(manifest))
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()

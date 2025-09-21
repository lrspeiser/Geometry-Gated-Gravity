# -*- coding: utf-8 -*-
"""
root-m/pde/run_sparc_pde.py

End-to-end PDE run for SPARC-like tables (spherical-equivalent prototype):
- Build spherical rho_b(R,z) from Vbar(r) in predictions_by_radius.csv
- Solve ∇·(|∇φ| ∇φ) = - S0 ρ_b
- Predict v(R) from g_phi(R,0) + baryon g_N
- Write CSV + PNG under root-m/out/pde_sparc/<TAG>/

Note: This uses a spherical-equivalent rho_b derived from Vbar(r).
Disc geometry can be incorporated later by building axisymmetric rho_b maps.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

from .solve_phi import SolverParams, solve_axisym
from .baryon_maps import sparc_map_from_predictions
from .predict_rc import predict_v_from_phi_equatorial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--outdir', default='root-m/out/pde_sparc')
    ap.add_argument('--tag', default='all')
    ap.add_argument('--Rmax', type=float, default=80.0)
    ap.add_argument('--Zmax', type=float, default=80.0)
    ap.add_argument('--NR', type=int, default=128)
    ap.add_argument('--NZ', type=int, default=128)
    ap.add_argument('--S0', type=float, default=1.0e-7)
    ap.add_argument('--rc_kpc', type=float, default=15.0)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    df = pd.read_csv(in_path)

    # PDE map from spherical-equivalent rho_b
    Z, R, rho = sparc_map_from_predictions(in_path, R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ)

    # Solve PDE
    params = SolverParams(S0=args.S0, rc_kpc=args.rc_kpc)
    phi, gR, gZ = solve_axisym(R, Z, rho, params)

    # Predict v(R) at observed radii
    r_eval = np.asarray(df['R_kpc'], float)
    vbar = np.asarray(df['Vbar_kms'], float)
    v_pred, gphi, gN = predict_v_from_phi_equatorial(R, gR, r_eval, vbar)

    err = 100.0 * np.abs(v_pred - df['Vobs_kms']) / np.maximum(df['Vobs_kms'], 1e-9)
    med = float(np.median(100.0 - err))

    od = Path(args.outdir)/args.tag
    od.mkdir(parents=True, exist_ok=True)
    # save RC table
    out_csv = od/'rc_pde_predictions.csv'
    pd.DataFrame({'R_kpc': r_eval, 'Vobs_kms': df['Vobs_kms'], 'Vbar_kms': vbar,
                  'Vpred_pde_kms': v_pred, 'percent_close': (100.0 - err)}).to_csv(out_csv, index=False)
    # metrics
    (od/'summary.json').write_text(json.dumps({'S0': args.S0, 'rc_kpc': args.rc_kpc,
                                               'median_percent_close': med}, indent=2))

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(r_eval, df['Vobs_kms'], 'k.', ms=3, label='obs')
    plt.plot(r_eval, vbar, 'c--', lw=1.2, label='baryon')
    plt.plot(r_eval, v_pred, 'r-', lw=1.2, label='PDE pred')
    plt.xlabel('R [kpc]'); plt.ylabel('V [km/s]'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(od/'rc_pde_results.png', dpi=140); plt.close()

if __name__ == '__main__':
    main()

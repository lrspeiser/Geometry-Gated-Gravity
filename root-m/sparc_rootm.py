# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# G in (kpc km^2 s^-2 Msun^-1)
G = 4.300917270e-6

def compute_rootm_vpred(vbar_kms, r_kpc, A_kms=140.0, Mref_Msun=6.0e10):
    r = np.asarray(r_kpc, float)
    vbar2 = np.asarray(vbar_kms, float)**2
    M_enclosed = (vbar2 * r) / G
    v_tail2 = (A_kms**2) * np.sqrt(np.clip(M_enclosed / Mref_Msun, 0.0, None))
    v2 = vbar2 + v_tail2
    return np.sqrt(np.clip(v2, 0.0, None))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--out_dir', default='root-m/out/sparc')
    ap.add_argument('--A_kms', type=float, default=140.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    needed = ['R_kpc','Vbar_kms','Vobs_kms','galaxy','type']
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {in_path}")

    vpred = compute_rootm_vpred(df['Vbar_kms'], df['R_kpc'], A_kms=args.A_kms, Mref_Msun=args.Mref)
    df_out = df.copy()
    df_out['Vpred_rootm_kms'] = vpred
    # percent error and closeness
    err_pct = 100.0 * np.abs(df_out['Vpred_rootm_kms'] - df_out['Vobs_kms']) / np.maximum(df_out['Vobs_kms'], 1e-9)
    df_out['model_percent_off_rootm'] = err_pct
    df_out['percent_close_rootm'] = 100.0 - err_pct

    by_radius_path = out_dir/'sparc_predictions_by_radius_rootm.csv'
    df_out.to_csv(by_radius_path, index=False)

    # per-galaxy summary
    grp_gal = df_out.groupby('galaxy', as_index=False).agg(
        galaxy_type=('type','first'),
        N=('galaxy','size'),
        median_percent_close_rootm=('percent_close_rootm','median'),
        mean_percent_close_rootm=('percent_close_rootm','mean'),
        median_model_percent_off_rootm=('model_percent_off_rootm','median'),
        mean_model_percent_off_rootm=('model_percent_off_rootm','mean'),
    )
    by_gal_path = out_dir/'sparc_predictions_by_galaxy_rootm.csv'
    grp_gal.to_csv(by_gal_path, index=False)

    # by-type summary (outer points if is_outer exists)
    df_eff = df_out.copy()
    if 'is_outer' in df_eff.columns:
        df_eff = df_eff[df_eff['is_outer']==True]
    grp_type = df_eff.groupby(['type'], as_index=False).agg(
        galaxies=('galaxy','nunique'),
        points=('galaxy','size'),
        median_percent_close_rootm=('percent_close_rootm','median'),
        mean_percent_close_rootm=('percent_close_rootm','mean'),
    )
    by_type_path = out_dir/'sparc_summary_by_type_rootm.csv'
    grp_type.to_csv(by_type_path, index=False)

    # simple summary JSON
    summary = {
        'A_kms': args.A_kms,
        'Mref_Msun': args.Mref,
        'files': {
            'input': str(in_path),
            'predictions_by_radius_rootm': str(by_radius_path),
            'predictions_by_galaxy_rootm': str(by_gal_path),
            'summary_by_type_rootm': str(by_type_path),
        },
        'global': {
            'median_percent_close_rootm_all': float(df_out['percent_close_rootm'].median()),
            'mean_percent_close_rootm_all': float(df_out['percent_close_rootm'].mean()),
        }
    }
    (out_dir/'summary_rootm.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

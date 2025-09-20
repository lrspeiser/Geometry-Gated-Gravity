# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Tuple

from root_m.common import v_tail2_rootm_soft as v_tail2_soft  # safe import path

# G in (kpc km^2 s^-2 Msun^-1)
G = 4.300917270e-6

def compute_rootm_soft_vpred(vbar_kms, r_kpc, A_kms=140.0, Mref_Msun=6.0e10, rc_kpc=15.0):
    r = np.asarray(r_kpc, float)
    vbar2 = np.asarray(vbar_kms, float)**2
    M_enclosed = (vbar2 * r) / G
    v_tail2 = v_tail2_soft(r, M_enclosed, A_kms=A_kms, Mref=Mref_Msun, rc_kpc=rc_kpc)
    v2 = vbar2 + v_tail2
    return np.sqrt(np.clip(v2, 0.0, None))


def kfold_by_galaxy(galaxies: List[str], k: int) -> List[Tuple[List[str], List[str]]]:
    unique = sorted(set(galaxies))
    n = len(unique)
    folds = []
    for i in range(k):
        test = [g for idx, g in enumerate(unique) if idx % k == i]
        train = [g for g in unique if g not in test]
        folds.append((train, test))
    return folds


def score_pair_cv(df, A, rc, k=5) -> float:
    # Evaluate median percent_close on held-out outer points
    galaxies = df['galaxy'].tolist()
    folds = kfold_by_galaxy(galaxies, k)
    all_pc = []
    for train_g, test_g in folds:
        dtest = df[df['galaxy'].isin(test_g)].copy()
        if 'is_outer' in dtest.columns:
            dtest = dtest[dtest['is_outer']==True]
        if dtest.empty:
            continue
        vpred = compute_rootm_soft_vpred(dtest['Vbar_kms'], dtest['R_kpc'], A_kms=A, Mref_Msun=6.0e10, rc_kpc=rc)
        err_pct = 100.0 * np.abs(vpred - dtest['Vobs_kms']) / np.maximum(dtest['Vobs_kms'], 1e-9)
        pc = 100.0 - err_pct
        all_pc.append(pc.to_numpy())
    if not all_pc:
        return -np.inf
    all_pc = np.concatenate(all_pc)
    return float(np.median(all_pc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--out_dir', default='root-m/out/sparc_soft')
    ap.add_argument('--A_kms', type=float, default=140.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    ap.add_argument('--rc_kpc', type=float, default=15.0)
    ap.add_argument('--cv', type=int, default=0, help='k-fold CV by galaxy (0 disables)')
    ap.add_argument('--A_grid', default='120,140,160,180')
    ap.add_argument('--rc_grid', default='10,15,20')
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    needed = ['R_kpc','Vbar_kms','Vobs_kms','galaxy','type']
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {in_path}")

    best = dict(A_kms=args.A_kms, rc_kpc=args.rc_kpc, cv_median=None)
    if args.cv and args.cv > 1:
        A_list = [float(x) for x in args.A_grid.split(',') if x.strip()]
        rc_list = [float(x) for x in args.rc_grid.split(',') if x.strip()]
        best_score = -1e99
        for A in A_list:
            for rc in rc_list:
                s = score_pair_cv(df, A, rc, k=args.cv)
                if s > best_score:
                    best_score = s
                    best = dict(A_kms=A, rc_kpc=rc, cv_median=s)

    # Predictions on full table with chosen (A, rc)
    vpred = compute_rootm_soft_vpred(df['Vbar_kms'], df['R_kpc'], A_kms=best['A_kms'], Mref_Msun=args.Mref, rc_kpc=best['rc_kpc'])
    df_out = df.copy()
    df_out['Vpred_rootm_soft_kms'] = vpred
    err_pct = 100.0 * np.abs(df_out['Vpred_rootm_soft_kms'] - df_out['Vobs_kms']) / np.maximum(df_out['Vobs_kms'], 1e-9)
    df_out['model_percent_off_rootm_soft'] = err_pct
    df_out['percent_close_rootm_soft'] = 100.0 - err_pct

    by_radius_path = out_dir/'sparc_predictions_by_radius_rootm_soft.csv'
    df_out.to_csv(by_radius_path, index=False)

    # per-galaxy summary
    grp_gal = df_out.groupby('galaxy', as_index=False).agg(
        galaxy_type=('type','first'),
        N=('galaxy','size'),
        median_percent_close_rootm_soft=('percent_close_rootm_soft','median'),
        mean_percent_close_rootm_soft=('percent_close_rootm_soft','mean'),
        median_model_percent_off_rootm_soft=('model_percent_off_rootm_soft','median'),
        mean_model_percent_off_rootm_soft=('model_percent_off_rootm_soft','mean'),
    )
    by_gal_path = out_dir/'sparc_predictions_by_galaxy_rootm_soft.csv'
    grp_gal.to_csv(by_gal_path, index=False)

    # by-type summary (outer points if is_outer exists)
    df_eff = df_out.copy()
    if 'is_outer' in df_eff.columns:
        df_eff = df_eff[df_eff['is_outer']==True]
    grp_type = df_eff.groupby(['type'], as_index=False).agg(
        galaxies=('galaxy','nunique'),
        points=('galaxy','size'),
        median_percent_close_rootm_soft=('percent_close_rootm_soft','median'),
        mean_percent_close_rootm_soft=('percent_close_rootm_soft','mean'),
    )
    by_type_path = out_dir/'sparc_summary_by_type_rootm_soft.csv'
    grp_type.to_csv(by_type_path, index=False)

    # summary JSON
    summary = {
        'A_kms': best['A_kms'],
        'rc_kpc': best['rc_kpc'],
        'Mref_Msun': args.Mref,
        'cv_folds': int(args.cv),
        'cv_median_percent_close_outer': float(best['cv_median']) if best['cv_median'] is not None else None,
        'files': {
            'input': str(in_path),
            'predictions_by_radius_rootm_soft': str(by_radius_path),
            'predictions_by_galaxy_rootm_soft': str(by_gal_path),
            'summary_by_type_rootm_soft': str(by_type_path),
        },
        'global': {
            'median_percent_close_rootm_soft_all': float(df_out['percent_close_rootm_soft'].median()),
            'mean_percent_close_rootm_soft_all': float(df_out['percent_close_rootm_soft'].mean()),
        }
    }
    (out_dir/'summary_rootm_soft.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

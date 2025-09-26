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
    ap.add_argument('--in', dest='in_path', required=True, help='MW predictions_by_radius CSV path')
    ap.add_argument('--out_dir', default='root-m/out/mw')
    ap.add_argument('--A_kms', type=float, default=140.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    # Optional SOG (dynamics) flags for MW; default OFF
    ap.add_argument('--use_sog_fe', action='store_true')
    ap.add_argument('--use_sog_rho2', action='store_true')
    ap.add_argument('--use_sog_rg', action='store_true')
    ap.add_argument('--sog_sigma_star', type=float, default=100.0)
    ap.add_argument('--sog_g_star', type=float, default=1200.0)
    ap.add_argument('--sog_aSigma', type=float, default=2.0)
    ap.add_argument('--sog_ag', type=float, default=2.0)
    ap.add_argument('--sog_fe_lambda', type=float, default=1.0)
    ap.add_argument('--sog_rho2_eta', type=float, default=0.01)
    ap.add_argument('--sog_rg_A', type=float, default=0.8)
    ap.add_argument('--sog_rg_n', type=float, default=1.2)
    ap.add_argument('--sog_rg_g0', type=float, default=5.0)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    needed = ['R_kpc','Vbar_kms','Vobs_kms']
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {in_path}")

    vpred = compute_rootm_vpred(df['Vbar_kms'], df['R_kpc'], A_kms=args.A_kms, Mref_Msun=args.Mref)

    # Optional SOG augmentation for MW dynamics
    if args.use_sog_fe or args.use_sog_rho2 or args.use_sog_rg:
        # Dynamic import second_order
        try:
            import importlib.util as _ilu
            from pathlib import Path as _P
            _sog_path = _P(__file__).resolve().parent / 'pde' / 'second_order.py'
            spec = _ilu.spec_from_file_location('second_order', str(_sog_path))
            second_order = _ilu.module_from_spec(spec); spec.loader.exec_module(second_order)
        except Exception:
            second_order = None
        if second_order is not None:
            R = df['R_kpc'].to_numpy(float)
            Vbar = df['Vbar_kms'].to_numpy(float)
            g1 = (Vbar*Vbar) / np.maximum(R, 1e-9)
            # Sigma proxy from M(<R): Sigma = (1/(2πR)) dM/dR, with M=Vbar^2 R / G
            Menc = (Vbar*Vbar) * R / G
            dM_dR = np.gradient(Menc, R)
            with np.errstate(divide='ignore', invalid='ignore'):
                Sigma_kpc2 = dM_dR / (2.0 * np.pi * np.maximum(R, 1e-12))
            Sigma_pc2 = np.clip(Sigma_kpc2 / 1e6, 0.0, None)
            gate = {
                'Sigma_star': float(args.sog_sigma_star),
                'g_star': float(args.sog_g_star),
                'aSigma': float(args.sog_aSigma),
                'ag': float(args.sog_ag),
            }
            g2_add = np.zeros_like(g1)
            if args.use_sog_fe:
                g2_add += second_order.g2_field_energy(R, g1, Sigma_pc2, {'lambda': float(args.sog_fe_lambda), **gate})
            if args.use_sog_rho2:
                rho_proxy = (Sigma_pc2 * 1e6) / (2.0 * 0.3)
                g2_add += second_order.g2_rho2_local(R, rho_proxy, g1, Sigma_pc2, {'eta': float(args.sog_rho2_eta), **gate})
            if args.use_sog_rg:
                g2_add += second_order.g_runningG(R, g1, Sigma_pc2, {'A': float(args.sog_rg_A), 'n': float(args.sog_rg_n), 'g0': float(args.sog_rg_g0), **gate})
            v2 = np.clip(Vbar*Vbar + g2_add * R, 0.0, None)
            vpred = np.sqrt(v2)
    df_out = df.copy()
    df_out['Vpred_rootm_kms'] = vpred
    err_pct = 100.0 * np.abs(df_out['Vpred_rootm_kms'] - df_out['Vobs_kms']) / np.maximum(df_out['Vobs_kms'], 1e-9)
    df_out['model_percent_off_rootm'] = err_pct
    df_out['percent_close_rootm'] = 100.0 - err_pct

    by_radius_path = out_dir/'mw_predictions_by_radius_rootm.csv'
    df_out.to_csv(by_radius_path, index=False)

    summary = {
        'A_kms': args.A_kms,
        'Mref_Msun': args.Mref,
        'files': {'input': str(in_path), 'predictions_by_radius_rootm': str(by_radius_path)},
'global': {
            'median_percent_close_rootm_all': float(df_out['percent_close_rootm'].median()),
            'mean_percent_close_rootm_all': float(df_out['percent_close_rootm'].mean()),
        },
        'sog': {
            'used': bool(args.use_sog_fe or args.use_sog_rho2 or args.use_sog_rg),
            'mode': 'fe' if args.use_sog_fe else ('rho2' if args.use_sog_rho2 else ('rg' if args.use_sog_rg else 'none'))
        }
    }
    (out_dir/'summary_rootm.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

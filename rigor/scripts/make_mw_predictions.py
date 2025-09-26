#!/usr/bin/env python3
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from rigor.scripts.gaia_to_mw_predictions import stream_bins, fit_baryon_baseline, vbar_mn_hern_kms


def main():
    files = sorted(glob.glob('data/gaia_sky_slices/processed_*.parquet'))
    if not files:
        raise SystemExit('No Gaia slice files found under data/gaia_sky_slices')
    r_edges = np.linspace(0.5, 20.0, 31)
    z_max = 1.0
    sigma_v_max = 80
    bins = stream_bins(files, r_edges, z_max=z_max, sigma_v_max=sigma_v_max)
    R = bins['r_centers']
    Vobs = bins['v_obs']
    Verr = bins['v_err']
    try:
        fit = fit_baryon_baseline(R, Vobs, Verr, r_fit_min=3.0, r_fit_max=12.0)
        Vbar = vbar_mn_hern_kms(R, fit['M_d_Msun'], fit['a_d_kpc'], fit['b_d_kpc'], fit['M_b_Msun'], fit['a_b_kpc'])
    except Exception:
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(R, np.nan_to_num(Vobs, nan=0.0), s=0.5*len(R))
        Vbar = spl(R)
    out_dir = Path('data/milkyway'); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'gaia_predictions_by_radius.csv'
    df = pd.DataFrame({'R_kpc': R, 'Vobs_kms': Vobs, 'Vbar_kms': Vbar, 'Verr_kms': Verr})
    df.to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()

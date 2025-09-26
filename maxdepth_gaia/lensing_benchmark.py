# lensing_benchmark.py
from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np

from .utils import C_KMS

"""
Strong-lensing benchmark scaffolding (SLACS-style)

This module provides a minimal, model-agnostic evaluator to compare predicted
Einstein angles across models using a SIS-like mapping between a flat
circular speed and lensing deflection:

  theta_E (radians) = 2 * pi * (v_flat^2 / c^2) * (D_ls / D_s)

Notes:
- For GR+NFW and GR baselines, supply a representative flat circular speed
  (e.g., asymptotic v_c or v_c at a chosen scale).
- For the saturated-well model, v_flat is reported in fit_params.json.
- For MOND, a consistent relativistic lensing prescription (e.g., TeVeS/QUMOND)
  should be adopted; here we provide the same SIS-like mapping for first-order
  comparisons. Replace with the desired lensing law as needed.

Expected CSV columns (distances in the same units, e.g., Mpc):
  name, z_l, z_s, D_l, D_s, D_ls

Outputs:
  CSV with columns: name, model, theta_E_arcsec_pred
"""

def _theta_E_arcsec_from_vflat(v_flat_kms: float, D_ls_over_D_s: float) -> float:
    alpha_rad = 2.0 * np.pi * (v_flat_kms**2) / (C_KMS**2)
    theta_rad = alpha_rad * D_ls_over_D_s
    return float(theta_rad * (180.0/np.pi) * 3600.0)


def run_lensing_eval(lens_csv_path: str, fit_params_path: str, out_csv_path: str,
                     vflat_overrides: dict | None = None) -> None:
    df = pd.read_csv(lens_csv_path)
    with open(fit_params_path, 'r') as f:
        fit = json.load(f)

    rows = []
    # Saturated-well v_flat
    vflat_sw = (fit.get('saturated_well', {}) or {}).get('v_flat', np.nan)

    # Optionally provide other model-specific v_flat proxies here
    vflat_map = {
        'SatWell': vflat_sw,
    }
    if vflat_overrides:
        vflat_map.update(vflat_overrides)

    for _, r in df.iterrows():
        D_ratio = float(r['D_ls'])/float(r['D_s']) if float(r['D_s']) != 0 else np.nan
        for model, vflat in vflat_map.items():
            if np.isfinite(vflat) and np.isfinite(D_ratio):
                theta_arcsec = _theta_E_arcsec_from_vflat(float(vflat), D_ratio)
            else:
                theta_arcsec = np.nan
            rows.append(dict(name=r.get('name', ''), model=model, theta_E_arcsec_pred=theta_arcsec,
                             z_l=r.get('z_l', np.nan), z_s=r.get('z_s', np.nan)))

    pd.DataFrame(rows).to_csv(out_csv_path, index=False)

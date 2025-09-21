# -*- coding: utf-8 -*-
"""
root-m/pde/predict_rc.py

Compute rotation curves from a PDE solution φ(R,z) by sampling g_R(R,0)
and combining with the Newtonian baryon contribution (from Vbar or M(<r)).
"""
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def predict_v_from_phi_equatorial(R_grid: np.ndarray, gR_grid: np.ndarray,
                                  R_eval: np.ndarray,
                                  vbar_kms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
      R_grid: 1D R coordinates for PDE grid (NR)
      gR_grid: 2D gR(Z,R); we sample at z=0 plane (nearest row)
      R_eval: radii to evaluate (same units)
      vbar_kms: baryonic circular speed at R_eval (km/s)

    Returns:
      v_pred (km/s), g_phi (km^2 s^-2 / kpc), g_N (km^2 s^-2 / kpc)
    """
    # pick z=0 row
    NZ = gR_grid.shape[0]
    z0_idx = NZ//2
    gR0 = gR_grid[z0_idx, :]
    g_phi = np.interp(R_eval, R_grid, gR0, left=gR0[0], right=gR0[-1])
    g_N = (vbar_kms**2) / np.maximum(R_eval, 1e-9)
    v2 = np.clip(vbar_kms**2 + g_phi * R_eval, 0.0, None)
    return np.sqrt(v2), g_phi, g_N

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Gravitational constant in convenient units: kpc (km/s)^2 per Msun
# G ≈ 4.30091e-6 kpc (km/s)^2 / Msun
G_KPC_KMS2_PER_MSUN = 4.30091e-6

@dataclass
class EffectiveMassPoint:
    R_kpc: float
    g_req: float
    g_bar: float
    gX: float
    M_bar_eff_Msun: float
    M_X_eff_Msun: float
    f_X_over_bar: float


def compute_effective_mass_arrays(R_kpc, Vobs_kms, Vbar_kms):
    """Compute effective enclosed masses from rotation-curve quantities.
    Using spherical-equivalent relations:
      g = V^2 / R;  M(<R) = g R^2 / G = V^2 R / G.
    Returns arrays (M_bar_eff, M_X_eff, f_ratio, gX).
    """
    R = np.asarray(R_kpc)
    Vobs = np.asarray(Vobs_kms)
    Vbar = np.asarray(Vbar_kms)
    eps = 1e-12
    g_req = (Vobs * Vobs) / (R + eps)
    g_bar = (Vbar * Vbar) / (R + eps)
    gX = g_req - g_bar
    M_bar_eff = (Vbar * Vbar) * R / G_KPC_KMS2_PER_MSUN
    M_X_eff = gX * (R * R) / G_KPC_KMS2_PER_MSUN
    # Avoid division by zero
    f_ratio = M_X_eff / np.maximum(M_bar_eff, 1e-12)
    return M_bar_eff, M_X_eff, f_ratio, gX
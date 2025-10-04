#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, G, compute_potential_depth
)

CLUSTERS = [
    ("a209", "ABELL_0209"),
    ("macs0416", "MACSJ0416"),
    ("rxj1347", "RXJ1347"),
]

RADII = [100.0, 250.0]  # kpc


def enclosed_mass(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    M_enc = np.zeros_like(r)
    if r.size > 1:
        integrand = rho * r * r
        M_enc[1:] = 4.0 * np.pi * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
    return M_enc


def main():
    rows = []
    for cid, local in CLUSTERS:
        try:
            r, rho = load_real_cluster_profiles(local)
        except Exception as e:
            rows.append({"cluster": cid, "error": str(e)})
            continue
        r = np.asarray(r, float); rho = np.asarray(rho, float)
        M_enc = enclosed_mass(r, rho)
        # Build an R grid consistent with compute_cluster
        R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
        # Interpolate M_enc to R for measured Phi
        M_enc_R = np.interp(R, r, np.maximum(M_enc, 0.0))
        Phi_meas = compute_potential_depth(R, M_enc_R)  # km^2/s^2, array across R
        # Totals
        M_tot = float(M_enc[-1]) if M_enc.size else 0.0
        # Prepare outputs at each requested radius
        for Rq in RADII:
            # Use nearest available radius if profile doesn't extend that far
            if Rq > R[-1]:
                Ruse = float(R[-1])
            elif Rq < R[0]:
                Ruse = float(R[0])
            else:
                Ruse = float(Rq)
            # Interpolate measured Phi at Ruse
            phi_meas = float(np.interp(Ruse, R, Phi_meas))
            # Simple expected using total mass
            phi_expected_total = float(G * max(M_tot, 0.0) / max(Ruse, 1e-12))
            # Local expected using enclosed mass at Ruse
            M_local = float(np.interp(Ruse, R, M_enc_R))
            phi_expected_local = float(G * max(M_local, 0.0) / max(Ruse, 1e-12))
            rows.append({
                "cluster": cid,
                "R_kpc_requested": Rq,
                "R_kpc_used": Ruse,
                "Phi_measured_km2s2": phi_meas,
                "Phi_expected_total_km2s2": phi_expected_total,
                "Phi_expected_local_km2s2": phi_expected_local,
                "ratio_meas_over_expected_total": None if phi_expected_total == 0 else phi_meas / phi_expected_total,
                "ratio_meas_over_expected_local": None if phi_expected_local == 0 else phi_meas / phi_expected_local,
                "M_total_Msun": M_tot,
                "M_local_at_R_Msun": M_local
            })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

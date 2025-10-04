#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    G, KPC_CM, MSUN_G, M_P_G, ne_to_rho_gas_Msun_kpc3, compute_potential_depth
)

# Map short IDs to local folder names
CLUSTERS: List[Tuple[str, str]] = [
    ("A209", "ABELL_0209"),
    ("MACS0416", "MACSJ0416"),
    ("RXJ1347", "RXJ1347"),
]

RADII_AUDIT = [50.0, 100.0, 250.0, 500.0]
R_REPORT = 250.0


def enclosed_mass_from_density(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    rho = np.asarray(rho, float)
    # Ensure increasing r and finite arrays
    m = np.isfinite(r) & np.isfinite(rho) & (r > 0)
    r = r[m]; rho = rho[m]
    i = np.argsort(r)
    r = r[i]; rho = rho[i]
    M_enc = np.zeros_like(r)
    if r.size > 1:
        integrand = rho * r * r
        M_enc[1:] = 4.0 * np.pi * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
    return r, M_enc


def interp_enclosed_at(Rq: float, r: np.ndarray, M_enc: np.ndarray) -> float:
    Rq = float(Rq)
    if Rq <= r[0]:
        return float(M_enc[0])
    if Rq >= r[-1]:
        return float(M_enc[-1])
    return float(np.interp(Rq, r, M_enc))


def load_cluster_profiles(cluster_local: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = ROOT / 'data' / 'clusters' / cluster_local
    gpath = base / 'gas_profile.csv'
    spath = base / 'stars_profile.csv'
    if not gpath.exists():
        raise FileNotFoundError(f"Missing gas_profile.csv for {cluster_local}")
    gp = pd.read_csv(gpath)
    # Gas: accept either rho_gas or n_e
    if 'rho_gas_Msun_per_kpc3' in gp.columns:
        r_g = gp['r_kpc'].to_numpy(float)
        rho_g = gp['rho_gas_Msun_per_kpc3'].to_numpy(float)
    elif 'n_e_cm3' in gp.columns:
        r_g = gp['r_kpc'].to_numpy(float)
        rho_g = ne_to_rho_gas_Msun_kpc3(gp['n_e_cm3'].to_numpy(float))
    else:
        raise KeyError(f"gas_profile.csv missing rho_gas_Msun_per_kpc3 or n_e_cm3 for {cluster_local}")

    # Stars: required
    if not spath.exists():
        raise FileNotFoundError(f"Missing stars_profile.csv for {cluster_local}")
    sp = pd.read_csv(spath)
    if 'rho_star_Msun_per_kpc3' not in sp.columns:
        raise KeyError(f"stars_profile.csv missing rho_star_Msun_per_kpc3 for {cluster_local}")
    r_s = sp['r_kpc'].to_numpy(float)
    rho_s = sp['rho_star_Msun_per_kpc3'].to_numpy(float)

    return r_g, rho_g, r_s, rho_s


def measured_phi_at(Rq: float, r: np.ndarray, rho: np.ndarray) -> float:
    # Build an R grid spanning available range
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    # M_enc and potential depth
    _, M_enc = enclosed_mass_from_density(r, rho)
    M_R = np.interp(R, r, np.maximum(M_enc, 0.0))
    Phi = compute_potential_depth(R, M_R)
    return float(np.interp(Rq, R, Phi))


def accept_name_audit(pattern: str = '1347') -> Dict:
    path = ROOT / 'data' / 'ACCEPT.dat'
    if not path.exists():
        return {"error": f"{path} not found"}
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith('#') or ls.startswith('###'):
                continue
            parts = ls.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            if pattern.lower() in name.lower():
                try:
                    Rin = float(parts[1]); Rout = float(parts[2])
                    r_mid_kpc = 0.5 * (Rin + Rout) * 1000.0
                    ne = float(parts[3])
                    rows.append({"Name": name, "r_kpc": r_mid_kpc, "n_e_cm3": ne})
                except Exception:
                    continue
    return {"pattern": pattern, "count": len(rows), "head": rows[:5]}


def main():
    audit_rows = []
    a209_profile_rows = []

    for cid, local in CLUSTERS:
        r_g, rho_g, r_s, rho_s = load_cluster_profiles(local)
        # Gas mass profile
        r_g_sorted, Mgas = enclosed_mass_from_density(r_g, rho_g)
        # Stars mass profile
        r_s_sorted, Mstar = enclosed_mass_from_density(r_s, rho_s)
        # Combine on a common grid to compute |Phi| at R_REPORT
        r_union = np.union1d(r_g_sorted, r_s_sorted)
        rho_g_u = np.interp(r_union, r_g_sorted, rho_g)
        rho_s_u = np.interp(r_union, r_s_sorted, rho_s)
        rho_tot = rho_g_u + rho_s_u

        # Values at R_REPORT
        Mgas_R = interp_enclosed_at(R_REPORT, r_g_sorted, Mgas)
        Mstar_R = interp_enclosed_at(R_REPORT, r_s_sorted, Mstar)
        Mbar_R = Mgas_R + Mstar_R
        Phi_R = measured_phi_at(R_REPORT, r_union, rho_tot)

        audit_rows.append({
            "cluster": cid,
            "M_gas_lt250_Msun": Mgas_R,
            "M_star_lt250_Msun": Mstar_R,
            "M_baryon_lt250_Msun": Mbar_R,
            "Phi_250_km2s2": Phi_R,
        })

        if cid == 'A209':
            for Rq in RADII_AUDIT:
                Mgas_q = interp_enclosed_at(Rq, r_g_sorted, Mgas)
                a209_profile_rows.append({"R_kpc": Rq, "M_gas_Msun": Mgas_q})

    rxj_audit = accept_name_audit('1347')

    print(json.dumps({
        "audit": audit_rows,
        "A209_Mgas_profile": a209_profile_rows,
        "RXJ1347_accept_name_audit": rxj_audit
    }, indent=2))


if __name__ == '__main__':
    main()

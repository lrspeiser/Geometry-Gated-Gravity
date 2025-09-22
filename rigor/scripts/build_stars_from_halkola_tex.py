# -*- coding: utf-8 -*-
"""
rigor/scripts/build_stars_from_halkola_tex.py

Extract a BCG stellar profile for Abell 1689 from the local Halkola et al. TeX
source by selecting the brightest galaxy in the Sersic-fit table, and build a
3D Hernquist-equivalent stars_profile.csv on the cluster gas radius grid.

Notes
- This script approximates the deprojection by mapping Sersic Re to a Hernquist
  scale via Re ≈ 1.8153 a (exact for n=4; acceptable for n≈4-5).
- Stellar mass normalization uses a fiducial Mtot_Msun (configurable).
- Optionally adds a diffuse ICL Hernquist component.
- Output matches our cluster runner schema: r_kpc,rho_star_Msun_per_kpc3

Usage:
  py -u rigor\scripts\build_stars_from_halkola_tex.py \
     --tex data/Halkola-abell-paper-sources/halkola_a1689_strong_lensing.tex \
     --cluster_dir data/clusters/ABELL_1689 \
     --Mtot_Msun 1.0e12 --icl_Mtot_Msun 5.0e11 --icl_a_kpc 100.0

"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

RE_TABLE_LINE = re.compile(r"^\s*(\d+)\s*&.*\$([\d\.]+)\\pm[\d\.]+\$\s*&\s*\$([\d\.]+)\\pm[\d\.]+\$\s*&\s*\$([\d\.]+)\\pm[\d\.]+\$")
# Captures: ID, m_AB, n_ser, Re_kpc (more permissive)

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def hernquist_rho(r_kpc: np.ndarray, Mtot_Msun: float, a_kpc: float) -> np.ndarray:
    r = np.asarray(r_kpc)
    return (Mtot_Msun * a_kpc) / (2.0 * np.pi * np.maximum(r, 1e-12) * (r + a_kpc)**3)


def parse_brightest_from_tex(tex_path: Path) -> dict:
    best = None
    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if '&' not in ln or '$' not in ln:
                continue
            # Use robust split-by-& parsing
            parts = [p.strip() for p in ln.split('&')]
            if len(parts) < 6:
                continue
            try:
                obj_id = int(parts[0])
            except Exception:
                continue
            try:
                m_ab_tok = parts[3]
                n_ser_tok = parts[4]
                Re_tok = parts[5]
                def _val(tok: str) -> float:
                    tok = tok.replace('$','')
                    if '\\pm' in tok:
                        tok = tok.split('\\pm')[0]
                    return float(tok)
                m_ab = _val(m_ab_tok)
                n_ser = _val(n_ser_tok)
                Re_kpc = _val(Re_tok)
            except Exception:
                continue
            rec = dict(id=obj_id, m_ab=m_ab, n_ser=n_ser, Re_kpc=Re_kpc)
            if (best is None) or (m_ab < best['m_ab']):
                best = rec
    if best is None:
        raise RuntimeError("No Sersic table rows parsed from TeX; cannot build stars profile")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', required=True)
    ap.add_argument('--cluster_dir', required=True)
    ap.add_argument('--Mtot_Msun', type=float, default=1.0e12)
    ap.add_argument('--icl_Mtot_Msun', type=float, default=5.0e11)
    ap.add_argument('--icl_a_kpc', type=float, default=100.0)
    args = ap.parse_args()

    tex = Path(args.tex)
    cdir = Path(args.cluster_dir)
    g_csv = cdir / 'gas_profile.csv'
    if not g_csv.exists():
        raise SystemExit(f"Missing gas_profile.csv in {cdir}; cannot build r-grid")
    r_grid = pd.read_csv(g_csv)['r_kpc'].astype(float).values
    r_grid = np.array(sorted(np.unique(r_grid)))

    brightest = parse_brightest_from_tex(tex)
    Re = float(brightest['Re_kpc'])
    n_ser = float(brightest['n_ser'])
    # Map Sersic Re to Hernquist a via Re ≈ 1.8153 a (exact for n=4)
    a_kpc = Re / 1.8153

    rho = hernquist_rho(r_grid, Mtot_Msun=float(args.Mtot_Msun), a_kpc=a_kpc)
    if float(args.icl_Mtot_Msun) > 0.0 and float(args.icl_a_kpc) > 0.0:
        rho += hernquist_rho(r_grid, Mtot_Msun=float(args.icl_Mtot_Msun), a_kpc=float(args.icl_a_kpc))

    out = pd.DataFrame({'r_kpc': r_grid, 'rho_star_Msun_per_kpc3': rho})
    out_path = cdir / 'stars_profile.csv'
    out.to_csv(out_path, index=False)

    print(f"[build_stars_from_halkola_tex] BCG candidate: id={brightest['id']} m_AB={brightest['m_ab']:.2f} n={n_ser:.2f} Re={Re:.2f} kpc")
    print(f"[build_stars_from_halkola_tex] Hernquist: a={a_kpc:.3f} kpc, Mtot={args.Mtot_Msun:.3e} Msun; ICL: M={args.icl_Mtot_Msun:.3e} Msun, a={args.icl_a_kpc:.1f} kpc")
    print(f"[build_stars_from_halkola_tex] wrote {out_path}")

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
rigor/scripts/inspect_cluster_inputs.py

Print quick sanity summaries for cluster input CSVs:
- rows, r_kpc min/max
- column min/max for key columns when present
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

CLUSTERS = ["ABELL_0426", "ABELL_1689"]
BASE = Path("data/clusters")

KEYS = {
    "gas_profile.csv": ["n_e_cm3"],
    "temp_profile.csv": ["kT_keV", "kT_err_keV"],
    "clump_profile.csv": ["C"],
    "stars_profile.csv": ["rho_star_Msun_per_kpc3"],
}

def q(v):
    return f"{v:.6g}" if isinstance(v, (int, float)) else str(v)

def main():
    for cl in CLUSTERS:
        print(f"--- {cl} stats ---")
        cdir = BASE/cl
        for name, cols in KEYS.items():
            path = cdir/name
            if not path.exists():
                print(f"{name}: MISSING")
                continue
            df = pd.read_csv(path)
            rmin = float(df['r_kpc'].min()) if 'r_kpc' in df.columns else float('nan')
            rmax = float(df['r_kpc'].max()) if 'r_kpc' in df.columns else float('nan')
            pieces = [f"rows={len(df)}", f"r_kpc[min,max]=[{q(rmin)},{q(rmax)}]"]
            for c in cols:
                if c in df.columns:
                    vmin = float(df[c].min())
                    vmax = float(df[c].max())
                    pieces.append(f"{c}[min,max]=[{q(vmin)},{q(vmax)}]")
            print(f"{name}: ", "; ".join(pieces))

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
rigor/scripts/gate_vs_surface_density.py

Estimate the correlation between each galaxy's LogTail R0 (gate radius) and the
radius where the baryonic surface density Sigma_b crosses a single global
threshold Sigma_*.

Inputs:
- --pred_csv: data/sparc_predictions_by_radius.csv (not strictly needed here)
- --rotmod_parquet: data/sparc_rotmod_ltg.parquet (must contain columns:
    galaxy, R_kpc, Vgas_kms, Vdisk_kms, Vbul_kms [optional])
- --r0_summary: JSON file mapping galaxy -> R0_kpc (or list of objects with
    keys ['galaxy','R0_kpc'] or ['galaxy','r0'] etc.)
- --sigma_star_grid: comma-list of Sigma_* in Msun/kpc^2, e.g., "50,100,150,200"
- --out_dir: where to write JSON + PNG

Outputs:
- out_dir/gate_sigma_correlation.json with best Sigma_*, Spearman rho, p
- out_dir/gate_sigma_scatter.png scatter plot using the best Sigma_*

Notes:
- We estimate Sigma_b(R) from rotmod curves using M(<R)=V^2 R / G and
  Sigma(R)=(1/(2*pi*R)) dM/dR; bulge is included via Vbul if present.
- Robust to missing bulge: disc+gas are often dominant in the outer disc where
  the crossing radius typically lies.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def sigma_from_M_of_R(R_kpc: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    R = np.asarray(R_kpc, float)
    M = np.asarray(M_enc, float)
    dM_dR = np.gradient(M, R)
    with np.errstate(divide='ignore', invalid='ignore'):
        Sigma = (dM_dR) / (2.0 * np.pi * np.maximum(R, 1e-12))
    Sigma = np.clip(Sigma, 0.0, None)
    return Sigma


def load_r0_summary(path: Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    out: dict[str, float] = {}
    if isinstance(data, dict):
        # Either a mapping gal -> r0 OR dict with a 'rows' list
        if all(isinstance(v, (int, float)) for v in data.values()):
            for k, v in data.items():
                out[str(k)] = float(v)
        elif 'rows' in data and isinstance(data['rows'], list):
            for row in data['rows']:
                g = str(row.get('galaxy') or row.get('Galaxy') or row.get('name') or row.get('Name'))
                r0 = row.get('R0_kpc') or row.get('r0_kpc') or row.get('r0') or row.get('R0')
                if g and r0 is not None:
                    out[g] = float(r0)
        else:
            # try top-level list-like mapping
            for k, v in data.items():
                try:
                    out[str(k)] = float(v.get('R0_kpc') or v.get('r0_kpc') or v.get('r0'))
                except Exception:
                    pass
    elif isinstance(data, list):
        for row in data:
            g = str(row.get('galaxy') or row.get('Galaxy') or row.get('name') or row.get('Name'))
            r0 = row.get('R0_kpc') or row.get('r0_kpc') or row.get('r0') or row.get('R0')
            if g and r0 is not None:
                out[g] = float(r0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--rotmod_parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--r0_summary', required=True)
    ap.add_argument('--sigma_star_grid', default='50,100,150,200')
    ap.add_argument('--out_dir', default='out/gate_sigma')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load R0 per galaxy
    r0_map = load_r0_summary(Path(args.r0_summary))
    if not r0_map:
        raise SystemExit(f'Could not parse any R0 entries from {args.r0_summary}')

    rot = pd.read_parquet(Path(args.rotmod_parquet))
    rot = rot.sort_values(['galaxy', 'R_kpc'])

    # Pre-scan Sigma_* values
    def parse_grid(s: str) -> list[float]:
        vals = []
        for tok in str(s).split(','):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
        return vals
    sigma_grid = parse_grid(args.sigma_star_grid)

    results = []
    best = None

    for sigma_star in sigma_grid:
        gate_radii = []
        r0_list = []
        for gname, dfg in rot.groupby('galaxy'):
            if gname not in r0_map:
                continue
            R = dfg['R_kpc'].to_numpy(float)
            Vgas = dfg['Vgas_kms'].to_numpy(float) if 'Vgas_kms' in dfg.columns else np.zeros_like(R)
            Vdisk= dfg['Vdisk_kms'].to_numpy(float) if 'Vdisk_kms' in dfg.columns else np.zeros_like(R)
            Vbul = dfg['Vbul_kms'].to_numpy(float) if 'Vbul_kms' in dfg.columns else np.zeros_like(R)
            # M(<R) by component
            Mgas  = (Vgas*Vgas) * R / G
            Mdisk = (Vdisk*Vdisk) * R / G
            Mbul  = (Vbul*Vbul)  * R / G
            # Sigma by derivative
            Sgas  = sigma_from_M_of_R(R, Mgas)
            Sdisk = sigma_from_M_of_R(R, Mdisk)
            Sbul  = sigma_from_M_of_R(R, Mbul)
            Ssum  = Sgas + Sdisk + Sbul
            # Find R where Sigma_b crosses sigma_star
            # Use monotone interpolation in outer half (robust to wiggles)
            idx = np.argsort(R)
            Rm = R[idx]; Sm = Ssum[idx]
            # Work from large to small R to find first crossing from below
            cross_R = np.nan
            for k in range(len(Rm)-2, 0, -1):
                if (Sm[k] <= sigma_star and Sm[k+1] >= sigma_star) or (Sm[k] >= sigma_star and Sm[k+1] <= sigma_star):
                    # linear interp
                    x0, x1 = Rm[k], Rm[k+1]
                    y0, y1 = Sm[k], Sm[k+1]
                    t = (sigma_star - y0) / ((y1 - y0) + 1e-12)
                    cross_R = float(x0 + t*(x1-x0))
                    break
            if not np.isnan(cross_R):
                gate_radii.append(cross_R)
                r0_list.append(r0_map[gname])
        if len(r0_list) >= 5:
            rho, p = spearmanr(r0_list, gate_radii)
            results.append({'sigma_star': sigma_star, 'rho': float(rho), 'p': float(p), 'n': len(r0_list)})
            if best is None or rho > best['rho']:
                best = results[-1]

    if best is None:
        raise SystemExit('Not enough overlapping galaxies to compute correlation.')

    (out_dir/'gate_sigma_correlation.json').write_text(json.dumps({'results': results, 'best': best}, indent=2))

    # Build scatter plot for best sigma_* (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        sigma_star = best['sigma_star']
        xs = []
        ys = []
        for gname, dfg in rot.groupby('galaxy'):
            if gname not in r0_map:
                continue
            R = dfg['R_kpc'].to_numpy(float)
            Vgas = dfg['Vgas_kms'].to_numpy(float) if 'Vgas_kms' in dfg.columns else np.zeros_like(R)
            Vdisk= dfg['Vdisk_kms'].to_numpy(float) if 'Vdisk_kms' in dfg.columns else np.zeros_like(R)
            Vbul = dfg['Vbul_kms'].to_numpy(float) if 'Vbul_kms' in dfg.columns else np.zeros_like(R)
            Mgas  = (Vgas*Vgas) * R / G
            Mdisk = (Vdisk*Vdisk) * R / G
            Mbul  = (Vbul*Vbul)  * R / G
            Ssum  = sigma_from_M_of_R(R, Mgas) + sigma_from_M_of_R(R, Mdisk) + sigma_from_M_of_R(R, Mbul)
            idx = np.argsort(R)
            Rm = R[idx]; Sm = Ssum[idx]
            cross_R = np.nan
            for k in range(len(Rm)-2, 0, -1):
                if (Sm[k] <= sigma_star and Sm[k+1] >= sigma_star) or (Sm[k] >= sigma_star and Sm[k+1] <= sigma_star):
                    x0, x1 = Rm[k], Rm[k+1]
                    y0, y1 = Sm[k], Sm[k+1]
                    t = (sigma_star - y0) / ((y1 - y0) + 1e-12)
                    cross_R = float(x0 + t*(x1-x0))
                    break
            if not np.isnan(cross_R):
                xs.append(r0_map[gname])
                ys.append(cross_R)
        plt.figure(figsize=(5,4))
        plt.scatter(xs, ys, s=10, alpha=0.7)
        plt.xlabel('R0 (kpc) from LogTail')
        plt.ylabel(f'R where Sigma_b=Sigma_* (Sigma_*={sigma_star:.0f} Msun/kpc^2)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir/'gate_sigma_scatter.png', dpi=140)
        plt.close()
    except Exception as e:
        # No plotting backend; ignore
        pass


if __name__ == '__main__':
    main()

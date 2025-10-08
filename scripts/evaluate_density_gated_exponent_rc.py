#!/usr/bin/env python3
"""
Evaluate a density-gated exponent gravity model on SPARC galaxy rotation curves.

Model summary (density-gated, NOT acceleration-gated):

- Gating function (sigma in [0,1]):
    sigma(ρ) = ( (ρ/ρ_c)^m ) / ( 1 + (ρ/ρ_c)^m )
  where ρ is (proxy) surface density Σ [Msun/pc^2], ρ_c is a threshold, and m controls sharpness.

- Acceleration law:
    a(r) = (G M_b(r) / r^2) * (r / r_t)^(1 - sigma(ρ(r)))

- Circular velocity:
    v^2(r) = a(r) * r = (G M_b(r) / r) * (r / r_t)^(1 - sigma(ρ(r)))
           = Vbar^2(r) * (r / r_t)^(1 - sigma(ρ(r)))

  where Vbar(r) is the baryonic circular velocity (from SPARC decomposition), using the identity
  Vbar^2 = G M_b(r)/r (spherical-equivalent enclosed-mass mapping).

- Transition length (global):
    r_t = sqrt( G M_b,total / a0 )

  with a0 in units of (km/s)^2/kpc. If a0 is provided in SI (m/s^2), we convert.

Outputs:
- Per-galaxy metrics (RMSE, median APE)
- Summary statistics across dataset
- Optional montage plot of RC overlays for a subset

Usage:
  python scripts/evaluate_density_gated_exponent_rc.py \
      --rho_c 10.0 --m 4.0 --a0_si 1.2e-10 --limit_galaxies 24 --montage_limit 16

Notes:
- This evaluation uses Σ_bar if available in SPARC parquet; otherwise falls back to an exponential proxy.
- This is a galaxy RC evaluation (not lensing). It is density-gated by construction (no RAR).
"""
from __future__ import annotations
import os, math, json, argparse, datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rigor.rigor.data import load_sparc

# Astrophysical constants
G_KPC_KMS2_MSUN = 4.300917270e-6  # G in kpc (km/s)^2 / Msun
KPC_IN_M = 3.085677581e19


def a0_si_to_kms2_per_kpc(a0_si: float) -> float:
    """Convert a0 from m/s^2 to (km/s)^2 per kpc."""
    # 1 (km/s)^2 per kpc = (1e6 m^2/s^2) / (KPC_IN_M m) = 1e6 / KPC_IN_M m/s^2
    one_kms2_per_kpc_in_si = 1e6 / KPC_IN_M  # m/s^2
    return a0_si / one_kms2_per_kpc_in_si


def sigma_gate(Sigma_pc2: np.ndarray, rho_c: float, m_shape: float) -> np.ndarray:
    """Compute sigma(ρ) with ρ ≈ Σ_bar (Msun/pc^2).
    sigma = (x^m) / (1 + x^m), where x = ρ/ρ_c.
    Clipped for numerical stability.
    """
    rho = np.maximum(Sigma_pc2, 1e-12)
    x = rho / max(rho_c, 1e-12)
    xm = np.power(x, np.maximum(m_shape, 1e-6))
    sigma = xm / (1.0 + xm)
    return np.clip(sigma, 0.0, 1.0)


def rt_from_Mb_a0(Mb_total_Msun: float, a0_kms2_per_kpc: float) -> float:
    """r_t = sqrt(G * M_b / a0) in kpc."""
    return float(np.sqrt(G_KPC_KMS2_MSUN * max(Mb_total_Msun, 1e-12) / max(a0_kms2_per_kpc, 1e-12)))


def predict_velocity_density_gated(R_kpc: np.ndarray,
                                   Vbar_kms: np.ndarray,
                                   Sigma_pc2: np.ndarray,
                                   Mb_total_Msun: float,
                                   rho_c: float,
                                   m_shape: float,
                                   a0_kms2_per_kpc: float) -> np.ndarray:
    """Compute v_model(r) using the density-gated exponent law.

    v^2 = Vbar^2 * (r / r_t)^(1 - sigma(Sigma(r)))
    """
    R = np.asarray(R_kpc)
    Vb2 = np.maximum(np.asarray(Vbar_kms), 0.0)**2
    sig = sigma_gate(Sigma_pc2=np.asarray(Sigma_pc2), rho_c=rho_c, m_shape=m_shape)
    rt_kpc = rt_from_Mb_a0(Mb_total_Msun=Mb_total_Msun, a0_kms2_per_kpc=a0_kms2_per_kpc)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.power(np.maximum(R / max(rt_kpc, 1e-12), 1e-12), (1.0 - sig))
    Vmod2 = Vb2 * factor
    return np.sqrt(np.maximum(Vmod2, 0.0))


def evaluate_dataset(rho_c: float, m_shape: float, a0_kms2_per_kpc: float,
                     limit_galaxies: int = -1, montage_limit: int = 16,
                     outdir: Path = Path("gravity_learn/experiments/eval/density_gated")) -> Dict:
    ds = load_sparc()
    items = []
    for g in ds.galaxies:
        R = np.asarray(g.R_kpc)
        Vobs = np.asarray(g.Vobs_kms)
        Vbar = np.asarray(g.Vbar_kms)
        if R.size < 6:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        R = R[mask]; Vobs = Vobs[mask]; Vbar = Vbar[mask]
        # Surface density proxy (Msun/pc^2)
        if g.Sigma_bar is not None:
            Sigma = np.asarray(g.Sigma_bar)[mask]
            # ensure strictly positive for logs
            Sigma = np.clip(np.nan_to_num(Sigma, nan=1.0, posinf=1.0, neginf=1.0), 1e-4, None)
        else:
            # fallback: exponential profile with scale ~ median R, normalized to 100 Msun/pc^2 at R=0
            Rd = np.maximum(np.nanmedian(R), 1.0)
            Sigma = 100.0 * np.exp(-R / Rd)
        # Total baryon mass if available
        Mb_total = None
        if g.Mbar_Msun is not None and np.isfinite(g.Mbar_Msun) and g.Mbar_Msun > 0:
            Mb_total = float(g.Mbar_Msun)
        else:
            # spherical-equivalent from outermost Vbar
            Mb_total = float((Vbar[-1]**2) * R[-1] / G_KPC_KMS2_MSUN)
        items.append({
            'name': g.name,
            'R': R, 'Vobs': Vobs, 'Vbar': Vbar,
            'Sigma': Sigma, 'Mb_total': Mb_total,
        })
    if limit_galaxies > 0:
        items = items[:limit_galaxies]

    # Evaluate
    results = []
    for it in items:
        Vmod = predict_velocity_density_gated(
            R_kpc=it['R'], Vbar_kms=it['Vbar'], Sigma_pc2=it['Sigma'],
            Mb_total_Msun=it['Mb_total'], rho_c=rho_c, m_shape=m_shape,
            a0_kms2_per_kpc=a0_kms2_per_kpc
        )
        Vobs = it['Vobs']
        rmse = float(np.sqrt(np.mean((Vmod - Vobs)**2)))
        mape = float(np.median(np.abs((Vmod - Vobs) / np.maximum(np.abs(Vobs), 1e-6))))
        results.append({'Galaxy': it['name'], 'rmse': rmse, 'median_ape': mape, 'n_points': int(len(Vobs))})
    df = pd.DataFrame(results)
    summary = {
        'rho_c_Msun_pc2': rho_c,
        'm_shape': m_shape,
        'a0_kms2_per_kpc': a0_kms2_per_kpc,
        'rmse_median': float(df['rmse'].median()) if not df.empty else float('nan'),
        'mape_median': float(df['median_ape'].median()) if not df.empty else float('nan'),
        'n_galaxies': int(len(df)),
    }

    # Ensure output dir
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(outdir / f'per_galaxy_metrics_density_gated_{ts}.csv', index=False)
    with open(outdir / f'summary_density_gated_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Montage
    N = min(montage_limit, len(items))
    ncols = 4
    nrows = int(np.ceil(N / ncols)) if N > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(N):
        it = items[i]
        R = it['R']; Vobs = it['Vobs']; Vbar = it['Vbar']; Sigma = it['Sigma']
        Vmod = predict_velocity_density_gated(
            R_kpc=R, Vbar_kms=Vbar, Sigma_pc2=Sigma,
            Mb_total_Msun=it['Mb_total'], rho_c=rho_c, m_shape=m_shape,
            a0_kms2_per_kpc=a0_kms2_per_kpc
        )
        ax = axes[i]
        ax.plot(R, Vobs, 'k.', label='Observed')
        ax.plot(R, Vbar, color='#1f77b4', alpha=0.8, label='Baryons')
        ax.plot(R, Vmod, color='#d62728', alpha=0.9, label='Density-gated')
        ax.set_title(it['name'])
        ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]'); ax.grid(True, alpha=0.3)
    for j in range(N, len(axes)):
        fig.delaxes(axes[j])
    if N > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f"Density-gated exponent (rho_c={rho_c} Msun/pc^2, m={m_shape}, a0={a0_kms2_per_kpc:.2f} (km/s)^2/kpc)")
    fig.savefig(outdir / f'montage_density_gated_{ts}.png', dpi=150)
    plt.close(fig)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rho_c', type=float, default=10.0, help='Density threshold [Msun/pc^2]')
    ap.add_argument('--m', type=float, default=4.0, help='Sharpness of transition')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--a0_si', type=float, default=1.2e-10, help='a0 in m/s^2 (converted)')
    group.add_argument('--a0_kpc', type=float, default=None, help='a0 in (km/s)^2/kpc (if provided, used directly)')
    ap.add_argument('--limit_galaxies', type=int, default=24)
    ap.add_argument('--montage_limit', type=int, default=16)
    ap.add_argument('--outdir', type=str, default=str(Path('gravity_learn/experiments/eval/density_gated')))
    args = ap.parse_args()

    if args.a0_kpc is not None and args.a0_kpc > 0:
        a0_kpc = float(args.a0_kpc)
    else:
        a0_kpc = float(a0_si_to_kms2_per_kpc(args.a0_si))

    outdir = Path(args.outdir)
    summary = evaluate_dataset(rho_c=args.rho_c, m_shape=args.m, a0_kms2_per_kpc=a0_kpc,
                               limit_galaxies=args.limit_galaxies, montage_limit=args.montage_limit,
                               outdir=outdir)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

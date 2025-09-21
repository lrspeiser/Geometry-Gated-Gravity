#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We import baseline formulas (MOND, Burkert) from the local rigor package
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add .../rigor to sys.path
from rigor.baselines import mond_simple, nfw_velocity_kms


def percent_closeness(v_obs: np.ndarray, v_pred: np.ndarray) -> np.ndarray:
    v_obs = np.asarray(v_obs, dtype=float)
    v_pred = np.asarray(v_pred, dtype=float)
    denom = np.maximum(np.abs(v_obs), 1e-9)
    return 100.0 * (1.0 - np.abs(v_pred - v_obs) / denom)


def load_outliers(outliers_json: Path) -> set:
    if outliers_json is None or not outliers_json.exists():
        return set()
    data = json.loads(Path(outliers_json).read_text(encoding='utf-8'))
    return set(d['galaxy'] for d in data.get('excluded_outliers', []))


def compute_gr_closeness(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    Vobs = pd.to_numeric(df.loc[mask, 'Vobs_kms'], errors='coerce').to_numpy()
    Vbar = pd.to_numeric(df.loc[mask, 'Vbar_kms'], errors='coerce').to_numpy()
    cl = percent_closeness(Vobs, Vbar)
    return cl[np.isfinite(cl)]


def compute_shell_closeness(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    Vobs = pd.to_numeric(df.loc[mask, 'Vobs_kms'], errors='coerce').to_numpy()
    Vpred = pd.to_numeric(df.loc[mask, 'Vpred_kms'], errors='coerce').to_numpy()
    cl = percent_closeness(Vobs, Vpred)
    return cl[np.isfinite(cl)]


def compute_mond_closeness(df: pd.DataFrame, mask: np.ndarray, a0_hat: float) -> np.ndarray:
    R = pd.to_numeric(df.loc[mask, 'R_kpc'], errors='coerce').to_numpy()
    Vbar = pd.to_numeric(df.loc[mask, 'Vbar_kms'], errors='coerce').to_numpy()
    Vobs = pd.to_numeric(df.loc[mask, 'Vobs_kms'], errors='coerce').to_numpy()
    # mond_simple expects arrays (km/s, kpc, a0 in m/s^2); returns km/s
    Vmond = mond_simple(Vbar, R, a0=a0_hat)
    cl = percent_closeness(Vobs, Vmond)
    cl = cl[np.isfinite(cl)]
    return cl


def _fit_nfw_one_gal(gdf: pd.DataFrame,
                      rs_grid: np.ndarray,
                      rhos_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted velocities for best-fit NFW halo for one galaxy group.
    If too few points (<6) or invalid data, returns empty arrays.
    """
    R = pd.to_numeric(gdf['R_kpc'], errors='coerce').to_numpy()
    Vobs = pd.to_numeric(gdf['Vobs_kms'], errors='coerce').to_numpy()
    Vbar = pd.to_numeric(gdf['Vbar_kms'], errors='coerce').to_numpy()

    ok = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
    R = R[ok]; Vobs = Vobs[ok]; Vbar = Vbar[ok]
    if R.size < 6:
        return np.array([]), np.array([])

    best = None
    best_sse = np.inf
    # Coarse grid search (unweighted SSE)
    for rs in rs_grid:
        for rhos in rhos_grid:
            Vh = np.array([nfw_velocity_kms(r, rhos, rs) for r in R], dtype=float)
            Vpred = np.sqrt(np.maximum(Vbar**2 + Vh**2, 0.0))
            sse = float(np.sum((Vobs - Vpred)**2))
            if sse < best_sse:
                best_sse = sse
                best = (rhos, rs)
    if best is None:
        return np.array([]), np.array([])
    rhos_star, rs_star = best
    Vh_best = np.array([nfw_velocity_kms(r, rhos_star, rs_star) for r in R], dtype=float)
    Vpred_best = np.sqrt(np.maximum(Vbar**2 + Vh_best**2, 0.0))
    return Vobs, Vpred_best


def compute_nfw_closeness(df: pd.DataFrame, mask: np.ndarray, limit: int, seed: int) -> Tuple[np.ndarray, int]:
    rng = np.random.RandomState(seed)
    # Subset to masked rows
    sdf = df.loc[mask, ['galaxy','R_kpc','Vobs_kms','Vbar_kms']].copy()
    groups = list(sdf.groupby('galaxy'))
    if limit and limit > 0 and limit < len(groups):
        idx = rng.choice(len(groups), size=limit, replace=False)
        groups = [groups[int(i)] for i in idx]
    # Grids (coarse to keep runtime reasonable): r_s in [0.5, 50] kpc; rho_s in [1e6, 1e10] Msun/kpc^3
    rs_grid   = np.geomspace(0.5, 50.0, 14)
    rhos_grid = np.geomspace(1e6, 1e10, 14)

    all_cl = []
    n_gal = 0
    for gal, gdf in groups:
        Vobs, Vpred = _fit_nfw_one_gal(gdf, rs_grid, rhos_grid)
        if Vobs.size == 0:
            continue
        cl = percent_closeness(Vobs, Vpred)
        cl = cl[np.isfinite(cl)]
        if cl.size == 0:
            continue
        all_cl.append(cl)
        n_gal += 1
    if not all_cl:
        return np.array([]), 0
    return np.concatenate(all_cl), n_gal


def summarize(name: str, values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"N": 0, "mean": float('nan'), "median": float('nan')}
    return {
        "N": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def make_plot(stats_all: Dict[str, Dict[str, float]],
              stats_no: Dict[str, Dict[str, float]],
              out_png: Path,
              title: str) -> None:
    methods = ["Shell", "GR", "MOND", "NFW"]
    med_all = [stats_all[m]["median"] for m in methods]
    med_no  = [stats_no[m]["median"] for m in methods]
    N_all   = [stats_all[m]["N"] for m in methods]
    N_no    = [stats_no[m]["N"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    x = np.arange(len(methods))

    for ax, vals, Ns, subtitle in [
        (axes[0], med_all, N_all, 'All outer points'),
        (axes[1], med_no,  N_no,  'Excluding outliers'),
    ]:
        ax.bar(x, vals, color=['C0','C2','C3','C1'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Median % closeness')
        ax.set_title(subtitle)
        for xi, v, n in zip(x, vals, Ns):
            ax.text(xi, min(98, v + 2.0), f"{v:.1f}%\nN={n}", ha='center', va='bottom', fontsize=9)

    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Plot accuracy comparison (Shell vs GR vs MOND vs Burkert).')
    ap.add_argument('--predictions', default=str(Path('out/analysis/type_breakdown/predictions_by_radius.csv')),
                    help='Per-radius predictions CSV with columns galaxy,R_kpc,is_outer,Vobs_kms,Vbar_kms,Vpred_kms,...')
    ap.add_argument('--outliers', default=str(Path('out/analysis/type_breakdown/closeness_summary.json')),
                    help='closeness_summary.json to get excluded_outliers list (optional).')
    ap.add_argument('--baselines-json', default=str(Path('out/baselines_summary.json')),
                    help='If present, use its MOND a0_hat; otherwise fallback to 1.2e-10.')
    ap.add_argument('--out-dir', default=str(Path('out/analysis/type_breakdown')))
    ap.add_argument('--burkert-limit', type=int, default=40,
                    help='Max galaxies to fit for Burkert (to limit runtime). 0 => all.')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out_dir   = Path(args.out_dir)
    outliers_path = Path(args.outliers)
    baselines_path= Path(args.baselines_json)

    if not pred_path.exists():
        raise FileNotFoundError(f'predictions_by_radius not found: {pred_path}')

    df = pd.read_csv(pred_path)
    # outer mask (robust to variations)
    is_outer = df['is_outer'].astype(str).str.lower().isin(['true','1','t','yes','y'])

    # Outlier filtering by galaxy names
    outliers = load_outliers(outliers_path)
    no_out_mask = is_outer & ~df['galaxy'].astype(str).isin(outliers)

    # MOND a0
    a0_hat = 1.2e-10
    if baselines_path.exists():
        try:
            base = json.loads(baselines_path.read_text(encoding='utf-8'))
            a0_hat = float(base.get('MOND_simple', {}).get('a0_hat', a0_hat))
        except Exception:
            pass

    # Compute closeness arrays
    cl_shell_all = compute_shell_closeness(df, is_outer)
    cl_gr_all    = compute_gr_closeness(df, is_outer)
    cl_mond_all  = compute_mond_closeness(df, is_outer, a0_hat)
    cl_nfw_all, n_nfw_all = compute_nfw_closeness(df, is_outer, limit=args.burkert_limit, seed=args.seed)

    cl_shell_no  = compute_shell_closeness(df, no_out_mask)
    cl_gr_no     = compute_gr_closeness(df, no_out_mask)
    cl_mond_no   = compute_mond_closeness(df, no_out_mask, a0_hat)
    cl_nfw_no, n_nfw_no = compute_nfw_closeness(df, no_out_mask, limit=args.burkert_limit, seed=args.seed)

    # Summaries
    stats_all = {
        'Shell':  summarize('Shell',  cl_shell_all),
        'GR':     summarize('GR',     cl_gr_all),
        'MOND':   summarize('MOND',   cl_mond_all),
        'NFW':   {**summarize('NFW', cl_nfw_all), 'N_gal_fit': int(n_nfw_all)},
    }
    stats_no = {
        'Shell':  summarize('Shell',  cl_shell_no),
        'GR':     summarize('GR',     cl_gr_no),
        'MOND':   summarize('MOND',   cl_mond_no),
        'NFW':   {**summarize('NFW', cl_nfw_no), 'N_gal_fit': int(n_nfw_no)},
    }

    # Persist JSON
    out_json = out_dir / 'accuracy_comparison.json'
    out = {
        'a0_hat': a0_hat,
        'burkert_limit': args.burkert_limit,
        'all_outer': stats_all,
        'exclude_outliers': stats_no,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding='utf-8')

    # Plot
    out_png = out_dir / 'accuracy_comparison.png'
    title = 'Accuracy comparison (outer region): Shell vs GR vs MOND vs NFW'
    make_plot(stats_all, stats_no, out_png, title)

    print(f'Wrote {out_png}')
    print(f'Wrote {out_json}')


if __name__ == '__main__':
    main()
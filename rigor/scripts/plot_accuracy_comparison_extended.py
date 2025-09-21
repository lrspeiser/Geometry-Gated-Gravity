#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigor.baselines import mond_simple, nfw_velocity_kms

C_KM_S = 299_792.458
C2 = C_KM_S**2
KM_PER_KPC = 3.085677581e16  # km per kpc
OMEGA_SF = 1.0 / KM_PER_KPC  # (km/s)/kpc -> s^-1


def percent_closeness(v_obs: np.ndarray, v_pred: np.ndarray) -> np.ndarray:
    v_obs = np.asarray(v_obs, dtype=float)
    v_pred = np.asarray(v_pred, dtype=float)
    denom = np.maximum(np.abs(v_obs), 1e-9)
    return 100.0 * (1.0 - np.abs(v_pred - v_obs) / denom)


def summarize(values: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"N": 0, "mean": float('nan'), "median": float('nan')}
    return {"N": int(vals.size), "mean": float(np.mean(vals)), "median": float(np.median(vals))}


# Models to evaluate

def xi_anti_yukawa(r_kpc: np.ndarray, alpha: float, lam_kpc: float) -> np.ndarray:
    r = np.maximum(np.asarray(r_kpc, dtype=float), 0.0)
    lam = max(lam_kpc, 1e-6)
    x = r / lam
    return 1.0 + alpha * (1.0 - (1.0 + x) * np.exp(-x))


def v_linear_plus_newton(Vbar_kms: np.ndarray, r_kpc: np.ndarray, gamma_per_kpc: float) -> np.ndarray:
    Vbar2 = np.maximum(np.asarray(Vbar_kms, dtype=float), 0.0)**2
    r = np.maximum(np.asarray(r_kpc, dtype=float), 0.0)
    V2 = Vbar2 + 0.5 * float(gamma_per_kpc) * C2 * r
    return np.sqrt(np.maximum(V2, 0.0))


def xi_density_gate(r_kpc: np.ndarray, boundary_kpc: np.ndarray, eps: float, s_c: float, p: float) -> np.ndarray:
    r = np.maximum(np.asarray(r_kpc, dtype=float), 0.0)
    b = np.asarray(boundary_kpc, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = r / b
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)  # invalid where boundary missing
    sp = np.power(s, p, where=np.isfinite(s))
    scp = s_c**p
    frac = sp / (sp + scp)
    xi = 1.0 + eps * frac
    return xi


def fit_global_anti_yukawa(Vbar, Vobs, R, grid_alpha, grid_lambda) -> Tuple[float, float, np.ndarray]:
    best = None; best_med = -np.inf; best_pred=None
    for a in grid_alpha:
        for L in grid_lambda:
            xi = xi_anti_yukawa(R, a, L)
            Vpred = Vbar * np.sqrt(np.maximum(xi, 0.0))
            cl = percent_closeness(Vobs, Vpred)
            med = float(np.nanmedian(cl))
            if med > best_med:
                best_med = med; best = (a, L); best_pred = Vpred
    return best[0], best[1], best_pred


def fit_global_linear_newton(Vbar, Vobs, R, grid_gamma) -> Tuple[float, np.ndarray]:
    best = None; best_med = -np.inf; best_pred=None
    for g in grid_gamma:
        Vpred = v_linear_plus_newton(Vbar, R, g)
        cl = percent_closeness(Vobs, Vpred)
        med = float(np.nanmedian(cl))
        if med > best_med:
            best_med = med; best = g; best_pred = Vpred
    return float(best), best_pred


def fit_global_density_gate(Vbar, Vobs, R, B, grid_eps, grid_sc, grid_p) -> Tuple[Tuple[float,float,float], np.ndarray]:
    best = None; best_med = -np.inf; best_pred=None
    for e in grid_eps:
        for sc in grid_sc:
            for p in grid_p:
                xi = xi_density_gate(R, B, e, sc, p)
                Vpred = Vbar * np.sqrt(np.maximum(xi, 0.0))
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best = (e, sc, p); best_pred = Vpred
    return (float(best[0]), float(best[1]), float(best[2])), best_pred


def compute_nfw_closeness(df: pd.DataFrame, is_outer: np.ndarray, limit: int, seed: int) -> Tuple[np.ndarray, int]:
    rng = np.random.RandomState(seed)
    sdf = df.loc[is_outer, ['galaxy','R_kpc','Vobs_kms','Vbar_kms']].copy()
    groups = list(sdf.groupby('galaxy'))
    if limit and limit > 0 and limit < len(groups):
        idx = rng.choice(len(groups), size=limit, replace=False)
        groups = [groups[int(i)] for i in idx]
    rs_grid   = np.geomspace(0.5, 50.0, 14)
    rhos_grid = np.geomspace(1e6, 1e10, 14)
    all_cl = []; n_gal=0
    for gal, gdf in groups:
        R = pd.to_numeric(gdf['R_kpc'], errors='coerce').to_numpy()
        Vobs = pd.to_numeric(gdf['Vobs_kms'], errors='coerce').to_numpy()
        Vbar = pd.to_numeric(gdf['Vbar_kms'], errors='coerce').to_numpy()
        ok = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        R=R[ok]; Vobs=Vobs[ok]; Vbar=Vbar[ok]
        if R.size < 6:
            continue
        best=None; best_sse=np.inf
        for rs in rs_grid:
            for rhos in rhos_grid:
                Vh = np.array([nfw_velocity_kms(r, rhos, rs) for r in R], dtype=float)
                Vpred = np.sqrt(np.maximum(Vbar**2 + Vh**2, 0.0))
                sse = float(np.sum((Vobs - Vpred)**2))
                if sse < best_sse:
                    best_sse = sse; best = (rhos, rs)
        if best is None:
            continue
        rhos_star, rs_star = best
        Vh_best = np.array([nfw_velocity_kms(r, rhos_star, rs_star) for r in R], dtype=float)
        Vpred_best = np.sqrt(np.maximum(Vbar**2 + Vh_best**2, 0.0))
        cl = percent_closeness(Vobs, Vpred_best)
        cl = cl[np.isfinite(cl)]
        if cl.size == 0:
            continue
        all_cl.append(cl)
        n_gal += 1
    if not all_cl:
        return np.array([]), 0
    return np.concatenate(all_cl), n_gal


def omega_from_vr(v_kms: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    v = np.asarray(v_kms, dtype=float)
    r = np.asarray(r_kpc, dtype=float)
    return (v / np.maximum(r, 1e-9)) * OMEGA_SF  # s^-1

def make_plot(stats: Dict[str, Dict[str, float]], out_png: Path, title: str) -> None:
    methods = list(stats.keys())
    med = [stats[m]['median'] for m in methods]
    N   = [stats[m]['N'] for m in methods]
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    x = np.arange(len(methods))
    # Generate a color cycle long enough
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(methods), 8)))
    ax.bar(x, med, color=colors[:len(methods)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Median % closeness (outer)')
    for xi, v, n in zip(x, med, N):
        ax.text(xi, min(98, v+2.0), f"{v:.1f}%\nN={n}", ha='center', va='bottom', fontsize=9)
    ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def solve_iterative_v(r_kpc: np.ndarray,
                       vbar_kms: np.ndarray,
                       gate_fn,
                       relax: float = 0.4,
                       tol: float = 1e-6,
                       iters: int = 128) -> np.ndarray:
    vbar2 = np.maximum(np.asarray(vbar_kms, dtype=float), 0.0)**2
    r = np.asarray(r_kpc, dtype=float)
    v2 = np.array(vbar2, copy=True)
    for _ in range(iters):
        v = np.sqrt(np.maximum(v2, 0.0))
        mu = np.asarray(gate_fn(v, r, vbar2), dtype=float)
        v2_new = vbar2 * np.maximum(mu, 0.0)
        v2 = (1.0 - relax) * v2 + relax * v2_new
        if np.all(np.abs(v2_new - v2) / np.maximum(v2, 1e-12) < tol):
            break
    return np.sqrt(np.maximum(v2, 0.0))

def mu_rg_running(R_kpc: np.ndarray, nu: float, r0_kpc: float, rstar_kpc: float) -> np.ndarray:
    R = np.maximum(np.asarray(R_kpc, dtype=float), 0.0)
    r0 = max(float(r0_kpc), 1e-6)
    rstar = max(float(rstar_kpc), r0 + 1e-6)
    denom = np.log1p(rstar/r0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 1.0 + float(nu) * np.log1p(R/r0) / max(denom, 1e-9)
    return mu


def fit_global_mu_rg(Vbar, Vobs, R, grid_r0, grid_rstar, grid_nu) -> Tuple[Tuple[float,float,float], np.ndarray]:
    vbar2 = np.maximum(Vbar, 0.0)**2
    best=None; best_med=-np.inf; best_pred=None
    for r0 in grid_r0:
        for rst in grid_rstar:
            for nu in grid_nu:
                mu = mu_rg_running(R, nu, r0, rst)
                Vpred = np.sqrt(np.maximum(vbar2 * mu, 0.0))
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(nu), float(r0), float(rst)); best_pred=Vpred
    return best, best_pred


def fit_global_mu_phi(Vbar, Vobs, R, grid_eps, grid_vc, grid_p) -> Tuple[Tuple[float,float,float], np.ndarray]:
    vbar2 = np.maximum(Vbar, 0.0)**2
    abs_phiN = vbar2  # proxy for |Phi_N| in km^2/s^2
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for vc in grid_vc:
            Phi_c = (float(vc)**2)
            for p in grid_p:
                mu = 1.0 + float(eps) / (1.0 + np.power(np.maximum(abs_phiN, 0.0)/max(Phi_c, 1e-9), float(p)))
                Vpred = np.sqrt(np.maximum(vbar2 * mu, 0.0))
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(eps), float(vc), float(p)); best_pred=Vpred
    return best, best_pred


def fit_global_slip(Vbar, Vobs, R, grid_eps, grid_r0, grid_delta) -> Tuple[Tuple[float,float,float], np.ndarray]:
    vbar2 = np.maximum(Vbar, 0.0)**2
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for r0 in grid_r0:
            for dlt in grid_delta:
                S = 0.5*(1.0 + np.tanh((R - float(r0))/max(float(dlt), 1e-6)))
                mu = 1.0 + float(eps) * S
                Vpred = np.sqrt(np.maximum(vbar2 * mu, 0.0))
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(eps), float(r0), float(dlt)); best_pred=Vpred
    return best, best_pred


def fit_global_log_tail(Vbar, Vobs, R, grid_v0, grid_rc, grid_r0, grid_delta) -> Tuple[Tuple[float,float,float,float], np.ndarray]:
    vbar2 = np.maximum(Vbar, 0.0)**2
    best=None; best_med=-np.inf; best_pred=None
    for v0 in grid_v0:
        v0sq = float(v0)**2
        for rc in grid_rc:
            rcv = max(float(rc), 1e-6)
            for r0 in grid_r0:
                for dlt in grid_delta:
                    S = 0.5*(1.0 + np.tanh((R - float(r0))/max(float(dlt), 1e-6)))
                    tail = v0sq * (R / (R + rcv)) * S
                    V2 = vbar2 + np.maximum(tail, 0.0)
                    Vpred = np.sqrt(np.maximum(V2, 0.0))
                    cl = percent_closeness(Vobs, Vpred)
                    med = float(np.nanmedian(cl))
                    if med > best_med:
                        best_med = med; best=(float(v0), float(rc), float(r0), float(dlt)); best_pred=Vpred
    return best, best_pred

# --- Regenerative feedback models ---

def fit_global_omega_gate(Vbar, Vobs, R, grid_eps, grid_p, grid_omegac) -> Tuple[Tuple[float,float,float], np.ndarray]:
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for p in grid_p:
            for oc in grid_omegac:
                def gate_fn(v, r, vbar2):
                    omega = omega_from_vr(v, r)
                    return 1.0 + float(eps) / (1.0 + np.power(omega/float(oc), float(p)))
                Vpred = solve_iterative_v(R, Vbar, gate_fn)
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(eps), float(p), float(oc)); best_pred=Vpred
    return best, best_pred

def fit_global_j_gate(Vbar, Vobs, R, grid_eps, grid_q, grid_j0) -> Tuple[Tuple[float,float,float], np.ndarray]:
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for q in grid_q:
            for j0 in grid_j0:
                j0v = float(j0)
                def gate_fn(v, r, vbar2):
                    x = (r*np.maximum(v,0.0))/j0v
                    t = np.power(x, float(q))
                    return 1.0 + float(eps) * (t/(1.0 + t))
                Vpred = solve_iterative_v(R, Vbar, gate_fn)
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(eps), float(q), float(j0)); best_pred=Vpred
    return best, best_pred

def fit_global_p_gate(Vbar, Vobs, R, grid_eps, grid_q, grid_P0) -> Tuple[Tuple[float,float,float], np.ndarray]:
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for q in grid_q:
            for P0 in grid_P0:
                P0v = float(P0)
                def gate_fn(v, r, vbar2):
                    x = (np.maximum(v,0.0)**3)/(np.maximum(r,1e-9)*P0v)
                    t = np.power(x, float(q))
                    return 1.0 + float(eps) * (t/(1.0 + t))
                Vpred = solve_iterative_v(R, Vbar, gate_fn)
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(eps), float(q), float(P0)); best_pred=Vpred
    return best, best_pred

def compute_sbar_by_galaxy(df_outer: pd.DataFrame) -> pd.Series:
    # S_bar = - d ln(Vbar/r) / d ln r per galaxy
    svals = np.full(len(df_outer), np.nan, dtype=float)
    idx = df_outer.index.to_numpy()
    for gal, g in df_outer.groupby('galaxy'):
        g = g.sort_values('R_kpc')
        r = pd.to_numeric(g['R_kpc'], errors='coerce').to_numpy()
        vbar = pd.to_numeric(g['Vbar_kms'], errors='coerce').to_numpy()
        y = np.log(np.maximum(vbar/np.maximum(r,1e-9), 1e-12))
        x = np.log(np.maximum(r, 1e-9))
        dy = np.gradient(y, x)
        Sbar = -dy
        svals[np.isin(idx, g.index.to_numpy())] = Sbar
    return pd.Series(svals, index=df_outer.index)

def fit_global_s_gate(df_outer: pd.DataFrame, Vobs, Vbar, R, grid_eps, grid_q, grid_S0) -> Tuple[Tuple[float,float,float], np.ndarray]:
    Sbar = compute_sbar_by_galaxy(df_outer)
    Sbar_arr = Sbar.to_numpy()
    vbar2 = np.maximum(Vbar, 0.0)**2
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for q in grid_q:
            for S0 in grid_S0:
                t = np.power(np.maximum(Sbar_arr, 0.0)/float(S0), float(q))
                mu = 1.0 + float(eps) * (t/(1.0 + t))
                Vpred = np.sqrt(np.maximum(vbar2 * mu, 0.0))
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(eps), float(q), float(S0)); best_pred=Vpred
    return best, best_pred

def fit_global_d_gate(Vbar, Vobs, R, grid_eps, grid_p, grid_omegac, grid_q) -> Tuple[Tuple[float,float,float,float], np.ndarray]:
    best=None; best_med=-np.inf; best_pred=None
    for eps in grid_eps:
        for p in grid_p:
            for oc in grid_omegac:
                for q in grid_q:
                    def gate_fn(v, r, vbar2):
                        omega = omega_from_vr(v, r)
                        g = 1.0 / (1.0 + np.power(omega/float(oc), float(p)))
                        D = np.maximum(v,0.0)**2 / np.maximum(vbar2, 1e-9)
                        h = np.power(D, float(q)) / (1.0 + np.power(D, float(q)))
                        return 1.0 + float(eps) * g * h
                    Vpred = solve_iterative_v(R, Vbar, gate_fn)
                    cl = percent_closeness(Vobs, Vpred)
                    med = float(np.nanmedian(cl))
                    if med > best_med:
                        best_med = med; best=(float(eps), float(p), float(oc), float(q)); best_pred=Vpred
    return best, best_pred

def fit_global_gm_gate(Vbar, Vobs, R, grid_beta, grid_p, grid_omegac) -> Tuple[Tuple[float,float,float], np.ndarray]:
    # Toy gravito-magnetic-like augmentation; iterate on v
    vbar2 = np.maximum(Vbar, 0.0)**2
    best=None; best_med=-np.inf; best_pred=None
    for beta in grid_beta:
        for p in grid_p:
            for oc in grid_omegac:
                def gate_fn(v, r, vbar2):
                    omega = omega_from_vr(v, r)
                    g = 1.0 / (1.0 + np.power(omega/float(oc), float(p)))
                    # v^2 = vbar^2 + beta * (2/c^2) * g * omega * v * vbar^2
                    return 1.0 + (float(beta) * 2.0 / C2) * g * omega * np.maximum(v,0.0)
                Vpred = solve_iterative_v(R, Vbar, gate_fn, relax=0.3)
                cl = percent_closeness(Vobs, Vpred)
                med = float(np.nanmedian(cl))
                if med > best_med:
                    best_med = med; best=(float(beta), float(p), float(oc)); best_pred=Vpred
    return best, best_pred
    vbar2 = np.maximum(Vbar, 0.0)**2
    best=None; best_med=-np.inf; best_pred=None
    for v0 in grid_v0:
        v0sq = float(v0)**2
        for rc in grid_rc:
            rcv = max(float(rc), 1e-6)
            for r0 in grid_r0:
                for dlt in grid_delta:
                    S = 0.5*(1.0 + np.tanh((R - float(r0))/max(float(dlt), 1e-6)))
                    tail = v0sq * (R / (R + rcv)) * S
                    V2 = vbar2 + np.maximum(tail, 0.0)
                    Vpred = np.sqrt(np.maximum(V2, 0.0))
                    cl = percent_closeness(Vobs, Vpred)
                    med = float(np.nanmedian(cl))
                    if med > best_med:
                        best_med = med; best=(float(v0), float(rc), float(r0), float(dlt)); best_pred=Vpred
    return best, best_pred


def main():
    ap = argparse.ArgumentParser(description='Extended accuracy comparison (Shell, GR, MOND, NFW, plus Anti-Yukawa, Linear+Newton, Density-gated, RG-running G, Potential-gated, Slip, Log-tail).')
    ap.add_argument('--predictions', default=str(Path('out/analysis/type_breakdown/predictions_by_radius.csv')))
    ap.add_argument('--baselines-json', default=str(Path('out/baselines_summary.json')))
    ap.add_argument('--out-dir', default=str(Path('out/analysis/type_breakdown')))
    ap.add_argument('--nfw-limit', type=int, default=40, help='Per-galaxy NFW fit limit (0=all)')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    # Load data
    df = pd.read_csv(pred_path)
    is_outer = df['is_outer'].astype(str).str.lower().isin(['true','1','t','yes','y'])
    Vobs = pd.to_numeric(df.loc[is_outer,'Vobs_kms'], errors='coerce').to_numpy()
    Vbar = pd.to_numeric(df.loc[is_outer,'Vbar_kms'], errors='coerce').to_numpy()
    R    = pd.to_numeric(df.loc[is_outer,'R_kpc'], errors='coerce').to_numpy()
    B    = pd.to_numeric(df.loc[is_outer,'boundary_kpc'], errors='coerce').to_numpy()

    # Shell (already computed): use Vpred_kms column
    Vshell = pd.to_numeric(df.loc[is_outer,'Vpred_kms'], errors='coerce').to_numpy()
    cl_shell = percent_closeness(Vobs, Vshell)

    # GR
    cl_gr = percent_closeness(Vobs, Vbar)

    # MOND
    a0_hat = 1.2e-10
    base_path = Path(args.baselines_json)
    if base_path.exists():
        try:
            base = json.loads(base_path.read_text(encoding='utf-8'))
            a0_hat = float(base.get('MOND_simple',{}).get('a0_hat', a0_hat))
        except Exception:
            pass
    Vmond = mond_simple(Vbar, R, a0=a0_hat)
    cl_mond = percent_closeness(Vobs, Vmond)

    # NFW per-galaxy
    cl_nfw, n_nfw = compute_nfw_closeness(df, is_outer, limit=args.nfw_limit, seed=args.seed)

    # Additional models (global parameter search)
    # Anti-Yukawa
    grid_alpha = [0.2, 0.5, 1.0, 1.5]
    grid_lambda = [3.0, 5.0, 8.0, 12.0, 16.0]
    ay_alpha, ay_lambda, Vay = fit_global_anti_yukawa(Vbar, Vobs, R, grid_alpha, grid_lambda)
    cl_ay = percent_closeness(Vobs, Vay)

    # Linear + Newton
    grid_gamma = [1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10, 1e-9]
    ln_gamma, Vln = fit_global_linear_newton(Vbar, Vobs, R, grid_gamma)
    cl_ln = percent_closeness(Vobs, Vln)

    # Density-gated metric (using normalized radius r/boundary as proxy)
    grid_eps = [0.2, 0.5, 1.0, 1.5]
    grid_sc  = [0.5, 1.0]
    grid_p   = [1.0, 2.0, 3.0]
    (dg_eps, dg_sc, dg_p), Vdg = fit_global_density_gate(Vbar, Vobs, R, B, grid_eps, grid_sc, grid_p)
    cl_dg = percent_closeness(Vobs, Vdg)

    # RG-running G
    rg_grid_r0    = [1.0, 2.0, 4.0, 8.0]
    rg_grid_rstar = [30.0, 60.0, 100.0]
    rg_grid_nu    = list(np.linspace(0.0, 0.25, 9))
    (rg_nu, rg_r0, rg_rstar), Vrg = fit_global_mu_rg(Vbar, Vobs, R, rg_grid_r0, rg_grid_rstar, rg_grid_nu)
    cl_rg = percent_closeness(Vobs, Vrg)

    # Potential-gated mu_phi
    phi_grid_eps = [0.2, 0.5, 1.0, 1.5]
    phi_grid_vc  = [80.0, 120.0, 160.0, 220.0]
    phi_grid_p   = [1.0, 2.0, 3.0, 4.0]
    (phi_eps, phi_vc, phi_p), Vphi = fit_global_mu_phi(Vbar, Vobs, R, phi_grid_eps, phi_grid_vc, phi_grid_p)
    cl_phi = percent_closeness(Vobs, Vphi)

    # Slip-gated mu(r)
    slip_grid_eps   = [0.2, 0.5, 1.0, 1.5]
    slip_grid_r0    = [1.0, 2.0, 4.0, 8.0]
    slip_grid_delta = [1.0, 2.0, 3.0]
    (slip_eps, slip_r0, slip_delta), Vslip = fit_global_slip(Vbar, Vobs, R, slip_grid_eps, slip_grid_r0, slip_grid_delta)
    cl_slip = percent_closeness(Vobs, Vslip)

    # Logarithmic tail with gate
    log_grid_v0   = [50.0, 80.0, 120.0, 160.0, 200.0]
    log_grid_rc   = [1.0, 3.0, 5.0, 10.0]
    log_grid_r0   = [1.0, 3.0, 5.0]
    log_grid_dlt  = [1.0, 2.0, 3.0]
    (log_v0, log_rc, log_r0, log_dlt), Vlog = fit_global_log_tail(Vbar, Vobs, R, log_grid_v0, log_grid_rc, log_grid_r0, log_grid_dlt)
    cl_log = percent_closeness(Vobs, Vlog)

    # Omega gate
    omg_grid_eps = [0.3, 0.6, 0.9, 1.2]
    omg_grid_p   = [1.0, 2.0, 3.0]
    omg_grid_oc  = [3e-13, 1e-12]
    (omg_eps, omg_p, omg_oc), Vomg = fit_global_omega_gate(Vbar, Vobs, R, omg_grid_eps, omg_grid_p, omg_grid_oc)
    cl_omg = percent_closeness(Vobs, Vomg)

    # j gate
    j_grid_eps = [0.3, 0.6, 0.9]
    j_grid_q   = [1.0, 2.0]
    j_grid_j0  = [960.0, 1280.0, 1600.0]  # kpc*km/s
    (j_eps, j_q, j_j0), Vj = fit_global_j_gate(Vbar, Vobs, R, j_grid_eps, j_grid_q, j_grid_j0)
    cl_j = percent_closeness(Vobs, Vj)

    # P gate
    p_grid_eps = [0.3, 0.6, 1.0]
    p_grid_q   = [1.0, 2.0]
    p_grid_P0  = [3e5, 5e5, 8e5]  # in (km/s)^3 per kpc
    (p_eps, p_q, p_P0), Vp = fit_global_p_gate(Vbar, Vobs, R, p_grid_eps, p_grid_q, p_grid_P0)
    cl_p = percent_closeness(Vobs, Vp)

    # S gate (nonrecursive)
    s_grid_eps = [0.3, 0.6, 0.9]
    s_grid_q   = [1.0, 2.0]
    s_grid_S0  = [1.0, 1.2]
    (s_eps, s_q, s_S0), Vs = fit_global_s_gate(df.loc[is_outer, :], Vobs, Vbar, R, s_grid_eps, s_grid_q, s_grid_S0)
    cl_s = percent_closeness(Vobs, Vs)

    # D gate (with Omega gate)
    d_grid_eps = [0.3, 0.6, 0.9]
    d_grid_p   = [1.0, 2.0]
    d_grid_oc  = [3e-13, 1e-12]
    d_grid_q   = [1.0, 2.0]
    (d_eps, d_p, d_oc, d_q), Vd = fit_global_d_gate(Vbar, Vobs, R, d_grid_eps, d_grid_p, d_grid_oc, d_grid_q)
    cl_d = percent_closeness(Vobs, Vd)

    # GM toy (heavily gated; tiny effect expected)
    gm_grid_beta = [1e18, 1e20, 1e22]
    gm_grid_p    = [2.0]
    gm_grid_oc   = [1e-12]
    (gm_beta, gm_p, gm_oc), Vgm = fit_global_gm_gate(Vbar, Vobs, R, gm_grid_beta, gm_grid_p, gm_grid_oc)
    cl_gm = percent_closeness(Vobs, Vgm)

    # Summaries (all outer points only)
    stats = {
        'Shell': summarize(cl_shell),
        'GR': summarize(cl_gr),
        'MOND': summarize(cl_mond),
        'NFW': {**summarize(cl_nfw), 'N_gal_fit': int(n_nfw)},
        'RG-RunningG': {**summarize(cl_rg), 'nu': rg_nu, 'r0_kpc': rg_r0, 'rstar_kpc': rg_rstar},
        'MuPhi': {**summarize(cl_phi), 'eps': phi_eps, 'v_c_kms': phi_vc, 'p': phi_p},
        'SlipGate': {**summarize(cl_slip), 'eps': slip_eps, 'r0_kpc': slip_r0, 'delta_kpc': slip_delta},
        'LogTail': {**summarize(cl_log), 'v0_kms': log_v0, 'r_c_kpc': log_rc, 'r0_kpc': log_r0, 'delta_kpc': log_dlt},
        'OmegaGate': {**summarize(cl_omg), 'eps': omg_eps, 'p': omg_p, 'omega_c_s-1': omg_oc},
        'JGate': {**summarize(cl_j), 'eps': j_eps, 'q': j_q, 'j0_kpc_kms': j_j0},
        'PGate': {**summarize(cl_p), 'eps': p_eps, 'q': p_q, 'P0_units': p_P0},
        'SGate': {**summarize(cl_s), 'eps': s_eps, 'q': s_q, 'S0': s_S0},
        'DGate': {**summarize(cl_d), 'eps': d_eps, 'p': d_p, 'omega_c_s-1': d_oc, 'q': d_q},
        'GMtoy': {**summarize(cl_gm), 'beta': gm_beta, 'p': gm_p, 'omega_c_s-1': gm_oc},
        'AntiYukawa': {**summarize(cl_ay), 'alpha': ay_alpha, 'lambda_kpc': ay_lambda},
        'Linear+Newton': {**summarize(cl_ln), 'gamma_per_kpc': ln_gamma},
        'DensityGated': {**summarize(cl_dg), 'eps': dg_eps, 's_c': dg_sc, 'p': dg_p},
    }

    out_json = out_dir / 'accuracy_comparison_extended.json'
    out_json.write_text(json.dumps({'a0_hat': a0_hat, 'models': stats}, indent=2), encoding='utf-8')

    out_png = out_dir / 'accuracy_comparison_extended.png'
    # Choose a plotting order to keep core baselines leftmost
    plot_order = ['Shell','GR','MOND','NFW',
                  'LogTail','MuPhi','RG-RunningG','SlipGate',
                  'OmegaGate','JGate','PGate','SGate','DGate','GMtoy',
                  'AntiYukawa','Linear+Newton','DensityGated']
    plot_order = [m for m in plot_order if m in stats]
    make_plot({k: stats[k] for k in plot_order},
              out_png,
              'Accuracy comparison (outer): Core baselines + proposed models')

    print(f'Wrote {out_png}')
    print(f'Wrote {out_json}')


if __name__ == '__main__':
    main()
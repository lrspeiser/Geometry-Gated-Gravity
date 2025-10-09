#!/usr/bin/env python3
from __future__ import annotations
"""
Plot HLSP lensing overview for a given cluster/team/version.

Renders a 2x2 grid:
- kappa with tangential critical curve (T = 1 - kappa - |gamma| = 0) and seeded-main component overlay
- |mu| (abs magnification) with negative-mu region overlay (if signed)
- Deflection streamlines/quiver (arcsec units if available)
- Diagnostics: text box with theta_E from (kappa,gamma) vs (mu), pixel scale, and counts

Saves PNG to: out/plots/<cluster>_<team>_<version>_hlsp_overview.png

Usage:
  python scripts/plot_hlsp_lensing_overview.py --cluster macs0416 --team cats --version v4.1
"""
import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'out' / 'plots'
OUTDIR.mkdir(parents=True, exist_ok=True)
HLSP = ROOT / 'data' / 'frontier' / 'hlsp'

# project imports for GR/accepted deflection curves
import sys
sys.path.insert(0, str(ROOT))
from scripts.lensing_utils import alpha_fun_GR_baryons, alpha_fun_ACCEPTED, alpha_fun_GE, solve_theta_E_from_alpha, CLASH  # type: ignore


def find_first_file(base: Path, substrings: list[str]) -> Path | None:
    subs_l = [s.lower() for s in substrings]
    if not base.exists() or not base.is_dir():
        return None
    for p in base.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if all(s in name for s in subs_l):
            return p
    return None


def pixel_scale_arcsec(header) -> float:
    try:
        w = WCS(header)
        sc = proj_plane_pixel_scales(w)  # deg/pix
        return float(np.mean(sc[:2]) * 3600.0)
    except Exception:
        return 0.2


def load_hlsp_maps(cluster: str, team: str, version: str):
    base = HLSP / cluster.lower() / team.lower() / version
    if not base.exists():
        raise FileNotFoundError(f"Missing {base}")
    # kappa
    kappa_fp = find_first_file(base, ['kappa', '.fits'])
    if kappa_fp is None:
        raise FileNotFoundError(f"No kappa*.fits in {base}")
    with fits.open(str(kappa_fp)) as hk:
        kappa = np.array(hk[0].data, dtype=float)
        hdr_k = hk[0].header
    # gamma
    gamma_fp = find_first_file(base, ['gamma.fits'])
    if gamma_fp is not None:
        with fits.open(str(gamma_fp)) as hg:
            gam = np.array(hg[0].data, dtype=float)
        g1_fp = find_first_file(base, ['gamma1', '.fits'])
        g2_fp = find_first_file(base, ['gamma2', '.fits'])
        if g1_fp is not None and g2_fp is not None:
            with fits.open(str(g1_fp)) as h1, fits.open(str(g2_fp)) as h2:
                gamma1 = np.array(h1[0].data, dtype=float)
                gamma2 = np.array(h2[0].data, dtype=float)
        else:
            # split not available; assume gamma map already magnitude, place in g1 and zeros in g2
            gamma1 = gam
            gamma2 = np.zeros_like(gam)
    else:
        g1_fp = find_first_file(base, ['gamma1', '.fits'])
        g2_fp = find_first_file(base, ['gamma2', '.fits'])
        if g1_fp is None or g2_fp is None:
            raise FileNotFoundError(f"Missing gamma files in {base}")
        with fits.open(str(g1_fp)) as h1, fits.open(str(g2_fp)) as h2:
            gamma1 = np.array(h1[0].data, dtype=float)
            gamma2 = np.array(h2[0].data, dtype=float)
    # magnification (z02) if present
    mu_fp = find_first_file(base, ['z02-magnif', '.fits'])
    mu = None
    if mu_fp is not None:
        with fits.open(str(mu_fp)) as hm:
            mu = np.array(hm[0].data, dtype=float)
    # deflection (arcsec preferred)
    fx = (find_first_file(base, ['x-arcsec-deflect', '.fits']) or
          find_first_file(base, ['x-pixels-deflect', '.fits']))
    fy = (find_first_file(base, ['y-arcsec-deflect', '.fits']) or
          find_first_file(base, ['y-pixels-deflect', '.fits']))
    ax = ay = None
    if fx is not None and fy is not None:
        with fits.open(str(fx)) as hx, fits.open(str(fy)) as hy:
            ax = np.array(hx[0].data, dtype=float); hdrx = hx[0].header
            ay = np.array(hy[0].data, dtype=float); hdry = hy[0].header
        # convert pixel deflection to arcsec
        if 'pixels-deflect' in fx.name.lower() or hdrx.get('BUNIT','').lower().startswith('pixel'):
            ps = pixel_scale_arcsec(hdrx)
            ax = ax * ps
        if 'pixels-deflect' in fy.name.lower() or hdry.get('BUNIT','').lower().startswith('pixel'):
            ps = pixel_scale_arcsec(hdry)
            ay = ay * ps
    # clean NaNs
    kappa = np.nan_to_num(kappa, nan=0.0)
    gamma1 = np.nan_to_num(gamma1, nan=0.0)
    gamma2 = np.nan_to_num(gamma2, nan=0.0)
    if mu is not None:
        mu = np.nan_to_num(mu, nan=np.nan, posinf=np.nan, neginf=np.nan)
    if ax is not None:
        ax = np.nan_to_num(ax, nan=0.0)
    if ay is not None:
        ay = np.nan_to_num(ay, nan=0.0)
    return {
        'base': base,
        'kappa': kappa,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'mu': mu,
        'ax': ax,
        'ay': ay,
        'hdr_k': hdr_k,
    }


def main_tangential_mask(kappa: np.ndarray, g1: np.ndarray, g2: np.ndarray, tol: float = 0.02):
    gam = np.sqrt(g1*g1 + g2*g2)
    T = 1.0 - kappa - gam
    cy, cx = np.unravel_index(int(np.nanargmax(kappa)), kappa.shape)
    ny, nx = kappa.shape
    visited = np.zeros_like(T, dtype=bool)
    stack = [(cy, cx)]
    while stack:
        y, x = stack.pop()
        if not (0 <= y < ny and 0 <= x < nx) or visited[y, x]:
            continue
        if T[y, x] <= tol:
            visited[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        stack.append((y + dy, x + dx))
    mask = (T <= 0) & visited
    return T, mask, (cy, cx)


def plot_overview(cluster: str, team: str, version: str, vmax_mu: float = 50.0, zs: float = 2.0,
                 ge_a: float = 3.0, ge_b: float = 0.2, ge_d: float = 0.1,
                 ge_gamma1: float = 0.2, ge_gamma2: float = 0.1,
                 ge_Rd_kpc: float = 1000.0, ge_Rscale_kpc: float = 100.0,
                 ge_A_core: float = 0.25, ge_p_core: float = 2.0,
                 ge_Sigma0_hat: float = 0.0, ge_beta_core: float = 0.4,
                 ge_smooth_R_kpc: float = 5.0,
                 make_extras: bool = False):
    data = load_hlsp_maps(cluster, team, version)
    kappa = data['kappa']; g1 = data['gamma1']; g2 = data['gamma2']
    mu = data['mu']; ax = data['ax']; ay = data['ay']
    pixscale = pixel_scale_arcsec(data['hdr_k'])

    T, mask, (cy, cx) = main_tangential_mask(kappa, g1, g2)
    n_pix = int(np.count_nonzero(mask))
    if n_pix > 0:
        area_arcsec2 = n_pix * (pixscale**2)
        thetaE_T = math.sqrt(area_arcsec2 / math.pi)
    else:
        thetaE_T = float('nan')

    thetaE_mu = None
    neg_mu_mask = None
    if mu is not None and np.nanmin(mu) < 0:
        neg_mu_mask = (mu < 0)
        # take largest connected negative area (simple heuristic via labeling by flood fill around cy,cx)
        # fallback: area of all neg
        n_neg = int(np.count_nonzero(neg_mu_mask))
        if n_neg > 10:
            area_m = n_neg * (pixscale**2)
            thetaE_mu = math.sqrt(area_m / math.pi)

    ny, nx = kappa.shape
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: kappa with T=0 contour and seeded mask
    ax1 = axes[0,0]
    im1 = ax1.imshow(kappa, origin='lower', cmap='magma')
    c1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    c1.set_label('κ')
    # T=0 level
    CS = ax1.contour(T, levels=[0.0], colors='cyan', linewidths=1.0)
    # seeded main component boundary
    ax1.contour(mask.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
    # annotate center and thetaE circle
    ax1.plot([cx], [cy], 'wo', ms=3)
    if not math.isnan(thetaE_T):
        r_pix = thetaE_T / pixscale
        th = np.linspace(0, 2*np.pi, 360)
        ax1.plot(cx + r_pix*np.cos(th), cy + r_pix*np.sin(th), color='lime', lw=1.0, alpha=0.8)
    ax1.set_title(f"{cluster} {team} {version}: κ with T=0 (cyan), main T≤0 (lime)")

    # Panel 2: |mu| with negative-μ region overlay
    ax2 = axes[0,1]
    if mu is not None:
        abs_mu = np.abs(mu)
        disp = np.clip(abs_mu, 0, vmax_mu)
        im2 = ax2.imshow(disp, origin='lower', cmap='viridis')
        c2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        c2.set_label('|μ| (clipped)')
        if neg_mu_mask is not None:
            ax2.contour(neg_mu_mask.astype(float), levels=[0.5], colors='red', linewidths=0.8)
        ax2.set_title('|μ| with negative-μ region (red)')
    else:
        ax2.text(0.5, 0.5, 'No magnification map', transform=ax2.transAxes, ha='center')
        ax2.set_title('|μ| (unavailable)')

    # Panel 3: Deflection quiver/streamlines (arcsec)
    ax3 = axes[1,0]
    if ax is not None and ay is not None:
        # decimate for readability
        step = max(1, min(ny, nx)//30)
        yy, xx = np.mgrid[0:ny:step, 0:nx:step]
        U = ax[::step, ::step]
        V = ay[::step, ::step]
        scale = None  # auto
        ax3.quiver(xx, yy, U, V, color='w', angles='xy', scale_units='xy', scale=scale, width=0.002)
        ax3.imshow(kappa, origin='lower', cmap='gray', alpha=0.6)
        ax3.set_title('Deflection (arcsec) quiver over κ')
    else:
        ax3.imshow(kappa, origin='lower', cmap='gray')
        ax3.set_title('Deflection unavailable')

    # Panel 4: Deflection curves (GR baryons vs accepted) and diagnostics
    ax4 = axes[1,1]
    # Build accepted deflection (alpha_y along vertical cut) and GR baryons-only
    theta_grid = np.linspace(2.0, min(100.0, max(nx, ny) * pixscale * 0.4), 200)
    y_acc = None; y_gr = None; y_ge = None
    thetaE_ACC = None; thetaE_GR = None; thetaE_GE = None
    # Accepted from HLSP deflection maps
    try:
        if cluster.lower() in CLASH:
            local_name, z_lens = CLASH[cluster.lower()]
        else:
            z_lens = 0.3
        alpha_acc = alpha_fun_ACCEPTED(cluster.lower(), team.lower(), version, z_lens, zs)
        if alpha_acc is not None:
            y_acc = np.array([alpha_acc(th) for th in theta_grid])  # ARCSEC
            thetaE_ACC = solve_theta_E_from_alpha(lambda th: alpha_acc(th), theta_guess_arcsec=20.0, theta_min_arcsec=5.0, theta_max_arcsec=80.0)
    except Exception:
        y_acc = None
        thetaE_ACC = None
    # GR baryons-only from profiles
    # DISABLED: GR calculation currently broken (produces values 1000x too large)
    # try:
    #     if cluster.lower() in CLASH:
    #         local_name, z_lens = CLASH[cluster.lower()]
    #         alpha_gr = alpha_fun_GR_baryons(local_name, z_lens, zs)
    #         if alpha_gr is not None:
    #             y_gr = np.array([alpha_gr(th) for th in theta_grid])  # ARCSEC
    #             thetaE_GR = solve_theta_E_from_alpha(lambda th: alpha_gr(th), theta_guess_arcsec=20.0, theta_min_arcsec=5.0, theta_max_arcsec=80.0)
    # except Exception:
    #     y_gr = None
    #     thetaE_GR = None
    # GE (custom formula) from Sigma_eff
    # DISABLED: GE calculation currently broken (returns None)
    # try:
    #     if cluster.lower() in CLASH:
    #         local_name, z_lens = CLASH[cluster.lower()]
    #         alpha_ge = alpha_fun_GE(local_name, z_lens, zs,
    #                                 a=ge_a, b=ge_b, d=ge_d,
    #                                 gamma1=ge_gamma1, gamma2=ge_gamma2,
    #                                 Rd_kpc=ge_Rd_kpc, R_scale_kpc=ge_Rscale_kpc,
    #                                 beta_clip=(1.0, 5.0),
    #                                 A_core=ge_A_core, p_core=ge_p_core,
    #                                 Sigma0_hat=ge_Sigma0_hat, beta_core=ge_beta_core,
    #                                 smooth_R_kpc=ge_smooth_R_kpc)
    #         if alpha_ge is not None:
    #             y_ge = np.array([alpha_ge(th) for th in theta_grid])  # ARCSEC
    #             thetaE_GE = solve_theta_E_from_alpha(lambda th: alpha_ge(th), theta_guess_arcsec=20.0, theta_min_arcsec=5.0, theta_max_arcsec=80.0)
    # except Exception:
    #     y_ge = None
    #     thetaE_GE = None

    # Plot
    ax4.plot(theta_grid, theta_grid, 'k--', lw=1.0, label='α=θ')
    if y_acc is not None:
        ax4.plot(theta_grid, y_acc, color='tab:blue', lw=2.0, label='Accepted α(θ) from HLSP')
    if y_gr is not None:
        ax4.plot(theta_grid, y_gr, color='tab:orange', lw=2.0, label='GR (baryons) α(θ)')
    if y_ge is not None:
        ax4.plot(theta_grid, y_ge, color='tab:green', lw=2.0, label='GE (custom) α(θ)')
    if thetaE_ACC is not None:
        ax4.axvline(thetaE_ACC, color='tab:blue', ls=':', lw=1.2)
    if thetaE_GR is not None:
        ax4.axvline(thetaE_GR, color='tab:orange', ls=':', lw=1.2)
    if thetaE_GE is not None:
        ax4.axvline(thetaE_GE, color='tab:green', ls=':', lw=1.2)
    ax4.set_xlabel('θ (arcsec)')
    ax4.set_ylabel('α(θ) (arcsec)')
    ax4.set_title('Deflection curves')
    ax4.legend(loc='lower right', fontsize=8)
    # Diagnostics inset
    lines = [
        f"Pixel scale: {pixscale:.3f} arcsec/pixel",
        f"Main T≤0 pix: {n_pix}",
        f"θ_E (T≤0 area): {thetaE_T:.2f}″" if not math.isnan(thetaE_T) else "θ_E (T≤0 area): n/a",
        f"θ_E (accepted α): {thetaE_ACC:.2f}″" if thetaE_ACC is not None else "θ_E (accepted α): n/a",
        f"θ_E (GR baryons): {thetaE_GR:.2f}″" if thetaE_GR is not None else "θ_E (GR baryons): n/a",
        f"θ_E (GE custom): {thetaE_GE:.2f}″" if thetaE_GE is not None else "θ_E (GE custom): n/a",
        f"GE params: a={ge_a}, b={ge_b}, d={ge_d}, γ1={ge_gamma1}, γ2={ge_gamma2}, Rd={ge_Rd_kpc} kpc, Rscale={ge_Rscale_kpc} kpc",
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), va='top', ha='left', transform=ax4.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    fig.tight_layout()
    outp = OUTDIR / f"{cluster}_{team}_{version}_hlsp_overview.png"
    fig.savefig(outp, dpi=140)
    plt.close(fig)

    # --- Optional extra plots to make GR/GE visible and comparable ---
    if make_extras:
        # 1) Deflection zoom (focus on GR/GE amplitudes)
        if (y_gr is not None) or (y_ge is not None):
            plt.figure(figsize=(6,4))
            if y_acc is not None:
                plt.plot(theta_grid, y_acc, color='tab:blue', lw=1.0, alpha=0.4, label='Accepted α(θ) [context]')
            if y_gr is not None:
                plt.plot(theta_grid, y_gr, color='tab:orange', lw=2.0, label='GR (baryons)')
            if y_ge is not None:
                plt.plot(theta_grid, y_ge, color='tab:green', lw=2.0, label='GE (custom)')
            ymax = 0.0
            for arr in (y_gr, y_ge):
                if arr is not None:
                    ymax = max(ymax, float(np.nanpercentile(arr, 99)))
            ymax = max(ymax, 1.0)
            plt.ylim(0, ymax*1.2)
            plt.xlabel('θ (arcsec)'); plt.ylabel('α(θ) (arcsec)'); plt.title('Deflection (zoom)')
            plt.legend(loc='upper left', fontsize=8)
            plt.grid(alpha=0.3, ls=':')
            zpath = OUTDIR / f"{cluster}_{team}_{version}_deflection_zoom.png"
            plt.tight_layout(); plt.savefig(zpath, dpi=140); plt.close()

        # 2) Mean convergence k̄ = α/θ for all three
        plt.figure(figsize=(6,4))
        eps = 1e-6
        if y_acc is not None:
            plt.plot(theta_grid, y_acc/np.maximum(theta_grid, eps), color='tab:blue', lw=1.5, label='Accepted k̄(<θ)')
        if y_gr is not None:
            plt.plot(theta_grid, y_gr/np.maximum(theta_grid, eps), color='tab:orange', lw=1.5, label='GR k̄(<θ)')
        if y_ge is not None:
            plt.plot(theta_grid, y_ge/np.maximum(theta_grid, eps), color='tab:green', lw=1.5, label='GE k̄(<θ)')
        plt.axhline(1.0, color='k', ls='--', lw=1.0, alpha=0.5)
        plt.xlabel('θ (arcsec)'); plt.ylabel('k̄(<θ)'); plt.title('Mean convergence k̄(<θ) = α/θ')
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(alpha=0.3, ls=':')
        kpath = OUTDIR / f"{cluster}_{team}_{version}_kbar.png"
        plt.tight_layout(); plt.savefig(kpath, dpi=140); plt.close()

        # 3) GE/GR ratio
        if (y_gr is not None) and (y_ge is not None):
            plt.figure(figsize=(6,4))
            ratio = np.divide(y_ge, np.maximum(y_gr, eps))
            plt.plot(theta_grid, ratio, color='tab:purple', lw=1.8)
            plt.axhline(1.0, color='k', ls='--', lw=1.0, alpha=0.5)
            plt.xlabel('θ (arcsec)'); plt.ylabel('α_GE/α_GR'); plt.title('GE/GR deflection ratio')
            plt.grid(alpha=0.3, ls=':')
            rpath = OUTDIR / f"{cluster}_{team}_{version}_ge_over_gr.png"
            plt.tight_layout(); plt.savefig(rpath, dpi=140); plt.close()

    return outp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--team', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--vmax-mu', type=float, default=50.0)
    ap.add_argument('--zs', type=float, default=2.0)
    # GE (custom) parameters
    ap.add_argument('--ge-a', type=float, default=3.0)
    ap.add_argument('--ge-b', type=float, default=0.2)
    ap.add_argument('--ge-d', type=float, default=0.1)
    ap.add_argument('--ge-gamma1', type=float, default=0.2)
    ap.add_argument('--ge-gamma2', type=float, default=0.1)
    ap.add_argument('--ge-Rd-kpc', type=float, default=1000.0)
    ap.add_argument('--ge-Rscale-kpc', type=float, default=100.0)
    ap.add_argument('--ge-A-core', type=float, default=0.25)
    ap.add_argument('--ge-p-core', type=float, default=2.0)
    ap.add_argument('--ge-Sigma0-hat', type=float, default=0.0)
    ap.add_argument('--ge-beta-core', type=float, default=0.4)
    ap.add_argument('--ge-smooth-R-kpc', type=float, default=5.0)
    ap.add_argument('--extras', action='store_true', help='Write extra zoom/ratio plots')
    args = ap.parse_args()
    outp = plot_overview(
        args.cluster, args.team, args.version,
        vmax_mu=args.vmax_mu, zs=args.zs,
        ge_a=args.ge_a, ge_b=args.ge_b, ge_d=args.ge_d,
        ge_gamma1=args.ge_gamma1, ge_gamma2=args.ge_gamma2,
        ge_Rd_kpc=args.ge_Rd_kpc, ge_Rscale_kpc=args.ge_Rscale_kpc,
        ge_A_core=args.ge_A_core, ge_p_core=args.ge_p_core,
        ge_Sigma0_hat=args.ge_Sigma0_hat, ge_beta_core=args.ge_beta_core,
        ge_smooth_R_kpc=args.ge_smooth_R_kpc,
        make_extras=bool(args.extras)
    )
    print(f"Wrote {outp}")


if __name__ == '__main__':
    main()
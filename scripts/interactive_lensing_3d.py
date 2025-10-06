#!/usr/bin/env python3
from __future__ import annotations
"""
Interactive 3D lensing visualization with mass-scale slider.

- Draw a translucent sphere for the cluster center
- Draw two sets of photon paths (thin-lens kink at x=0) in 3D (z=0 plane):
  • Actual (HLSP-based) — blue
  • GR (baryons) scaled by a mass factor — red (default scale=1.0)
- Provide a slider for mass_scale that multiplies GR deflection α_GR(θ) so you can
  see whether scaling baryonic mass alone could reproduce the actual bending.

Outputs per cluster:
  out/visualizations/<cluster_id>/interactive_3d.html

Usage:
  python scripts/interactive_lensing_3d.py --cluster_id a209 --zs 2.0 --rays 9

Notes:
- Deflection per ray depends on impact parameter θ for both Actual and GR cases.
- Actual deflection uses HLSP κ radialization and WCS pixel scale if available.
  If WCS is unavailable, we normalize such that α(θ_obs)=θ_obs for visibility.
- Colors: Actual = blue, GR(baryons) = red (consistent with labels on-plot).
"""
import argparse
from pathlib import Path
import sys
import glob
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc,
)

try:
    import plotly.graph_objects as go  # type: ignore
    from astropy.io import fits  # type: ignore
    from astropy.wcs import WCS  # type: ignore
    from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore
    PKG_OK = True
except Exception:
    PKG_OK = False

CLASH = {
    'a1423': ('ABELL_1423', 0.213), 'a209': ('ABELL_0209', 0.206), 'a2261': ('ABELL_2261', 0.224),
    'a383': ('ABELL_0383', 0.187), 'a611': ('ABELL_0611', 0.288), 'clj1226': ('CLJ1226', 0.890),
    'macs0329': ('MACSJ0329', 0.450), 'macs0416': ('MACSJ0416', 0.396), 'macs0429': ('MACSJ0429', 0.399),
    'macs0647': ('MACSJ0647', 0.584), 'macs0717': ('MACSJ0717', 0.548), 'macs0744': ('MACSJ0744', 0.686),
    'macs1115': ('MACSJ1115', 0.352), 'macs1149': ('MACSJ1149', 0.544), 'macs1206': ('MACSJ1206', 0.440),
    'macs1311': ('MACSJ1311', 0.494), 'macs1423': ('MACSJ1423', 0.545), 'macs1720': ('MACSJ1720', 0.391),
    'macs1931': ('MACSJ1931', 0.352), 'macs2129': ('MACSJ2129', 0.570), 'ms2137': ('MS2137', 0.313),
    'rxj1347': ('RXJ1347', 0.451), 'rxj1532': ('RXJ1532', 0.345), 'rxj2129': ('RXJ2129', 0.234), 'rxj2248': ('RXJ2248', 0.348),
}

def find_hlsp_kappa_with_scale(cluster_id: str):
    base = ROOT / 'data' / 'clash' / 'hlsp' / cluster_id.lower()
    pats = [str(base / '**' / '*kappa*.fits'), str(base / '**' / '*kappa1*.fits')]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files_sorted = sorted(files, key=lambda s: (0 if 'kappa.fits' in Path(s).name else 1, len(s)))
    for fp in files_sorted:
        try:
            with fits.open(fp) as hdul:
                hdr = hdul[0].header
                arr = hdul[0].data
                if arr is None or arr.size == 0:
                    continue
                arr = np.array(arr, dtype=float)
                arcsec_per_pix = None
                try:
                    w = WCS(hdr)
                    scales = proj_plane_pixel_scales(w)  # deg/pix
                    if scales is not None and len(scales) >= 2:
                        arcsec_per_pix = float(np.mean(scales[:2]) * 3600.0)
                except Exception:
                    arcsec_per_pix = None
                return arr, arcsec_per_pix
        except Exception:
            continue
    return None, None


def radial_cummean_kappa_from_map(kappa2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(kappa2d, float)
    ny, nx = img.shape
    y = np.arange(ny) - (ny - 1) / 2.0
    x = np.arange(nx) - (nx - 1) / 2.0
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X*X + Y*Y)
    rmax = R.max()
    nbins = 300
    rbins = np.linspace(0, rmax, nbins+1)
    rmid = 0.5*(rbins[:-1] + rbins[1:])
    kcum = np.zeros_like(rmid)
    for i in range(1, len(rmid)):
        m = R <= rbins[i]
        if np.any(m):
            kcum[i] = float(np.mean(img[m]))
        else:
            kcum[i] = kcum[i-1]
    return rmid, kcum


def alpha_fun_GR_baryons(local_name: str, z_lens: float, z_source: float):
    r, rho = load_real_cluster_profiles(local_name)
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    Mproj = np.array([2*np.pi*np.trapezoid(Sigma_bar[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
    area = np.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    kbar_mean = Sbar / Sigma_crit
    Dd = angular_diameter_distance_kpc(z_lens)
    theta_arcsec_grid = (R / max(Dd, 1e-12)) * 206265.0
    def alpha_of(theta_arcsec: float) -> float:
        kbar = float(np.interp(theta_arcsec, theta_arcsec_grid, kbar_mean,
                               left=kbar_mean[0], right=kbar_mean[-1]))
        theta_rad = theta_arcsec / 206265.0
        return kbar * theta_rad
    return alpha_of


def alpha_fun_HLSP(cluster_id: str, theta_obs_arcsec: float):
    kappa, arcsec_per_pix = find_hlsp_kappa_with_scale(cluster_id)
    if kappa is None or arcsec_per_pix is None or not np.isfinite(arcsec_per_pix) or arcsec_per_pix <= 0:
        return None
    # Build 2D deflection field via FFT Poisson solve: ∇²ψ = 2κ, α = ∇ψ
    img = np.asarray(kappa, float)
    ny, nx = img.shape
    K = np.fft.fft2(img)
    fx = np.fft.fftfreq(nx)  # cycles per pixel
    fy = np.fft.fftfreq(ny)
    kx = (2*np.pi) * fx  # radians per pixel
    ky = (2*np.pi) * fy
    kxg, kyg = np.meshgrid(kx, ky)
    k2 = kxg*kxg + kyg*kyg
    # Avoid division by zero at k=0
    mask0 = (k2 == 0)
    k2_safe = k2.copy(); k2_safe[mask0] = 1.0
    Psi = -2.0 * K / k2_safe
    Psi[mask0] = 0.0
    Ax = (1j) * kxg * Psi
    Ay = (1j) * kyg * Psi
    ax_pix = np.real(np.fft.ifft2(Ax))
    ay_pix = np.real(np.fft.ifft2(Ay))
    # Convert to arcsec units
    ax_arcsec = ax_pix * arcsec_per_pix
    ay_arcsec = ay_pix * arcsec_per_pix
    # Re-center to κ peak
    peak_idx = np.unravel_index(int(np.nanargmax(img)), img.shape)
    cy, cx = float(peak_idx[0]), float(peak_idx[1])
    # Normalize deflection so mean |α| at θ_obs equals θ_obs (in radians)
    rp = float(theta_obs_arcsec / arcsec_per_pix)
    nang = 360
    ang = np.linspace(0, 2*np.pi, nang, endpoint=False)
    xs = cx + rp * np.cos(ang)
    ys = cy + rp * np.sin(ang)
    def bilinear(a, xp, yp):
        x0 = np.clip(np.floor(xp).astype(int), 0, nx-1)
        y0 = np.clip(np.floor(yp).astype(int), 0, ny-1)
        x1 = np.clip(x0+1, 0, nx-1)
        y1 = np.clip(y0+1, 0, ny-1)
        tx = xp - x0; ty = yp - y0
        return ( (1-tx)*(1-ty)*a[y0, x0] + tx*(1-ty)*a[y0, x1] + (1-tx)*ty*a[y1, x0] + tx*ty*a[y1, x1] )
    ax_ring = bilinear(ax_arcsec, xs, ys)
    ay_ring = bilinear(ay_arcsec, xs, ys)
    alpha_mag_ring_rad = np.sqrt(ax_ring**2 + ay_ring**2) / 206265.0
    target = float(theta_obs_arcsec / 206265.0)
    current = float(np.nanmean(alpha_mag_ring_rad)) if np.isfinite(alpha_mag_ring_rad).all() else 1.0
    scale = target / current if current > 0 else 1.0
    ax_arcsec *= scale; ay_arcsec *= scale
    # Build interpolator for α_y along central vertical cut through the peak
    def alpha_y_of(y_arcsec: float) -> float:
        y_pix = cy + (y_arcsec / arcsec_per_pix)
        x_pix = cx
        x0 = int(np.floor(x_pix)); y0 = int(np.floor(y_pix))
        x1 = min(x0 + 1, nx - 1); y1 = min(y0 + 1, ny - 1)
        tx = x_pix - x0; ty = y_pix - y0
        ay_here_arcsec = ( (1-tx)*(1-ty)*ay_arcsec[y0, x0] + tx*(1-ty)*ay_arcsec[y0, x1] + (1-tx)*ty*ay_arcsec[y1, x0] + tx*ty*ay_arcsec[y1, x1] )
        return float(ay_here_arcsec / 206265.0)
    return alpha_y_of


def paths_for_scale(theta_list: np.ndarray, alpha_gr_fun, alpha_hl_fun_y, mass_scale: float, mag: float,
                    xspan_factor: float = 2.5) -> tuple[list[np.ndarray], list[np.ndarray]]:
    paths_gr = []
    paths_hl = []
    for theta0 in theta_list:
        xspan = xspan_factor * theta0
        # pre
        x_pre = np.linspace(-xspan, 0.0, 100)
        y_pre = np.full_like(x_pre, theta0)
        z_pre = np.zeros_like(x_pre)
        # post
        x_post = np.linspace(0.0, xspan, 200)
        # GR: use magnitude α(θ) but we need y-component; for our x-axis rays, sign by y0
        a_gr = alpha_gr_fun(abs(theta0)) * mass_scale
        a_gr_y = np.sign(theta0) * a_gr
        # HLSP: use 2D α_y field at y=theta0
        a_hl_y = alpha_hl_fun_y(theta0)
        y_post_gr = theta0 - (a_gr_y * mag) * x_post
        y_post_hl = theta0 - (a_hl_y * mag) * x_post
        z_post = np.zeros_like(x_post)
        pgr = np.vstack([np.concatenate([x_pre, x_post]),
                         np.concatenate([y_pre, y_post_gr]),
                         np.concatenate([z_pre, z_post])]).T
        phl = np.vstack([np.concatenate([x_pre, x_post]),
                         np.concatenate([y_pre, y_post_hl]),
                         np.concatenate([z_pre, z_post])]).T
        paths_gr.append(pgr)
        paths_hl.append(phl)
    return paths_gr, paths_hl


def build_sphere(theta_E_arcsec: float, radius_factor: float = 0.5):
    r = radius_factor * theta_E_arcsec
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs


def fig_for_cluster(cid: str, zs: float, rays: int, mag: float, y_src_frac: float = 0.0, sphere_mode: str = 'wireframe') -> Path:
    if not PKG_OK:
        raise RuntimeError('This visualization requires plotly and astropy installed')

    # observed θE
    obs_csv = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
    df = pd.read_csv(obs_csv)
    df['cluster_id'] = df['cluster_id'].str.lower()
    row = df[df['cluster_id'] == cid]
    if len(row) == 0:
        raise RuntimeError('No observed θE for this cluster')
    theta_obs = float(row.iloc[0]['theta_E_observed_arcsec'])

    # cluster spec
    if cid not in CLASH:
        raise RuntimeError('Unknown cluster_id')
    local_name, z_lens = CLASH[cid]

    # deflection functions
    alpha_gr = alpha_fun_GR_baryons(local_name, z_lens, zs)
    alpha_hl_y = alpha_fun_HLSP(cid, theta_obs)
    if alpha_hl_y is None:
        # fallback normalization: α(θ)=θ (radians)
        def alpha_hl_y(theta_arcsec: float) -> float:
            return theta_arcsec / 206265.0

    # impact parameters around θE, centered at source offset
    nr = max(1, int(rays))
    scales = np.linspace(0.6, 1.4, nr) if nr > 1 else np.array([1.0])
    theta_center = float(y_src_frac) * theta_obs
    theta_list = theta_center + scales * theta_obs

    # precompute for multiple mass scales
    mass_scales = np.round(np.linspace(0.5, 3.0, 11), 2)  # 0.5 to 3.0
    bundles_by_scale = {}
    for ms in mass_scales:
        bundles_by_scale[ms] = paths_for_scale(theta_list, alpha_gr, alpha_hl_y, ms, mag)

    # build 3D traces for initial scale (1.0)
    init_scale = 1.0
    pgr_list, phl_list = bundles_by_scale[init_scale]

    # sphere mesh
    xs, ys, zs = build_sphere(theta_obs, radius_factor=0.5)

    # Build wireframe sphere lines
    wf_traces = []
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    # iso-u lines (vary v for each fixed u)
    for ui in u:
        xw = (theta_obs*0.5) * np.cos(ui) * np.sin(v)
        yw = (theta_obs*0.5) * np.sin(ui) * np.sin(v)
        zw = (theta_obs*0.5) * np.cos(v)
        wf_traces.append(go.Scatter3d(x=xw, y=yw, z=zw, mode='lines', line=dict(color='#AAAAAA', width=2), showlegend=False, name='Sphere wire'))
    # iso-v lines (vary u for each fixed v excluding poles)
    for vv in v[1:-1]:
        xw = (theta_obs*0.5) * np.cos(u) * np.sin(vv)
        yw = (theta_obs*0.5) * np.sin(u) * np.sin(vv)
        zw = (theta_obs*0.5) * np.full_like(u, np.cos(vv))
        wf_traces.append(go.Scatter3d(x=xw, y=yw, z=zw, mode='lines', line=dict(color='#AAAAAA', width=2), showlegend=False, name='Sphere wire'))

    # Compute fixed axis ranges across all mass scales so the sphere never flattens or re-zooms
    all_x = [xs.min(), xs.max()]
    all_y = [ys.min(), ys.max()]
    all_z = [zs.min(), zs.max()]
    for ms in mass_scales:
        pgr_ms, phl_ms = bundles_by_scale[ms]
        for arr in pgr_ms + phl_ms:
            all_x.extend([arr[:,0].min(), arr[:,0].max()])
            all_y.extend([arr[:,1].min(), arr[:,1].max()])
            all_z.extend([arr[:,2].min(), arr[:,2].max()])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    zmin, zmax = float(np.min(all_z)), float(np.max(all_z))

    traces = []
    # Sphere traces: solid surface first, then wireframe lines (we’ll toggle visibility)
    solid_sphere = go.Surface(x=xs, y=ys, z=zs, opacity=0.15, colorscale=[[0,'#D3D3D3'],[1,'#D3D3D3']], showscale=False, name='Cluster')
    traces.append(solid_sphere)
    # Append wireframe traces
    for t in wf_traces:
        traces.append(t)

    # Actual (blue) and GR (red) for initial scale (legend on first of each)
    for idx, phl in enumerate(phl_list):
        traces.append(go.Scatter3d(x=phl[:,0], y=phl[:,1], z=phl[:,2], mode='lines',
                                   line=dict(color='#0077ff', width=5), name='Actual (HLSP)', showlegend=(idx==0)))
    for idx, pgr in enumerate(pgr_list):
        traces.append(go.Scatter3d(x=pgr[:,0], y=pgr[:,1], z=pgr[:,2], mode='lines',
                                   line=dict(color='#dd2222', width=5, dash='dash'), name='GR (baryons)', showlegend=(idx==0)))

    # For slider steps, we will update only GR traces (keep Actual fixed) and keep axis ranges constant
    frames = []
    for ms in mass_scales:
        pgr_ms, phl_ms = bundles_by_scale[ms]
        frame_data = []
        offset = 1 + len(phl_list)
        for idx, tr in enumerate(traces):
            if idx < offset:
                frame_data.append(tr)
            elif idx < offset + len(pgr_list):
                j = idx - offset
                frame_data.append(go.Scatter3d(x=pgr_ms[j][:,0], y=pgr_ms[j][:,1], z=pgr_ms[j][:,2], mode='lines', line=dict(color='#dd2222', width=5, dash='dash'), name='GR (baryons)', showlegend=False))
        frames.append(go.Frame(data=frame_data, name=f'scale={ms:.2f}', layout=dict(
            scene=dict(xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]), zaxis=dict(range=[zmin, zmax],),
                       aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6))
        )))

    # Visibility switcher (Actual only / GR only / Both)
    n_solid = 1
    n_wire = len(wf_traces)
    n_sphere_total = n_solid + n_wire
    n_actual = len(phl_list)
    n_gr = len(pgr_list)
    # Base sphere visibility modes
    if sphere_mode.lower() == 'wireframe':
        sphere_vis = [False] + [True]*n_wire
    elif sphere_mode.lower() == 'off':
        sphere_vis = [False] + [False]*n_wire
    else:  # solid
        sphere_vis = [True] + [False]*n_wire
    vis_actual = sphere_vis + [True]*n_actual + [False]*n_gr
    vis_gronly = sphere_vis + [False]*n_actual + [True]*n_gr
    vis_both   = sphere_vis + [True]*n_actual + [True]*n_gr

    # Layout with slider
    steps = []
    for i, ms in enumerate(mass_scales):
        steps.append(dict(method='animate', label=f'{ms:.2f}', args=[[f'scale={ms:.2f}'], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))]))

    sliders = [dict(active=int(np.where(mass_scales==init_scale)[0][0]) if init_scale in mass_scales else 0,
                    currentvalue={'prefix': 'Mass scale (GR): '},
                    pad={'t': 10}, steps=steps)]

    # Apply initial visibility for sphere mode
    init_visible = vis_both
    fig = go.Figure(data=traces, frames=frames)
    for i in range(len(traces)):
        fig.data[i].visible = init_visible[i] if i < len(init_visible) else True
    fig.update_layout(title=f'{cid}: 3D photon paths (Actual blue vs GR red) – rotate with mouse',
                      scene=dict(xaxis_title='x (arcsec)', yaxis_title='y (arcsec)', zaxis_title='z (arcsec)',
                                 xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]), zaxis=dict(range=[zmin, zmax]),
                                 aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6)),
                      sliders=sliders, showlegend=True,
                      updatemenus=[
                          dict(type='buttons', direction='left', x=0.0, y=1.15, buttons=[
                              dict(label='Actual only', method='update', args=[{'visible': vis_actual}]),
                              dict(label='GR only', method='update', args=[{'visible': vis_gronly}]),
                              dict(label='Both', method='update', args=[{'visible': vis_both}]),
                          ]),
                          dict(type='buttons', showactive=False, y=1.1,
                               buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=0, redraw=True), fromcurrent=True)] )])
                      ],
                      annotations=[dict(text='Actual (blue)', x=0.01, y=0.97, xref='paper', yref='paper', showarrow=False, font=dict(color='#0077ff', size=12)),
                                   dict(text='GR (red, dashed)', x=0.17, y=0.97, xref='paper', yref='paper', showarrow=False, font=dict(color='#dd2222', size=12))])

    outdir = ROOT / 'out' / 'visualizations' / cid
    outdir.mkdir(parents=True, exist_ok=True)
    out_html = outdir / 'interactive_3d.html'
    fig.write_html(str(out_html), include_plotlyjs='cdn')
    return out_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_id', required=True)
    ap.add_argument('--zs', type=float, default=2.0)
    ap.add_argument('--rays', type=int, default=9)
    ap.add_argument('--mag', type=float, default=1000.0, help='Visual exaggeration of deflection angles')
    ap.add_argument('--y_src_frac', type=float, default=0.0, help='Source vertical offset as fraction of θE (e.g. 0.0)')
    ap.add_argument('--sphere_mode', type=str, default='wireframe', choices=['solid','wireframe','off'])
    args = ap.parse_args()

    if not PKG_OK:
        print('This visualization requires plotly and astropy. Please install them (pip install plotly astropy).')
        sys.exit(1)

    cid = args.cluster_id.lower()
    if cid not in CLASH:
        print('Unknown cluster_id')
        sys.exit(1)

    out_html = fig_for_cluster(cid, args.zs, args.rays, args.mag, args.y_src_frac, args.sphere_mode)
    print(f'Wrote {out_html}')

if __name__ == '__main__':
    main()

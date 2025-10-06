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

from scripts.lensing_utils import CLASH, alpha_fun_GR_baryons, alpha_fun_HLSP

try:
    import plotly.graph_objects as go  # type: ignore
    from astropy.io import fits  # type: ignore
    from astropy.wcs import WCS  # type: ignore
    from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore
    PKG_OK = True
except Exception:
    PKG_OK = False










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

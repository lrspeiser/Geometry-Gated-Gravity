#!/usr/bin/env python3
from __future__ import annotations
"""
2D lensing ray visualization with model toggle and deflection annotation.

- Draw a circle representing the edge of the stellar component (radius = edge_factor * theta_E_observed)
- Place a source "star" at x=-X on one side (y = y_src_frac * theta_E_observed)
- Trace a thin-lens path with a kink at x=0 for:
  • Actual (HLSP-based deflection) — blue solid
  • GR (baryons-only) — red dashed
- Toggle visibility: Actual only, GR only, Both
- Annotate the deflection angle α at the lens plane (in arcseconds) for both models

Output: out/visualizations/<cluster_id>/ray_2d.html

Usage example:
  python scripts/interactive_lensing_2d.py --cluster_id a209 --zs 2.0 --y_src_frac 1.0 --edge_factor 0.5 --mag 2000

Notes:
- α is computed in radians by the lensing utils; we convert to arcsec for the displayed annotation.
- Paths are visual exaggerations using a slope multiplier `mag` to make bends visible in arcsec coordinates.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lensing_utils import CLASH, alpha_fun_GR_baryons, alpha_fun_HLSP

import plotly.graph_objects as go  # type: ignore


def build_ray_paths(y0_arcsec: float, alpha_gr_y_rad: float, alpha_hl_y_rad: float, mag: float, x_extent: float):
    # Pre-lens: from -x_extent to 0 at constant y0
    x_pre = np.linspace(-x_extent, 0.0, 200)
    y_pre = np.full_like(x_pre, y0_arcsec)
    # Post-lens: from 0 to +x_extent with slope proportional to alpha
    x_post = np.linspace(0.0, x_extent, 400)
    y_post_gr = y0_arcsec - (alpha_gr_y_rad * mag) * x_post
    y_post_hl = y0_arcsec - (alpha_hl_y_rad * mag) * x_post
    return (x_pre, y_pre, x_post, y_post_gr, y_post_hl)


def fig_2d(cid: str, zs: float, y_src_frac: float, edge_factor: float, mag: float, x_extent_factor: float) -> Path:
    # observed theta_E
    obs_csv = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
    df = pd.read_csv(obs_csv)
    df['cluster_id'] = df['cluster_id'].str.lower()
    row = df[df['cluster_id'] == cid]
    if len(row) == 0:
        raise RuntimeError('No observed θE for this cluster')
    theta_obs = float(row.iloc[0]['theta_E_observed_arcsec'])

    if cid not in CLASH:
        raise RuntimeError('Unknown cluster_id')
    local_name, z_lens = CLASH[cid]

    # deflection functions
    alpha_gr = alpha_fun_GR_baryons(local_name, z_lens, zs)
    alpha_hl_y_fun = alpha_fun_HLSP(cid, theta_obs)
    if alpha_hl_y_fun is None:
        def alpha_hl_y_fun(theta_arcsec: float) -> float:
            return theta_arcsec / 206265.0

    # source position (one side)
    y0 = float(y_src_frac) * theta_obs

    # compute deflections (radians)
    a_gr_y = np.sign(y0) * alpha_gr(abs(y0))
    a_hl_y = alpha_hl_y_fun(y0)

    # choose extent in arcsec
    x_extent = x_extent_factor * theta_obs

    # paths
    x_pre, y_pre, x_post, y_post_gr, y_post_hl = build_ray_paths(y0, a_gr_y, a_hl_y, mag, x_extent)

    # figure
    fig = go.Figure()

    # lens plane (x=0)
    fig.add_trace(go.Scatter(x=[0,0], y=[-1.2*theta_obs, 1.2*theta_obs], mode='lines',
                             line=dict(color='#888888', dash='dot'), name='Lens plane', showlegend=False))

    # stellar edge circle
    r_edge = edge_factor * theta_obs
    circle_theta = np.linspace(0, 2*np.pi, 360)
    cx, cy = 0.0, 0.0
    fig.add_trace(go.Scatter(x=cx + r_edge*np.cos(circle_theta), y=cy + r_edge*np.sin(circle_theta), mode='lines',
                             line=dict(color='#AAAAAA'), name='Stellar edge (circle)', showlegend=True))

    # source star
    fig.add_trace(go.Scatter(x=[-x_extent], y=[y0], mode='markers+text', marker=dict(size=10, color='#ffcc00', symbol='star'),
                             text=['Source star'], textposition='bottom right', name='Source', showlegend=False))

    # Actual (HLSP): blue
    fig.add_trace(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_hl]), mode='lines',
                             line=dict(color='#0077ff', width=4), name='Actual (HLSP)'))

    # GR (baryons): red dashed
    fig.add_trace(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_gr]), mode='lines',
                             line=dict(color='#dd2222', width=4, dash='dash'), name='GR (baryons)'))

    # deflection annotations (arcsec)
    a_gr_arcsec = abs(a_gr_y) * 206265.0
    a_hl_arcsec = abs(a_hl_y) * 206265.0
    annos = [
        dict(x=0.02*x_extent, y=y0 + 0.05*theta_obs, xref='x', yref='y', text=f"α_actual ≈ {a_hl_arcsec:.2f} arcsec",
             showarrow=False, font=dict(color='#0077ff')),
        dict(x=0.02*x_extent, y=y0 - 0.10*theta_obs, xref='x', yref='y', text=f"α_GR ≈ {a_gr_arcsec:.2f} arcsec",
             showarrow=False, font=dict(color='#dd2222')),
    ]

    # visibility toggle: Actual / GR / Both
    vis_actual = [True, True, True, True, False]   # lens plane, circle, source, actual, gr
    vis_gronly = [True, True, True, False, True]
    vis_both   = [True, True, True, True, True]

    fig.update_layout(title=f"{cid}: 2D ray (Actual vs GR). α labeled in arcsec; thin-lens kink at x=0",
                      xaxis_title='x (arcsec)', yaxis_title='y (arcsec)',
                      xaxis=dict(scaleanchor='y', scaleratio=1),
                      showlegend=True,
                      updatemenus=[
                          dict(type='buttons', direction='left', x=0.0, y=1.15, buttons=[
                              dict(label='Actual only', method='update', args=[{'visible': vis_actual}, {'annotations': annos[:1]}]),
                              dict(label='GR only', method='update', args=[{'visible': vis_gronly}, {'annotations': annos[1:]}]),
                              dict(label='Both', method='update', args=[{'visible': vis_both}, {'annotations': annos}]),
                          ])
                      ],
                      annotations=annos)

    # axis ranges
    xlim = (-x_extent, x_extent)
    ylim = (-1.3*theta_obs, 1.3*theta_obs)
    fig.update_xaxes(range=xlim)
    fig.update_yaxes(range=ylim)

    outdir = ROOT / 'out' / 'visualizations' / cid
    outdir.mkdir(parents=True, exist_ok=True)
    out_html = outdir / 'ray_2d.html'
    fig.write_html(str(out_html), include_plotlyjs='cdn')
    return out_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_id', required=True)
    ap.add_argument('--zs', type=float, default=2.0)
    ap.add_argument('--y_src_frac', type=float, default=1.0, help='Source vertical offset as fraction of θE (e.g. 1.0)')
    ap.add_argument('--edge_factor', type=float, default=0.5, help='Circle radius = edge_factor * θE')
    ap.add_argument('--mag', type=float, default=2000.0, help='Visual slope magnification for bends')
    ap.add_argument('--x_extent_factor', type=float, default=2.5, help='Ray extent in x as factor of θE')
    args = ap.parse_args()

    cid = args.cluster_id.lower()
    out_html = fig_2d(cid, args.zs, args.y_src_frac, args.edge_factor, args.mag, args.x_extent_factor)
    print(f'Wrote {out_html}')


if __name__ == '__main__':
    main()

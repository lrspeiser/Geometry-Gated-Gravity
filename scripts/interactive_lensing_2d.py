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

from scripts.lensing_utils import CLASH, alpha_fun_GR_baryons, alpha_fun_HLSP, alpha_fun_GE, solve_theta_E_from_alpha

import plotly.graph_objects as go  # type: ignore
from scripts.lensing_utils import load_gold_standard_catalog


def build_ray_paths(y0_arcsec: float, alpha_gr_y_rad: float, alpha_hl_y_rad: float, alpha_ge_y_rad: float | None, mag: float, x_extent: float):
    # Pre-lens: from -x_extent to 0 at constant y0
    x_pre = np.linspace(-x_extent, 0.0, 200)
    y_pre = np.full_like(x_pre, y0_arcsec)
    # Post-lens: from 0 to +x_extent with slope proportional to alpha
    x_post = np.linspace(0.0, x_extent, 400)
    y_post_gr = y0_arcsec - (alpha_gr_y_rad * mag) * x_post
    y_post_hl = y0_arcsec - (alpha_hl_y_rad * mag) * x_post
    y_post_ge = None if alpha_ge_y_rad is None else y0_arcsec - (alpha_ge_y_rad * mag) * x_post
    return (x_pre, y_pre, x_post, y_post_gr, y_post_hl, y_post_ge)


def fig_2d(cid: str, zs: float, y_src_frac: float, edge_factor: float, mag: float, x_extent_factor: float,
           ge_gamma1: float, ge_gamma2: float, ge_a: float, ge_b: float, ge_d: float,
           ge_Rd_kpc: float, ge_Rscale_kpc: float) -> Path:
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
    alpha_ge_fun = alpha_fun_GE(local_name, z_lens, zs, a=ge_a, b=ge_b, d=ge_d,
                                gamma1=ge_gamma1, gamma2=ge_gamma2,
                                Rd_kpc=ge_Rd_kpc, R_scale_kpc=ge_Rscale_kpc,
                                beta_clip=(1.0, 5.0))

    # source position (one side)
    y0 = float(y_src_frac) * theta_obs

    # compute deflections (radians)
    a_gr_y = np.sign(y0) * alpha_gr(abs(y0))
    a_hl_y = alpha_hl_y_fun(y0)
    a_ge_y = None if alpha_ge_fun is None else np.sign(y0) * alpha_ge_fun(abs(y0))

    # choose extent in arcsec
    x_extent = x_extent_factor * theta_obs

    # paths
    x_pre, y_pre, x_post, y_post_gr, y_post_hl, y_post_ge = build_ray_paths(y0, a_gr_y, a_hl_y, a_ge_y, mag, x_extent)

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

    # Accepted reference: purple (α_ref = θE_accepted)
    alpha_acc_rad = (theta_obs / 206265.0)
    y_post_acc = y0 - (alpha_acc_rad * mag) * x_post
    fig.add_trace(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_acc]), mode='lines',
                             line=dict(color='#6f42c1', width=3, dash='dashdot'), name='Accepted (θE ref)'))

    # GE (our formula): green
    if y_post_ge is not None:
        fig.add_trace(
            go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_ge]), mode='lines',
                       line=dict(color='#00aa55', width=4, dash='dot'), name='GE (ours)')
        )

    # deflection annotations (arcsec)
    a_gr_arcsec = abs(a_gr_y) * 206265.0
    a_hl_arcsec = abs(a_hl_y) * 206265.0
    a_ge_arcsec = None if a_ge_y is None else abs(a_ge_y) * 206265.0
    # Predict θE for GR/GE against accepted θE (simple bisection on α(θ)=θ)
    thetaE_gr = solve_theta_E_from_alpha(lambda th: np.sign(y0)*alpha_gr(abs(th)) if alpha_gr else 0.0,
                                         theta_obs, 0.2*theta_obs, 2.0*theta_obs) if alpha_gr else None
    thetaE_ge = solve_theta_E_from_alpha(lambda th: np.sign(y0)*alpha_ge_fun(abs(th)) if alpha_ge_fun else 0.0,
                                         theta_obs, 0.2*theta_obs, 2.0*theta_obs) if alpha_ge_fun else None
    annos = [
        dict(x=0.02*x_extent, y=y0 + 0.09*theta_obs, xref='x', yref='y', text=f"Accepted θE ≈ {theta_obs:.2f} arcsec",
             showarrow=False, font=dict(color='#6f42c1')),
        dict(x=0.02*x_extent, y=y0 + 0.05*theta_obs, xref='x', yref='y', text=f"α_actual ≈ {a_hl_arcsec:.2f} arcsec",
             showarrow=False, font=dict(color='#0077ff')),
        dict(x=0.02*x_extent, y=y0 - 0.06*theta_obs, xref='x', yref='y', text=f"α_GR ≈ {a_gr_arcsec:.2f} arcsec; θE_GR {(thetaE_gr if thetaE_gr else float('nan')):.2f}",
             showarrow=False, font=dict(color='#dd2222')),
    ]
    if a_ge_arcsec is not None:
        annos.append(
            dict(x=0.02*x_extent, y=y0 - 0.16*theta_obs, xref='x', yref='y', text=f"α_GE ≈ {a_ge_arcsec:.2f} arcsec; θE_GE {(thetaE_ge if thetaE_ge else float('nan')):.2f}",
                 showarrow=False, font=dict(color='#00aa55'))
        )

    # visibility toggle: Actual / GR / Both
    # Build visibility lists including optional GE trace
    has_ge = (y_post_ge is not None)
    # Index layout: [lens, circle, source, actual, gr, accepted, ge?]
    if has_ge:
        vis_actual = [True, True, True, True, False, True, False]
        vis_gronly = [True, True, True, False, True, True, False]
        vis_accepted = [True, True, True, False, False, True, False]
        vis_geonly = [True, True, True, False, False, True, True]
        vis_all    = [True, True, True, True, True, True, True]
    else:
        vis_actual = [True, True, True, True, False, True]
        vis_gronly = [True, True, True, False, True, True]
        vis_accepted = [True, True, True, False, False, True]
        vis_all    = [True, True, True, True, True, True]

    fig.update_layout(title=f"{cid}: 2D ray (Actual vs GR vs Accepted vs GE). α and θE labels; thin-lens kink at x=0",
                      xaxis_title='x (arcsec)', yaxis_title='y (arcsec)',
                      xaxis=dict(scaleanchor='y', scaleratio=1),
                      showlegend=True,
                      updatemenus=[
                          dict(type='buttons', direction='left', x=0.0, y=1.15, buttons=(
                              [dict(label='Actual only', method='update', args=[{'visible': vis_actual}, {'annotations': annos}]),
                               dict(label='GR only', method='update', args=[{'visible': vis_gronly}, {'annotations': annos}]),
                               dict(label='Accepted only', method='update', args=[{'visible': vis_accepted}, {'annotations': annos}])]
                              + ([dict(label='GE only', method='update', args=[{'visible': vis_geonly}, {'annotations': annos}] )] if has_ge else [])
                              + [dict(label='All', method='update', args=[{'visible': vis_all}, {'annotations': annos}])]
                          ))
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


def build_cluster_bundle(cid: str, zs: float, y_src_frac: float, edge_factor: float, mag: float, x_extent_factor: float,
                         ge_params: dict):
    # observed theta_E and z_lens
    gold = load_gold_standard_catalog()
    accepted = gold.get(cid, {}).get('accepted', None)
    z_lens = None
    # prefer CLASH table if present
    obs_csv = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
    theta_obs = None
    if obs_csv.exists():
        df = pd.read_csv(obs_csv)
        df['cluster_id'] = df['cluster_id'].str.lower()
        row = df[df['cluster_id'] == cid]
        if len(row) > 0:
            theta_obs = float(row.iloc[0]['theta_E_observed_arcsec'])
            z_lens = float(row.iloc[0]['z_lens']) if 'z_lens' in row.columns else None
    # fallback to gold-standard values
    if theta_obs is None and accepted is not None:
        theta_obs = float(accepted.get('theta_E_arcsec', np.nan))
    if z_lens is None:
        z_lens = float(gold.get(cid, {}).get('z_lens', np.nan))

    # sanity
    if not np.isfinite(theta_obs):
        raise RuntimeError(f'Missing observed/accepted θE for {cid}')
    if not np.isfinite(z_lens):
        raise RuntimeError(f'Missing z_lens for {cid}')

    # deflection functions
    local_name = CLASH.get(cid, (None, None))[0]
    alpha_gr = alpha_fun_GR_baryons(local_name, z_lens, zs) if local_name is not None else None
    alpha_hl_y_fun = alpha_fun_HLSP(cid, theta_obs)
    alpha_ge_fun = alpha_fun_GE(local_name, z_lens, zs, **ge_params) if local_name is not None else None

    # source position
    y0 = float(y_src_frac) * theta_obs

    # compute deflections (rad)
    a_gr_y = None if alpha_gr is None else np.sign(y0) * alpha_gr(abs(y0))
    a_hl_y = None if alpha_hl_y_fun is None else alpha_hl_y_fun(y0)
    a_ge_y = None if alpha_ge_fun is None else np.sign(y0) * alpha_ge_fun(abs(y0))

    # extent and paths
    x_extent = x_extent_factor * theta_obs
    x_pre = np.linspace(-x_extent, 0.0, 200)
    y_pre = np.full_like(x_pre, y0)
    x_post = np.linspace(0.0, x_extent, 400)

    traces = []
    # lens plane
    traces.append(go.Scatter(x=[0,0], y=[-1.2*theta_obs, 1.2*theta_obs], mode='lines',
                             line=dict(color='#888888', dash='dot'), name='Lens plane', showlegend=False))
    # stellar circle
    r_edge = edge_factor * theta_obs
    circle_theta = np.linspace(0, 2*np.pi, 360)
    traces.append(go.Scatter(x=r_edge*np.cos(circle_theta), y=r_edge*np.sin(circle_theta), mode='lines',
                             line=dict(color='#AAAAAA'), name='Stellar edge (circle)', showlegend=True))
    # source
    traces.append(go.Scatter(x=[-x_extent], y=[y0], mode='markers+text', marker=dict(size=10, color='#ffcc00', symbol='star'),
                             text=['Source star'], textposition='bottom right', name='Source', showlegend=False))

    # Actual
    if a_hl_y is not None:
        y_post_hl = y0 - (a_hl_y * mag) * x_post
        traces.append(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_hl]), mode='lines',
                                 line=dict(color='#0077ff', width=4), name='Actual (HLSP)'))
    else:
        traces.append(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#0077ff', width=4), name='Actual (HLSP)'))

    # GR
    if a_gr_y is not None:
        y_post_gr = y0 - (a_gr_y * mag) * x_post
        traces.append(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_gr]), mode='lines',
                                 line=dict(color='#dd2222', width=4, dash='dash'), name='GR (baryons)'))
    else:
        traces.append(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#dd2222', width=4, dash='dash'), name='GR (baryons)'))

    # Accepted path
    alpha_acc_rad = ( (accepted['theta_E_arcsec'] if accepted else theta_obs) / 206265.0 )
    y_post_acc = y0 - (alpha_acc_rad * mag) * x_post
    traces.append(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_acc]), mode='lines',
                             line=dict(color='#6f42c1', width=3, dash='dashdot'), name='Accepted (θE ref)'))

    # GE
    if a_ge_y is not None:
        y_post_ge = y0 - (a_ge_y * mag) * x_post
        traces.append(go.Scatter(x=np.concatenate([x_pre, x_post]), y=np.concatenate([y_pre, y_post_ge]), mode='lines',
                                 line=dict(color='#00aa55', width=4, dash='dot'), name='GE (ours)'))
    else:
        traces.append(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#00aa55', width=4, dash='dot'), name='GE (ours)'))

    # annotations
    annos = []
    acc_te = (accepted['theta_E_arcsec'] if accepted else theta_obs)
    annos.append(dict(x=0.02*x_extent, y=y0 + 0.08*theta_obs, xref='x', yref='y',
                      text=f"Accepted θE(zs={accepted['zs'] if accepted else zs:.1f}) ≈ {acc_te:.1f} ± {accepted.get('sigma', 0) if accepted else 0} arcsec",
                      showarrow=False, font=dict(color='#444')))
    if a_hl_y is not None:
        annos.append(dict(x=0.02*x_extent, y=y0 + 0.03*theta_obs, xref='x', yref='y',
                          text=f"α_actual ≈ {abs(a_hl_y)*206265.0:.2f} arcsec", showarrow=False, font=dict(color='#0077ff')))
    if a_gr_y is not None:
        annos.append(dict(x=0.02*x_extent, y=y0 - 0.07*theta_obs, xref='x', yref='y',
                          text=f"α_GR ≈ {abs(a_gr_y)*206265.0:.2f} arcsec", showarrow=False, font=dict(color='#dd2222')))
    if a_ge_y is not None:
        annos.append(dict(x=0.02*x_extent, y=y0 - 0.17*theta_obs, xref='x', yref='y',
                          text=f"α_GE ≈ {abs(a_ge_y)*206265.0:.2f} arcsec", showarrow=False, font=dict(color='#00aa55')))

    # visibility map for this cluster: order [lens, circle, source, actual, gr, accepted, ge]
    vis_all = [True, True, True, True, True, True, True]
    vis_actual = [True, True, True, True, False, True, False]
    vis_gr = [True, True, True, False, True, True, False]
    vis_acc = [True, True, True, False, False, True, False]
    vis_ge = [True, True, True, False, False, True, True]

    # axis extent for this cluster
    xlim = (-x_extent, x_extent)
    ylim = (-1.3*theta_obs, 1.3*theta_obs)

    meta = {'cid': cid, 'z_lens': z_lens, 'theta_obs': theta_obs, 'xlim': xlim, 'ylim': ylim, 'annos': annos,
            'vis': {'all': vis_all, 'actual': vis_actual, 'gr': vis_gr, 'accepted': vis_acc, 'ge': vis_ge}}
    return traces, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_id', required=True)
    ap.add_argument('--zs', type=float, default=2.0)
    ap.add_argument('--y_src_frac', type=float, default=1.0, help='Source vertical offset as fraction of θE (e.g. 1.0)')
    ap.add_argument('--edge_factor', type=float, default=0.5, help='Circle radius = edge_factor * θE')
    ap.add_argument('--mag', type=float, default=2000.0, help='Visual slope magnification for bends')
    ap.add_argument('--x_extent_factor', type=float, default=2.5, help='Ray extent in x as factor of θE')
    # GE params
    ap.add_argument('--ge_gamma1', type=float, default=0.6)
    ap.add_argument('--ge_gamma2', type=float, default=0.2)
    ap.add_argument('--ge_a', type=float, default=1.0)
    ap.add_argument('--ge_b', type=float, default=0.5)
    ap.add_argument('--ge_d', type=float, default=0.5)
    ap.add_argument('--ge_Rd_kpc', type=float, default=1000.0)
    ap.add_argument('--ge_Rscale_kpc', type=float, default=100.0)
    ap.add_argument('--preset', type=str, choices=['single','gold'], default='single')
    ap.add_argument('--cluster_list', type=str, default=None, help='Comma-separated cluster ids to include in multi mode')
    args = ap.parse_args()

    if args.preset == 'single':
        cid = args.cluster_id.lower()
        out_html = fig_2d(cid, args.zs, args.y_src_frac, args.edge_factor, args.mag, args.x_extent_factor,
                          args.ge_gamma1, args.ge_gamma2, args.ge_a, args.ge_b, args.ge_d,
                          args.ge_Rd_kpc, args.ge_Rscale_kpc)
        print(f'Wrote {out_html}')
        return

    # Multi-cluster gold-standard dropdown
    gold = load_gold_standard_catalog()
    if args.cluster_list:
        cluster_ids = [c.strip().lower() for c in args.cluster_list.split(',') if c.strip()]
    else:
        # default: use gold list order prioritizing best-characterized
        cluster_ids = ['macs0416','a370','a2744','macs0717','rxj1347','a1689']
    # Build bundles
    ge_params = dict(gamma1=args.ge_gamma1, gamma2=args.ge_gamma2, a=args.ge_a, b=args.ge_b, d=args.ge_d,
                     Rd_kpc=args.ge_Rd_kpc, R_scale_kpc=args.ge_Rscale_kpc)

    all_traces = []
    metas = []
    for cid in cluster_ids:
        try:
            traces, meta = build_cluster_bundle(cid, args.zs, args.y_src_frac, args.edge_factor, args.mag, args.x_extent_factor, ge_params)
            all_traces.extend(traces)
            metas.append(meta)
        except Exception as e:
            # Add empty placeholders to keep indexing simple
            empty = [go.Scatter(x=[], y=[], mode='lines') for _ in range(6)]
            all_traces.extend(empty)
            metas.append({'cid': cid, 'z_lens': np.nan, 'theta_obs': np.nan,
                          'xlim': (-1,1), 'ylim': (-1,1), 'annos': [dict(text=f"{cid}: {e}", x=0, y=0, showarrow=False)],
                          'vis': {'all':[False]*6, 'actual':[False]*6, 'gr':[False]*6, 'ge':[False]*6}})

    # Compute global axes
    xmin = min(m['xlim'][0] for m in metas)
    xmax = max(m['xlim'][1] for m in metas)
    ymin = min(m['ylim'][0] for m in metas)
    ymax = max(m['ylim'][1] for m in metas)

    fig = go.Figure(data=all_traces)

    # Initial: first cluster, all models
    init_vis = []
    for i, m in enumerate(metas):
        vis = m['vis']['all'] if i == 0 else [False]*6
        init_vis.extend(vis)
    for i in range(len(all_traces)):
        fig.data[i].visible = init_vis[i]

    # Title and axes
    first = metas[0]
    cname = gold.get(first['cid'], {}).get('name', first['cid'])
    fig.update_layout(title=f"Gold clusters: {cname} (z_l={first['z_lens']:.3f}) — 2D ray (Actual/GR/GE)",
                      xaxis_title='x (arcsec)', yaxis_title='y (arcsec)',
                      xaxis=dict(scaleanchor='y', scaleratio=1, range=[xmin, xmax]),
                      showlegend=True,
                      annotations=first['annos'])
    fig.update_yaxes(range=[ymin, ymax])

    # Dropdown options per cluster+mode
    buttons = []
    for idx, m in enumerate(metas):
        base = idx*6
        # visibility builders
        def vis_for(mode):
            vis = []
            for j in range(len(metas)):
                setv = metas[j]['vis'][mode] if j == idx else [False]*6
                vis.extend(setv)
            return vis
        name = gold.get(m['cid'], {}).get('name', m['cid'])
        for mode, label in [('all','All'), ('actual','Actual only'), ('gr','GR only'), ('accepted','Accepted only'), ('ge','GE only')]:
            buttons.append(dict(label=f"{name}: {label}", method='update',
                                args=[{'visible': vis_for(mode)},
                                      {'annotations': m['annos'], 'title': f"Gold clusters: {name} (z_l={m['z_lens']:.3f}) — 2D ray (Actual/GR/GE)"}]))

    fig.update_layout(updatemenus=[dict(type='dropdown', x=0.0, y=1.19, direction='down',
                                        buttons=buttons, showactive=True)])

    outdir = ROOT / 'out' / 'visualizations' / 'gold'
    outdir.mkdir(parents=True, exist_ok=True)
    out_html = outdir / 'ray_2d_gold.html'
    fig.write_html(str(out_html), include_plotlyjs='cdn')
    print(f'Wrote {out_html}')


if __name__ == '__main__':
    main()

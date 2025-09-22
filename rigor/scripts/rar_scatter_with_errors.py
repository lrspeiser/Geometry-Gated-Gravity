# -*- coding: utf-8 -*-
"""
rigor/scripts/rar_scatter_with_errors.py

Compute orthogonal scatter diagnostics for the RAR by comparing observed points
with the model curve and (optionally) deconvolving measurement errors.

Inputs (JSON files; flexible structure):
- --rar_obs: should contain arrays of (x_obs, y_obs) and optional per-point
  uncertainties (sigma_x, sigma_y). If the structure is a dict containing a
  key 'points' with a list of objects {'x':..,'y':..,'sx':..,'sy':..}, those are
  used. Otherwise, the script attempts to use top-level arrays.
- --rar_model: should contain arrays for the model curve (x_model, y_model). If
  the structure is a dict containing a key 'curve' with arrays {'x':..,'y':..},
  those are used. Otherwise, the script attempts to use top-level arrays.
- --out: output JSON with summary statistics; a PNG residual histogram is saved
  next to it.

Outputs:
- out JSON fields: sigma_perp_obs, mean_sigma_meas (if given), sigma_intrinsic
  (quadrature), n_points
- residual_hist.png in the same directory
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _load_points(path: Path):
    data = json.loads(path.read_text())
    # Try flexible structures
    def _get(arrname, default=None):
        return np.asarray(data.get(arrname, default), float) if arrname in data else default
    if 'points' in data:
        pts = data['points']
        x = np.asarray([p.get('x') for p in pts], float)
        y = np.asarray([p.get('y') for p in pts], float)
        sx = np.asarray([p.get('sx', np.nan) for p in pts], float)
        sy = np.asarray([p.get('sy', np.nan) for p in pts], float)
        return x, y, sx, sy
    # top-level arrays
    x = _get('x') or _get('x_obs')
    y = _get('y') or _get('y_obs')
    sx = _get('sx') or _get('sigma_x')
    sy = _get('sy') or _get('sigma_y')
    if x is None or y is None:
        raise ValueError(f'Could not parse x/y from {path}')
    return np.asarray(x, float), np.asarray(y, float), (np.asarray(sx, float) if sx is not None else None), (np.asarray(sy, float) if sy is not None else None)


def _load_curve(path: Path):
    data = json.loads(path.read_text())
    if 'curve' in data:
        xx = np.asarray(data['curve'].get('x'), float)
        yy = np.asarray(data['curve'].get('y'), float)
        return xx, yy
    xx = data.get('x_model') or data.get('x')
    yy = data.get('y_model') or data.get('y')
    if xx is None or yy is None:
        raise ValueError(f'Could not parse model curve from {path}')
    return np.asarray(xx, float), np.asarray(yy, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rar_obs', required=True)
    ap.add_argument('--rar_model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    x_obs, y_obs, sx, sy = _load_points(Path(args.rar_obs))
    x_mod, y_mod = _load_curve(Path(args.rar_model))

    # Interpolate model at observed x
    y_mod_on_obs = np.interp(x_obs, x_mod, y_mod)
    dy = y_obs - y_mod_on_obs

    # Orthogonal scatter approximation: if relation is close to y=f(x), use dy
    sigma_perp = float(np.sqrt(np.mean(dy**2)))

    # Measurement error proxy (if sx/sy present): average sy or combine
    if sy is not None and np.all(np.isfinite(sy)):
        mean_sigma_meas = float(np.nanmean(np.abs(sy)))
    else:
        mean_sigma_meas = None

    sigma_intrinsic = None
    if mean_sigma_meas is not None:
        s2 = max(0.0, sigma_perp**2 - mean_sigma_meas**2)
        sigma_intrinsic = float(np.sqrt(s2))

    out = {'sigma_perp_obs': sigma_perp,
           'mean_sigma_meas': mean_sigma_meas,
           'sigma_intrinsic': sigma_intrinsic,
           'n_points': int(len(x_obs))}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # residual histogram
    try:
        plt.figure(figsize=(5,3))
        plt.hist(dy, bins=40, alpha=0.8)
        plt.xlabel('Residual (y_obs - y_model)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path.with_name('residual_hist.png'), dpi=140)
        plt.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()

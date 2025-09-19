# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, asdict
from typing import Tuple, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt  # optional
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ----------------------------- IO helpers -----------------------------

def read_plik_lite_tt(plik_lite_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read Planck plik_lite TT bandpowers and covariance from the _external directory.

    Expects files:
      - cl_cmb_plik_v22.dat (columns: ell_eff, C_ell[µK^2], sigma[µK^2])
      - c_matrix_plik_v22.dat (covariance for the binned vector; text or binary)
    Optionally present:
      - blmin.dat, blmax.dat, bweight.dat (for reference only)
    """
    def _find(fname: str) -> str:
        p = os.path.join(plik_lite_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
        return p

    p_cl = _find('cl_cmb_plik_v22.dat')
    data = np.loadtxt(p_cl)
    if data.shape[1] < 3:
        raise ValueError(f"Unexpected format in {p_cl}; need 3 columns: ell, Cl, sigma")
    ell = data[:, 0].astype(float)
    y = data[:, 1].astype(float)  # µK^2
    ysig = data[:, 2].astype(float)

    # Covariance: attempt text first, then binary fallback
    p_cov = _find('c_matrix_plik_v22.dat')
    cov = None
    try:
        cov = np.loadtxt(p_cov)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov not square")
    except Exception:
        # Binary fallback: try float64 packed matrix
        try:
            raw = np.fromfile(p_cov, dtype=np.float64)
            N = len(y)
            if raw.size < N*N:
                raise ValueError(f"Binary covariance too small: {raw.size} < {N*N}")
            # Try last N*N chunk (some files may have headers)
            cov = raw[-(N*N):].reshape((N, N))
        except Exception as e:
            raise RuntimeError(f"Could not read covariance matrix from {p_cov}: {e}")

    if cov.shape[0] != len(y) or cov.shape[1] != len(y):
        raise ValueError(f"Cov shape {cov.shape} does not match data length {len(y)}")

    return ell, y, ysig, cov


# ----------------------------- Templates -----------------------------

def gaussian_smooth_vec(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth y(x) in x-domain with width sigma (same units as x).
    Uses a simple FFT-free kernel with finite window for robustness.
    """
    if sigma <= 0:
        return y.copy()
    # Build kernel on an x-grid centered at 0 with window ~ 5 sigma
    dx = np.median(np.diff(x)) if len(x) > 1 else 1.0
    half = max(1, int(round(5.0 * sigma / max(dx, 1e-9))))
    offsets = np.arange(-half, half+1)
    kern = np.exp(-0.5 * (offsets*dx/sigma)**2)
    kern /= np.sum(kern)
    # Convolve with reflection at boundaries
    z = np.zeros_like(y)
    for i in range(len(y)):
        idx = i + offsets
        idx = np.clip(idx, 0, len(y)-1)
        z[i] = np.sum(kern * y[idx])
    return z


def template_lens(ell: np.ndarray, y: np.ndarray, sigma_ell: float = 100.0) -> np.ndarray:
    """Lensing-like smoothing: template F = S(y) - y, where S is Gaussian smoothing in ell-space."""
    y_s = gaussian_smooth_vec(ell, y, sigma_ell)
    return (y_s - y)


def gate_low(ell: np.ndarray, ell0: float, width: float) -> np.ndarray:
    return 0.5*(1.0 + np.tanh((ell0 - ell)/max(width, 1e-9)))


def gate_high(ell: np.ndarray, ell0: float, width: float) -> np.ndarray:
    return 0.5*(1.0 + np.tanh((ell - ell0)/max(width, 1e-9)))


def template_gk(ell: np.ndarray, y: np.ndarray, ell0: float = 80.0, width: float = 40.0, power: float = -1.0) -> np.ndarray:
    """Low-ell gated gravitational kernel: F = y * G_low(ell) * (ell/max(ell0,1))**power"""
    G = gate_low(ell, ell0, width)
    scale = np.power(ell/ max(ell0, 1.0), power)
    scale = np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    return y * G * scale


def template_vea(ell: np.ndarray, y: np.ndarray, ell0: float = 800.0, width: float = 150.0, alpha: float = 0.15) -> np.ndarray:
    """High-ell void-envelope amplification: F = y * G_high(ell) * (ell/ell0)**alpha"""
    G = gate_high(ell, ell0, width)
    scale = np.power(ell/ max(ell0, 1.0), alpha)
    scale = np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    return y * G * scale


# ----------------------------- Stats -----------------------------

def sigma_amplitude(F: np.ndarray, cov: np.ndarray) -> float:
    """Return 1-sigma bound for amplitude A of template F: sigma_A = 1/sqrt(F^T C^{-1} F)."""
    # Solve C x = F  => F^T x = F^T C^{-1} F
    x = np.linalg.solve(cov, F)
    denom = float(np.dot(F, x))
    if denom <= 0:
        # Numerical issues or non-PD covariance; fall back to pseudo-inverse
        try:
            cov_inv = np.linalg.pinv(cov)
            denom = float(F @ cov_inv @ F)
        except Exception:
            raise RuntimeError("Covariance not positive-definite and pinv failed")
    if denom <= 0:
        raise RuntimeError("Non-positive quadratic form; cannot define sigma_A")
    return 1.0/np.sqrt(denom)


def peak_positions(ell: np.ndarray, y: np.ndarray, nmax: int = 5) -> Dict:
    """Crude peak-finder: return indices and ell of up-to nmax local maxima in a smoothed y."""
    ys = gaussian_smooth_vec(ell, y, sigma=50.0)
    idxs = []
    for i in range(1, len(ys)-1):
        if ys[i] > ys[i-1] and ys[i] > ys[i+1]:
            idxs.append(i)
    if not idxs:
        return {"n": 0}
    # take top n peaks by height
    idxs_sorted = sorted(idxs, key=lambda i: ys[i], reverse=True)[:nmax]
    idxs_sorted = sorted(idxs_sorted)
    ell_peaks = [float(ell[i]) for i in idxs_sorted]
    heights = [float(ys[i]) for i in idxs_sorted]
    spacings = [float(ell_peaks[i+1]-ell_peaks[i]) for i in range(len(ell_peaks)-1)] if len(ell_peaks) > 1 else []
    return {
        "n": len(ell_peaks),
        "ell_peaks": ell_peaks,
        "heights": heights,
        "spacings": spacings,
    }


# ----------------------------- Main runner -----------------------------

def run_mode(plik_lite_dir: str, mode: str, out_dir: str, **kwargs):
    ell, y, ysig, cov = read_plik_lite_tt(plik_lite_dir)

    # Build template
    if mode == 'lens':
        F = template_lens(ell, y, sigma_ell=float(kwargs.get('lens_sigma', 100.0)))
    elif mode == 'gk':
        F = template_gk(
            ell, y,
            ell0=float(kwargs.get('gk_l0', 80.0)),
            width=float(kwargs.get('gk_width', 40.0)),
            power=float(kwargs.get('gk_power', -1.0)),
        )
    elif mode == 'vea':
        F = template_vea(
            ell, y,
            ell0=float(kwargs.get('vea_l0', 800.0)),
            width=float(kwargs.get('vea_width', 150.0)),
            alpha=float(kwargs.get('vea_alpha', 0.15)),
        )
    else:
        raise ValueError("mode must be one of: lens, gk, vea")

    # Compute 1-sigma amplitude bound
    sigA = sigma_amplitude(F, cov)
    A95 = 1.96 * sigA

    os.makedirs(out_dir, exist_ok=True)

    # Write template CSV
    csv_path = os.path.join(out_dir, f"cmb_templates_{mode}.csv")
    arr = np.column_stack([
        ell, y, F, (y + sigA*F), (y - sigA*F)
    ])
    header = "ell_eff,Cl_uK2,template_F,Cl_plus_1sigma,Cl_minus_1sigma"
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")

    # Envelopes JSON
    meta = {
        "mode": mode,
        "nbins": int(len(ell)),
        "sigma_A": float(sigA),
        "A95": float(A95),
        "params": {
            "lens_sigma": float(kwargs.get('lens_sigma', 100.0)) if mode=='lens' else None,
            "gk_l0": float(kwargs.get('gk_l0', 80.0)) if mode=='gk' else None,
            "gk_width": float(kwargs.get('gk_width', 40.0)) if mode=='gk' else None,
            "gk_power": float(kwargs.get('gk_power', -1.0)) if mode=='gk' else None,
            "vea_l0": float(kwargs.get('vea_l0', 800.0)) if mode=='vea' else None,
            "vea_width": float(kwargs.get('vea_width', 150.0)) if mode=='vea' else None,
            "vea_alpha": float(kwargs.get('vea_alpha', 0.15)) if mode=='vea' else None,
        }
    }
    with open(os.path.join(out_dir, f"cmb_envelope_{mode}.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    # Peak metrics JSON
    peaks = peak_positions(ell, y, nmax=5)
    with open(os.path.join(out_dir, "cmb_peaks_metrics.json"), 'w') as f:
        json.dump(peaks, f, indent=2)

    # Optional plot
    if HAVE_MPL:
        try:
            fig, ax = plt.subplots(figsize=(9,5))
            ax.plot(ell, y, color='k', lw=1.2, label='Planck TT (binned)')
            ax.plot(ell, y + sigA*F, color='tab:blue', alpha=0.7, lw=1.0, label='+1σ envelope')
            ax.plot(ell, y - sigA*F, color='tab:orange', alpha=0.7, lw=1.0, label='-1σ envelope')
            ax.set_xlabel(r'$\ell_{\rm eff}$')
            ax.set_ylabel(r'$C_\ell\;[\mu\mathrm{K}^2]$')
            ax.set_title(f"Envelope test: {mode} (σ_A={sigA:.3g}, 95%={A95:.3g})")
            ax.legend(loc='best')
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"cmb_envelope_{mode}.png"), dpi=140)
            plt.close(fig)
        except Exception:
            pass

    print(f"Mode={mode}: sigma_A={sigA:.4g}, A95={A95:.4g}, nbins={len(ell)}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Planck plik_lite TT envelope tests (clean-slate late-time imprints)")
    ap.add_argument('--plik_lite_dir', required=True, help='Path to plik_lite _external directory (contains cl_cmb_plik_v22.dat, c_matrix_plik_v22.dat, etc.)')
    ap.add_argument('--out_dir', required=True, help='Where to write outputs (JSON/CSV/PNG)')
    ap.add_argument('--mode', choices=['lens','gk','vea'], required=True, help='Template mode: lens (smoothing), gk (low-ℓ gated kernel), vea (high-ℓ void envelope)')
    # lens
    ap.add_argument('--lens_sigma', type=float, default=100.0, help='Gaussian smoothing width in ℓ for lensing-like template')
    # gk
    ap.add_argument('--gk_l0', type=float, default=80.0, help='Low-ℓ gate center (transition)')
    ap.add_argument('--gk_width', type=float, default=40.0, help='Low-ℓ gate width')
    ap.add_argument('--gk_power', type=float, default=-1.0, help='Power-law index below ℓ0: F ~ (ℓ/ℓ0)^power')
    # vea
    ap.add_argument('--vea_l0', type=float, default=800.0, help='High-ℓ gate center (transition)')
    ap.add_argument('--vea_width', type=float, default=150.0, help='High-ℓ gate width')
    ap.add_argument('--vea_alpha', type=float, default=0.15, help='High-ℓ power index: F ~ (ℓ/ℓ0)^alpha')
    return ap


if __name__ == '__main__':
    ap = build_argparser()
    args = ap.parse_args()

    run_mode(
        plik_lite_dir=args.plik_lite_dir,
        mode=args.mode,
        out_dir=args.out_dir,
        lens_sigma=args.lens_sigma,
        gk_l0=args.gk_l0, gk_width=args.gk_width, gk_power=args.gk_power,
        vea_l0=args.vea_l0, vea_width=args.vea_width, vea_alpha=args.vea_alpha,
    )

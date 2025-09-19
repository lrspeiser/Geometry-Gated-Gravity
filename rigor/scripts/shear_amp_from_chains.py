# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, numpy as np, pandas as pd
from pathlib import Path

# --------- small helpers ---------
def _looks_like_header(path: Path) -> bool:
    try:
        with path.open('r', errors='ignore') as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                toks = ln.split()
                # If any token contains a letter or underscore, likely a header
                return any(any(ch.isalpha() or ch == '_' for ch in t) for t in toks)
    except Exception:
        pass
    return False

def _read_commented_header_names(path: Path) -> list[str]|None:
    try:
        with path.open('r', errors='ignore') as f:
            for ln in f:
                if not ln.strip():
                    continue
                if ln.startswith('#'):
                    hdr = ln[1:].strip()
                    if any(ch.isalpha() or ch == '_' for ch in hdr):
                        # split on any whitespace or tabs
                        return hdr.split()
                    else:
                        return None
                else:
                    # first non-empty line is data, not header
                    return None
    except Exception:
        return None


def _load_chain(path: Path, names_path: Path|None=None) -> pd.DataFrame:
    """
    Robustly read a CosmoMC/GetDist-style chain. Handles:
      - optional header row (commented or uncommented)
      - optional .paramnames file with 'name \t latex' rows
      - optional 'weight' column; if absent, uses unit weights
    """
    # Try to infer column names from .paramnames if provided
    colnames = None
    if names_path and names_path.exists():
        try:
            nn = pd.read_csv(names_path, sep=r'\s+', header=None, comment='#', engine='python')
            # first column are the machine-readable names
            colnames = nn.iloc[:,0].astype(str).tolist()
        except Exception:
            colnames = None

    # If still unknown, see if the first line is a commented header with names
    if colnames is None:
        commented_names = _read_commented_header_names(path)
        if commented_names:
            colnames = commented_names

    # Decide whether the file has an uncommented header row
    has_header = _looks_like_header(path) if (colnames is None) else False

    # Read chain as whitespace-delimited; skip commented lines
    if has_header:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=0, engine='python')
    else:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None if colnames is None else 0, names=colnames, engine='python')

    # If there are more columns than names, drop extras; if fewer, pad
    if colnames is not None and df.shape[1] != len(colnames):
        n = min(df.shape[1], len(colnames))
        df = df.iloc[:, :n]
        df.columns = colnames[:n]

    # Normalize column names to strings, then lowercase for robust lookup
    df.columns = [str(c) for c in df.columns]
    orig = {c.lower(): c for c in df.columns}
    df.columns = [c.lower() for c in df.columns]

    # Normalize a weight column
    if 'weight' not in df.columns and 'weights' in df.columns:
        df = df.rename(columns={'weights':'weight'})
    if 'weight' not in df.columns:
        df['weight'] = 1.0

    return df

def _pick(df: pd.DataFrame, *cands: str) -> str|None:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _weighted_percentiles(x: np.ndarray, w: np.ndarray, qs=(16,50,84)):
    m = np.isfinite(x) & np.isfinite(w) & (w>0)
    if not np.any(m):
        return [float('nan')]*len(qs)
    x, w = x[m], w[m]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return [float(np.interp(q/100.0, cdf, x)) for q in qs]

def s8_from(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (S8, weights). If an S8-like column exists, use it.
    Otherwise compute S8 = sigma8 * sqrt(omegam/0.3).
    """
    w = df['weight'].to_numpy()
    # common spellings
    s8col = _pick(df, 's8', 's_8', 's8m', 's8z', 's8mz', 's8matter')
    if s8col:
        return df[s8col].to_numpy(), w

    sig8 = _pick(df, 'sigma8', 'sig8', 'sigma_8')
    om   = _pick(df, 'omegam', 'omega_m', 'omega', 'om')
    if not (sig8 and om):
        # cannot compute S8
        return np.full(len(df), np.nan), w
    s8 = df[sig8].to_numpy() * np.sqrt(np.clip(df[om].to_numpy()/0.3, 1e-12, None))
    return s8, w

def summarize_one_chain(path: Path, paramnames: Path|None, s8_ref: float|None, phi_amp_json: Path|None) -> dict:
    df = _load_chain(path, paramnames)
    s8, w = s8_from(df)
    p16, p50, p84 = _weighted_percentiles(s8, w, (16,50,84))
    out = {
        'chain': str(path),
        'n_samples': int(len(df)),
        'S8_percentiles': {'p16': p16, 'p50': p50, 'p84': p84},
    }
    # shape-preserving amplitude proxy: A_shear ~ (S8/S8_ref)^2
    if s8_ref and np.isfinite(p50):
        A_med = (p50/s8_ref)**2
        A_lo  = (p16/s8_ref)**2 if np.isfinite(p16) else float('nan')
        A_hi  = (p84/s8_ref)**2 if np.isfinite(p84) else float('nan')
        out['A_shear_from_S8'] = {'median': A_med, 'p16': A_lo, 'p84': A_hi, 'S8_ref': s8_ref}

    # Optional comparison to Planck CMB φφ amplitude (alpha_phi ~ 1 if normalized)
    if phi_amp_json and phi_amp_json.exists():
        try:
            j = json.loads(Path(phi_amp_json).read_text())
            alpha_phi = float(j.get('alpha_hat', np.nan))
            sigma_phi = float(j.get('sigma', np.nan))
            out['phi_amp'] = {'alpha_hat': alpha_phi, 'sigma': sigma_phi}
            if 'A_shear_from_S8' in out and np.isfinite(alpha_phi) and np.isfinite(sigma_phi) and sigma_phi>0:
                A_med = out['A_shear_from_S8']['median']
                tension = abs(A_med - alpha_phi)/sigma_phi
                out['A_over_phi'] = {'A_shear_median': A_med, 'alpha_phi': alpha_phi, 'tension_sigma': tension}
        except Exception:
            pass
    return out

def combine(chunks: list[dict]) -> dict:
    # simple average of medians, just for a “joint quick look” (you’ll likely prefer the DES+KiDS joint chain)
    meds = [c['S8_percentiles']['p50'] for c in chunks if np.isfinite(c['S8_percentiles']['p50'])]
    comb = {'n_chains': len(chunks), 'S8_median_mean': float(np.mean(meds)) if meds else float('nan')}
    return comb

# --------- CLI ---------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', nargs='+', required=True, help='Paths to chain_*.txt files')
    ap.add_argument('--paramnames', nargs='*', help='Optional .paramnames files (same order as chains)')
    ap.add_argument('--out_dir', required=True, help='Output directory for summaries')
    ap.add_argument('--s8_ref', type=float, default=None, help='Reference S8 to define A_shear ≈ (S8/S8_ref)^2 (e.g. 0.83)')
    ap.add_argument('--phi_amp_json', default='out/cmb_envelopes/cmb_lensing_amp.json', help='Planck φφ amplitude JSON (alpha_hat, sigma)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    param_map = {}
    if args.paramnames:
        for i, p in enumerate(args.paramnames):
            param_map[i] = Path(p)

    all_summ = []
    for i, ch in enumerate(args.chains):
        pn = param_map.get(i)
        summ = summarize_one_chain(Path(ch), pn, args.s8_ref, Path(args.phi_amp_json) if args.phi_amp_json else None)
        all_summ.append(summ)

    combined = combine(all_summ)
    out = {'chains': all_summ, 'combined': combined}
    (out_dir/'shear_amp_summary.json').write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_dir/'shear_amp_summary.json'}")

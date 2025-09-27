# data_io.py
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import pandas as pd

from .utils import write_json

# Column alias tables for flexible ingestion
ALIASES_R = ('R_kpc','R','r_kpc','r')
ALIASES_Z = ('z_kpc','z')
ALIASES_VPHI = ('v_phi_kms','vphi_kms','vphi','vT','v_t_kms')
ALIASES_VPHI_ERR = ('v_phi_err_kms','vphi_err_kms','vphi_err','sigma_v','sigmav')
ALIASES_VR = ('v_R_kms','vR','vr_kms','v_r_kms','v_r')
ALIASES_PHI = ('phi_rad','phi_deg','phi','Phi','PHI')
ALIASES_QUAL = ('quality_flag','quality','flag','qflag')


def _find_col(cols, aliases) -> Optional[str]:
    s = set(cols)
    for a in aliases:
        if a in s:
            return a
    # try case-insensitive
    low = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None


def detect_source(slices_glob: str, mw_csv_path: str) -> str:
    """Return 'slices' if processed parquet slices exist else 'mw_csv' if CSV exists."""
    if glob.glob(slices_glob):
        return 'slices'
    if os.path.exists(csv_path):
        return 'mw_csv'
    raise FileNotFoundError(f"No data sources found. Looked for {slices_glob} and {csv_path}")


def load_slices(slices_glob: str,
                zmax: float = 0.5,
                sigma_vmax: Optional[float] = 30.0,
                vRmax: Optional[float] = 40.0,
                phi_bins: Optional[int] = None,
                phi_bin_index: Optional[int] = None,
                max_rows_plot_sample: int = 200000,
                logger=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load processed Gaia slices (Parquet) and return standardized DataFrame + meta.

    Standard columns in returned DataFrame:
      R_kpc, z_kpc, vphi_kms, vR_kms (optional), vphi_err_kms (optional)
    """
    files = sorted(glob.glob(slices_glob))
    if not files:
        raise FileNotFoundError(f"No Parquet slices matched: {slices_glob}")

    # Parquet load loop
    frames = []
    used_files = []
    total_before = dict(stars=0)
    total_after = dict(stars=0)

    # azimuth wedge
    use_wedge = (phi_bins is not None) and (phi_bins > 1) and (phi_bin_index is not None)
    if use_wedge:
        wedge_width = 360.0 / float(phi_bins)
        phi0 = wedge_width * int(phi_bin_index)
        phi1 = phi0 + wedge_width

    for p in files:
        try:
            # Read minimally necessary columns; fall back to full file if needed
            df0 = pd.read_parquet(p)
        except Exception as e:
            if logger:
                logger.warning(f"Could not read {p}: {e}")
            continue

        cols = df0.columns
        cR = _find_col(cols, ALIASES_R)
        cZ = _find_col(cols, ALIASES_Z)
        cV = _find_col(cols, ALIASES_VPHI)
        cVe = _find_col(cols, ALIASES_VPHI_ERR)
        cVR = _find_col(cols, ALIASES_VR)
        cPhi = _find_col(cols, ALIASES_PHI)
        cQ = _find_col(cols, ALIASES_QUAL)
        if cR is None or cV is None or cZ is None:
            if logger:
                logger.info(f"Skipping {p}: missing required columns (R,z,vphi)")
            continue

        total_before['stars'] += len(df0)
        # select
        keep = pd.Series(True, index=df0.index)
        keep &= np.isfinite(pd.to_numeric(df0[cR], errors='coerce'))
        keep &= np.isfinite(pd.to_numeric(df0[cZ], errors='coerce'))
        keep &= np.isfinite(pd.to_numeric(df0[cV], errors='coerce'))
        keep &= np.abs(pd.to_numeric(df0[cZ], errors='coerce')) <= float(zmax)
        if sigma_vmax is not None and cVe is not None:
            keep &= pd.to_numeric(df0[cVe], errors='coerce') <= float(sigma_vmax)
        if vRmax is not None and cVR is not None:
            keep &= np.abs(pd.to_numeric(df0[cVR], errors='coerce')) <= float(vRmax)
        if use_wedge and (cPhi is not None):
            # infer degrees if needed
            phi_vals = pd.to_numeric(df0[cPhi], errors='coerce')
            finite = phi_vals[np.isfinite(phi_vals)]
            if finite.size > 0 and finite.abs().max() <= 6.5:
                phi_deg = np.degrees(phi_vals)
            else:
                phi_deg = phi_vals
            phi_deg = (phi_deg % 360.0 + 360.0) % 360.0
            keep &= (phi_deg >= phi0) & (phi_deg < phi1)
        if cQ is not None and cQ in df0:
            q = pd.to_numeric(df0[cQ], errors='coerce')
            keep &= (q == 0)

        df = df0.loc[keep, [cR, cZ, cV] + [cVR] * (cVR is not None) + [cVe] * (cVe is not None)].copy()
        df.columns = ['R_kpc', 'z_kpc', 'vphi_kms'] + (['vR_kms'] if cVR is not None else []) + (['vphi_err_kms'] if cVe is not None else [])
        if len(df) == 0:
            continue
        frames.append(df)
        used_files.append(p)
        total_after['stars'] += len(df)

    if not frames:
        raise RuntimeError("No valid stars after filtering. Relax zmax/sigma_vmax/vRmax or check columns.")

    stars_df = pd.concat(frames, ignore_index=True)

    # sample for plotting only
    plot_sample = stars_df.sample(n=min(len(stars_df), max_rows_plot_sample), random_state=42) if len(stars_df) > 0 else stars_df

    meta = dict(
        mode='slices',
        files=used_files,
        counts=dict(before=total_before, after=total_after),
        plot_sample_size=len(plot_sample),
    )
    return stars_df, meta


# -----------------------------
# SPARC ingestion
# -----------------------------

def _read_rotmod_dat(path: str) -> pd.DataFrame:
    """Read a SPARC rotmod .dat file with header like:
    # Rad  Vobs errV Vgas Vdisk Vbul ...
    Returns DataFrame with columns: R_kpc, Vobs_kms, Verr_kms, Vgas_kms, Vdisk_kms, Vbul_kms.
    """
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                r, vobs, verr, vgas, vdisk, vbul = map(float, parts[:6])
                rows.append((r, vobs, verr, vgas, vdisk, vbul))
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    df = pd.DataFrame(rows, columns=['R_kpc','Vobs_kms','Verr_kms','Vgas_kms','Vdisk_kms','Vbul_kms'])
    return df


def load_sparc_catalog(rotmod_dir: str, master_sheet: str | None = None, names: List[str] | None = None, logger=None) -> List[Dict[str, Any]]:
    """Load SPARC galaxies from Rotmod_LTG directory; optionally filter by names.
    Returns list of dict(name, df) where df has R_kpc, vphi_kms, vphi_err_kms, vbar_kms.
    """
    if not os.path.isdir(rotmod_dir):
        raise FileNotFoundError(f"rotmod_dir not found: {rotmod_dir}")
    # discover available rotmod files
    files = sorted(glob.glob(os.path.join(rotmod_dir, '*_rotmod.dat')))
    if not files:
        raise RuntimeError(f"No *_rotmod.dat files in {rotmod_dir}")
    # map name->file
    name_to_file = {}
    for fp in files:
        base = os.path.basename(fp)
        name = base.replace('_rotmod.dat','')
        name_to_file[name] = fp
    # determine selection
    selected = []
    if names:
        for n in names:
            if n in name_to_file:
                selected.append(n)
            else:
                if logger:
                    logger.warning(f"SPARC galaxy not found in rotmod list: {n}")
    else:
        # pick first 10 by default
        selected = list(name_to_file.keys())[:10]
    out = []
    for n in selected:
        path = name_to_file[n]
        df = _read_rotmod_dat(path)
        vbar = np.sqrt(np.clip(np.power(df['Vgas_kms'],2) + np.power(df['Vdisk_kms'],2) + np.power(df['Vbul_kms'],2), 0.0, None))
        gal = pd.DataFrame(dict(R_kpc=df['R_kpc'].to_numpy(), vphi_kms=df['Vobs_kms'].to_numpy(), vphi_err_kms=df['Verr_kms'].to_numpy(), vbar_kms=vbar))
        out.append(dict(name=n, df=gal))
    if logger:
        logger.info(f"Loaded {len(out)} SPARC galaxies from {rotmod_dir}")
    return out


def load_mw_csv(csv_path: str,
                zmax: float = 0.5,
                sigma_vmax: Optional[float] = None,
                vRmax: Optional[float] = None,
                max_rows_plot_sample: int = 200000,
                logger=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load precomputed MW Gaia CSV (e.g., data/gaia_mw_real.csv).

    Expected columns: R_kpc, z_kpc, vphi (or vphi_kms), vR, vz (optional), vphi_err (optional).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df0 = pd.read_csv(csv_path)
    cols = df0.columns
    cR = _find_col(cols, ALIASES_R)
    cZ = _find_col(cols, ALIASES_Z)
    cV = _find_col(cols, ('vphi','vphi_kms','v_phi_kms'))
    cVR = _find_col(cols, ('vR','v_R_kms','vr_kms','v_r_kms','v_r'))
    cVe = _find_col(cols, ('vphi_err','vphi_err_kms','v_phi_err_kms'))

    if cR is None or cZ is None or cV is None:
        raise ValueError("CSV missing required columns among (R_kpc, z_kpc, vphi)")

    keep = np.isfinite(pd.to_numeric(df0[cR], errors='coerce')) & \
           np.isfinite(pd.to_numeric(df0[cZ], errors='coerce')) & \
           np.isfinite(pd.to_numeric(df0[cV], errors='coerce'))
    keep &= np.abs(pd.to_numeric(df0[cZ], errors='coerce')) <= float(zmax)
    if sigma_vmax is not None and cVe is not None:
        keep &= pd.to_numeric(df0[cVe], errors='coerce') <= float(sigma_vmax)
    if vRmax is not None and cVR is not None:
        keep &= np.abs(pd.to_numeric(df0[cVR], errors='coerce')) <= float(vRmax)

    cols_list = [cR, cZ, cV] + ([cVR] if cVR is not None else []) + ([cVe] if cVe is not None else [])
    df = df0.loc[keep, cols_list].copy()
    df.columns = ['R_kpc', 'z_kpc', 'vphi_kms'] + (['vR_kms'] if cVR is not None else []) + (['vphi_err_kms'] if cVe is not None else [])

    plot_sample = df.sample(n=min(len(df), max_rows_plot_sample), random_state=42) if len(df) > 0 else df

    meta = dict(
        mode='mw_csv',
        files=[csv_path],
        counts=dict(before=dict(stars=len(df0)), after=dict(stars=len(df))),
        plot_sample_size=len(plot_sample),
    )
    return df, meta
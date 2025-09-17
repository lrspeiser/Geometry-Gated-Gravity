
"""Data loading and preprocessing for SPARC and Milky Way (optional).

Assumptions based on your repository notes (index.md):
- SPARC parquet at 'data/sparc_rotmod_ltg.parquet'
- MasterSheet CSV at 'data/Rotmod_LTG/MasterSheet_SPARC.csv'

This loader is defensive: it tries several column name variants and computes
derived quantities needed by the models (surface density thresholds, masks, etc.).
"""
from __future__ import annotations
import os, math, json, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

M_SUN = 1.98847e30         # kg
KPC = 3.085677581e19       # m
PC  = 3.085677581e16       # m

@dataclass
class GalaxyData:
    name: str
    R_kpc: np.ndarray
    Vobs_kms: np.ndarray
    eVobs_kms: np.ndarray
    Vbar_kms: np.ndarray
    Sigma_bar: Optional[np.ndarray]  # Msun/pc^2 if available
    Rd_kpc: Optional[float]
    Mbar_Msun: Optional[float]
    outer_mask: np.ndarray           # boolean mask for "outer" definition chosen
    # Additional fields for plotting and bookkeeping
    meta: Dict

@dataclass
class Dataset:
    galaxies: List[GalaxyData]
    meta: Dict

def _first_present(container, *cands, default=None):
    """Return the first candidate name present in a container of names.
    Accepts a pandas Index, list-like, or dict-like (uses keys()). Returns the
    candidate string itself, not the value.
    """
    names = container
    if hasattr(container, 'keys') and not hasattr(container, 'dtype'):
        # Likely a dict/DataFrame; use keys for membership
        try:
            names = container.keys()
        except Exception:
            names = container
    for c in cands:
        if c is None:
            continue
        if c in names:
            return c
    return default

def _compute_outer_mask(R_kpc, V_kms, Sigma_bar, Rd_kpc, method="sigma", sigma_th=10.0, k_Rd=3.0, slope_eps=0.03):
    R = np.asarray(R_kpc)
    V = np.asarray(V_kms)
    if method == "sigma" and Sigma_bar is not None:
        return (Sigma_bar <= sigma_th)
    elif method == "kRd" and Rd_kpc is not None and Rd_kpc > 0:
        return (R >= (k_Rd * Rd_kpc))
    elif method == "slope":
        # Outer when |d ln V / d ln R| < eps for at least 2 consecutive bins
        # Approximate derivative
        eps = slope_eps
        with np.errstate(divide='ignore', invalid='ignore'):
            dlnV = np.diff(np.log(np.maximum(V, 1e-6)))
            dlnR = np.diff(np.log(np.maximum(R, 1e-6)))
            slope = np.zeros_like(R)
            slope[1:] = np.abs(dlnV / np.maximum(dlnR, 1e-12))
        # flag regions that are flat-ish; dilate by one bin to mark neighbors
        flat = slope < eps
        # require some minimum radius
        return flat & (R >= np.nanmedian(R))
    else:
        # Fallback to last third of radii if nothing else available
        idx = np.argsort(R)
        mask = np.zeros_like(R, dtype=bool)
        mask[idx[int(0.66*len(idx))]:] = True
        return mask

def load_sparc(path_parquet="data/sparc_rotmod_ltg.parquet",
               path_master="data/Rotmod_LTG/MasterSheet_SPARC.csv",
               outer_method="sigma",
               sigma_th=10.0,
               k_Rd=3.0,
               slope_eps=0.03,
               min_points=8) -> Dataset:
    # Load curve-level parquet
    if not os.path.exists(path_parquet):
        raise FileNotFoundError(f"SPARC parquet not found at {path_parquet}")
    df = pd.read_parquet(path_parquet)
    # Try to gracefully handle column names
    name_col = _first_present(df.columns, "galaxy", "Galaxy", "name", default="galaxy")
    R_col    = _first_present(df.columns, "R_kpc", "R", default="R_kpc")
    Vobs_col = _first_present(df.columns, "Vobs_kms", "Vobs", "Vobs [km/s]", default="Vobs_kms")
    eVobs_col= _first_present(df.columns, "e_Vobs_kms", "eVobs", "eVobs [km/s]", default="e_Vobs_kms")
    Vbar_col = _first_present(df.columns, "Vbar_kms", "Vbar", default=None)
    if Vbar_col is None:
        # Try to reconstruct Vbar^2 = Vgas^2 + Vdisk^2 + Vbul^2
        vgas = _first_present(df.columns, "Vgas_kms", "Vgas", default=None)
        vdisk= _first_present(df.columns, "Vdisk_kms", "Vdisk", default=None)
        vbul = _first_present(df.columns, "Vbul_kms", "Vbul", default=None)
        if vgas is not None and vdisk is not None and (vgas in df.columns) and (vdisk in df.columns):
            df["__Vbar2__"] = (df[vgas]**2 + df[vdisk]**2 + (df[vbul]**2 if (vbul is not None and vbul in df.columns) else 0.0))
            df["Vbar_kms"]  = np.sqrt(np.maximum(df["__Vbar2__"], 0.0))
            Vbar_col = "Vbar_kms"
        else:
            raise KeyError("Cannot find Vbar_kms or components to reconstruct it.")
    Sigma_col = _first_present(df.columns, "Sigma_bar", "Sigma", "Sigma [Msun/pc^2]", default=None)
    # Load master sheet for Rd and masses if available
    master = None
    if os.path.exists(path_master):
        try:
            master = pd.read_csv(path_master)
        except Exception:
            # Fallback: let pandas infer delimiter via python engine; skip bad lines
            try:
                master = pd.read_csv(path_master, sep=None, engine='python', on_bad_lines='skip')
            except Exception:
                master = None
        if master is not None:
            gname_m = _first_present(master.columns, "galaxy", "Galaxy", "NAME", default=None)
            if gname_m is None or gname_m not in master.columns:
                # Could not locate a valid galaxy-name column; ignore master
                master = None
            else:
                Rd_col  = _first_present(master.columns, "Rd_kpc", "Ropt_kpc", "R_d", default=None)
                Mbar_col= _first_present(master.columns, "Mbar_Msun", "Mbar", "Mbar [Msun]", default=None)
                master = master.rename(columns={gname_m:"Galaxy"})
                if Rd_col and Rd_col in master.columns: master = master.rename(columns={Rd_col:"Rd_kpc"})
                if Mbar_col and Mbar_col in master.columns: master = master.rename(columns={Mbar_col:"Mbar_Msun"})
    galaxies = []
    for gname, gdf in df.groupby(name_col):
        gdf = gdf.sort_values(R_col)
        R = gdf[R_col].to_numpy(float)
        Vobs = gdf[Vobs_col].to_numpy(float)
        eV = gdf[eVobs_col].to_numpy(float) if eVobs_col in gdf else np.full_like(Vobs, np.nan)
        Vbar= gdf[Vbar_col].to_numpy(float)
        Sigma = gdf[Sigma_col].to_numpy(float) if Sigma_col and Sigma_col in gdf else None
        Rd = None; Mbar=None
        if master is not None:
            row = master.loc[master["Galaxy"]==gname]
            if len(row):
                Rd  = float(row.iloc[0].get("Rd_kpc", np.nan)) if "Rd_kpc" in row.columns else None
                Mbar= float(row.iloc[0].get("Mbar_Msun", np.nan)) if "Mbar_Msun" in row.columns else None
                if (Rd is not None) and (not np.isfinite(Rd) or Rd<=0): Rd=None
                if (Mbar is not None) and (not np.isfinite(Mbar) or Mbar<=0): Mbar=None
        # Sanity
        if len(R) < min_points: 
            continue
        # Reasonable eVobs floor
        if not np.isfinite(eV).all():
            eV = np.nan_to_num(eV, nan=np.nanmedian(np.diff(Vobs, prepend=Vobs[0]))/2.0)
            eV = np.clip(eV, 2.0, None)
        outer_mask = _compute_outer_mask(R, Vobs, Sigma, Rd, method=outer_method, sigma_th=sigma_th, k_Rd=k_Rd, slope_eps=slope_eps)
        galaxies.append(GalaxyData(
            name=str(gname),
            R_kpc=R, Vobs_kms=Vobs, eVobs_kms=eV, Vbar_kms=Vbar,
            Sigma_bar=Sigma, Rd_kpc=Rd, Mbar_Msun=Mbar,
            outer_mask=outer_mask,
            meta={"n_points": len(R)}
        ))
    meta = {"N_galaxies": len(galaxies), "outer_method": outer_method, "sigma_th": sigma_th, "k_Rd": k_Rd, "slope_eps": slope_eps}
    return Dataset(galaxies=galaxies, meta=meta)

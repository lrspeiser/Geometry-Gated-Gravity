
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Optional
from .data import GalaxyData, Dataset

def bin_mw_stars(df: pd.DataFrame,
                 r_col="R_kpc",
                 vphi_col="Vphi_kms",
                 w_col=None,
                 r_min=2.0, r_max=20.0, nbins=100,
                 name="Milky Way",
                 sigma_floor=3.0) -> GalaxyData:
    """Compress many Gaia stars into a binned rotation curve with uncertainties.

    Expects columns:
      - R_kpc: Galactocentric cylindrical radius in kpc
      - Vphi_kms: Azimuthal velocity (circular component) in km/s
      - (optional) weight column for robust binning

    Returns a GalaxyData usable by the hierarchical fit.
    """
    df = df.copy()
    df = df[np.isfinite(df[r_col]) & np.isfinite(df[vphi_col])]
    mask = (df[r_col] >= r_min) & (df[r_col] <= r_max)
    df = df.loc[mask]
    bins = np.linspace(r_min, r_max, nbins+1)
    centers = 0.5*(bins[1:] + bins[:-1])
    R_kpc = []; V_kms = []; eV_kms = []; counts=[]

    for i in range(nbins):
        a,b = bins[i], bins[i+1]
        sub = df[(df[r_col]>=a) & (df[r_col]<b)]
        if len(sub) < 10:
            continue
        w = sub[w_col].to_numpy() if (w_col and w_col in sub) else np.ones(len(sub))
        v = sub[vphi_col].to_numpy()
        # Weighted robust statistics
        med = np.average(v, weights=w)
        # bootstrap stderr
        if len(v) > 1000:
            # quick bootstrap on a subset for speed
            idx = np.random.randint(0, len(v), size=(256, len(v)//10))
            boots = np.median(v[idx], axis=1)
            se = np.std(boots, ddof=1)
        else:
            se = np.std(v, ddof=1)/np.sqrt(len(v))
        R_kpc.append(centers[i]); V_kms.append(med); eV_kms.append(max(se, sigma_floor)); counts.append(len(v))

    R_kpc = np.asarray(R_kpc); V_kms=np.asarray(V_kms); eV_kms=np.asarray(eV_kms)
    return GalaxyData(
        name=name,
        R_kpc=R_kpc,
        Vobs_kms=V_kms,
        eVobs_kms=eV_kms,
        Vbar_kms=np.full_like(R_kpc, np.nan),   # We'll fill with a chosen MW baryonic model later if desired
        Sigma_bar=None,
        Rd_kpc=None,
        Mbar_Msun=None,
        outer_mask=(R_kpc >= np.nanmedian(R_kpc)),  # placeholder outer criterion for MW
        meta={"counts": counts, "source": "Gaia-binned"}
    )

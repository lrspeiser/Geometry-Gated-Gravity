#!/usr/bin/env python3
"""
baryon_loader.py

Loads per-cluster projected baryon surface-density profiles if available.
File path: data/baryon_profiles/{SANITIZED_NAME}.csv
Required columns: R_kpc, Sigma_baryon
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .cluster_overrides import normalize_cluster_name

REPO_ROOT = Path(__file__).resolve().parent.parent
BARYON_DIR = REPO_ROOT / "data" / "baryon_profiles"


def load_baryon_profile(cluster_name: str) -> Optional[pd.DataFrame]:
    key = normalize_cluster_name(cluster_name)
    path = BARYON_DIR / f"{key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"R_kpc", "Sigma_baryon"}.issubset(df.columns):
        raise ValueError(f"Baryon profile missing required columns in {path}")
    return df.sort_values("R_kpc").reset_index(drop=True)


def interpolate_baryon(df: pd.DataFrame, R_kpc: np.ndarray) -> np.ndarray:
    R = df["R_kpc"].values.astype(float)
    S = df["Sigma_baryon"].values.astype(float)
    # Extrapolate flat at ends
    return np.interp(R_kpc, R, S, left=S[0], right=S[-1])

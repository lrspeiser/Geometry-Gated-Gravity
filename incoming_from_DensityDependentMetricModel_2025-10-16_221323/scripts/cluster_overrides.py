#!/usr/bin/env python3
"""
cluster_overrides.py

Utilities to load per-cluster override configurations for:
- Source redshift distributions P(z_s)
- External convergence prior widths (kappa_ext_sigma)
- BCG parameters and extra baryon components

Override files live under: data/overrides/{SANITIZED_NAME}.json
where SANITIZED_NAME is the cluster name uppercased with spaces/hyphens/dots removed
and '+' replaced by 'PLUS'.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_DIR = REPO_ROOT / "data" / "overrides"


def normalize_cluster_name(name: str) -> str:
    """Sanitize cluster name to a filesystem-safe key for overrides."""
    key = name.upper()
    for ch in [" ", "-", "."]:
        key = key.replace(ch, "")
    key = key.replace("+", "PLUS")
    return key


def load_cluster_override(cluster_name: str, base_dir: Optional[Path] = None) -> Optional[Dict]:
    """
    Load override JSON for a given cluster if present.

    Schema (example):
    {
      "kappa_ext_sigma": 0.05,
      "bcg": { "M_Msun": 2.0e12, "a_kpc": 15.0 },
      "extra_baryon_components": [
        {"type": "hernquist", "M_Msun": 5.0e12, "a_kpc": 120.0}
      ],
      "source_distribution": {
        "type": "mixture_normal",
        "components": [
          {"weight": 0.6, "mu": 1.7, "sigma": 0.2},
          {"weight": 0.4, "mu": 3.0, "sigma": 0.3}
        ],
        "z_min": 0.1, "z_max": 6.0, "n_grid": 400
      }
    }
    """
    if base_dir is None:
        base_dir = OVERRIDES_DIR

    key = normalize_cluster_name(cluster_name)
    path = base_dir / f"{key}.json"

    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

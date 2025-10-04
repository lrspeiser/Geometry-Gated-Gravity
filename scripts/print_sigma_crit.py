#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import sigma_crit_Msun_per_kpc2

if __name__ == "__main__":
    z_d, z_s = 0.206, 2.0
    val = sigma_crit_Msun_per_kpc2(z_d, z_s)
    print(f"Sigma_crit(z_d={z_d}, z_s={z_s}) = {val:.6e} Msun/kpc^2")

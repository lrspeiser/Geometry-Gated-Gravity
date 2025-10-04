"""
GPU-Accelerated Galaxy Fitting for Squared Gravity Model (SPARC-like)

This script uses the GeometricExponentGravity model with CuPy acceleration
(or NumPy fallback) to fit galaxy rotation curves from processed SPARC-like data.

Expected per-galaxy CSV columns (data/sparc/processed/<galaxy>/profile.csv):
- R_kpc: radial bins in kpc
- Sigma_bar_kpc2: baryon surface density (Msun/kpc^2) at R
- v_obs_kms: observed circular velocity (km/s) per R bin
- Rd_kpc (optional): disk scale length; defaults to 3.0 kpc if missing

Outputs saved to: data/sparc/processed/squared_gravity/gpu_fits/
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from itertools import product
import os
import sys
import argparse

# Ensure local imports work when running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cupy_utils import get_array_module, trapz_cumsum
from geometric_exponent import GeometricExponentGravity


G_KPC_KMS2_PER_MSUN = 4.30091e-6  # Gravitational constant in kpc (km/s)^2 Msun^-1


def load_sparc_profiles(data_dir="data/sparc/processed"):
    """Load processed SPARC galaxy profiles. Returns dict of galaxies with arrays."""
    profiles = {}
    data_path = Path(data_dir)

    for gal_dir in data_path.glob("*"):
        if not gal_dir.is_dir():
            continue
        profile_file = gal_dir / "profile.csv"
        if not profile_file.exists():
            # Accept alternative naming convention
            profile_file = gal_dir / "baryon_profile.csv"
        if not profile_file.exists():
            continue
        try:
            df = pd.read_csv(profile_file)
        except Exception:
            continue

        if all(col in df.columns for col in ["R_kpc", "Sigma_bar_kpc2", "v_obs_kms"]):
            R_kpc = df["R_kpc"].values
            Sigma_bar = df["Sigma_bar_kpc2"].values
            v_obs = df["v_obs_kms"].values
            Rd_kpc = float(df.get("Rd_kpc", pd.Series([3.0])).iloc[0])

            profiles[gal_dir.name] = {
                "R_kpc": R_kpc,
                "Sigma_bar_kpc2": Sigma_bar,
                "v_obs_kms": v_obs,
                "Rd_kpc": Rd_kpc,
            }

    return profiles


def compute_v_model_gpu(xp, R_kpc, Sigma_eff_kpc2):
    """Compute model circular velocity from Σ_eff assuming spherical approximation.

    M(<R) = 2π ∫ Σ_eff(r) r dr, then v = sqrt(G M / R)
    """
    R = xp.asarray(R_kpc)
    Se = xp.asarray(Sigma_eff_kpc2)

    # Enclosed mass (Msun)
    M_enclosed = 2.0 * np.pi * trapz_cumsum(xp, Se * R, R)

    # Avoid divide by zero
    R_safe = xp.maximum(R, 1e-6)
    v2 = G_KPC_KMS2_PER_MSUN * M_enclosed / R_safe
    v = xp.sqrt(xp.maximum(v2, 0.0))
    return v


def fit_galaxy_gpu(xp, galaxy_data, params):
    """Fit a single galaxy with given parameters using GPU backend."""
    gamma1, gamma2, a, b, d = params

    model = GeometricExponentGravity(
        gamma1=gamma1,
        gamma2=gamma2,
        a=a,
        b=b,
        d=d,
        R_scale_kpc=10.0,
        beta_clip=(1.0, 5.0),
    )

    R_kpc = galaxy_data["R_kpc"]
    Sigma_bar = galaxy_data["Sigma_bar_kpc2"]
    Rd_kpc = galaxy_data["Rd_kpc"]
    v_obs = galaxy_data["v_obs_kms"]

    # Compute Σ_eff
    Sigma_eff, fX, beta = model.Sigma_effective_xp(xp, R_kpc, Sigma_bar, Rd_kpc)

    # Compute v_model from Σ_eff
    v_model = compute_v_model_gpu(xp, R_kpc, Sigma_eff)

    # Interpolate or sample at R positions of v_obs (assume aligned)
    v_obs_xp = xp.asarray(v_obs)

    # Chi2 with 10% fractional errors as a placeholder (until actual errors are provided)
    # NOTE: If error bars are available, replace this with the provided uncertainties
    sigma = xp.maximum(0.1 * xp.maximum(v_obs_xp, 1.0), 5.0)
    chi2 = xp.sum(((v_model - v_obs_xp) / sigma) ** 2)

    return float(chi2)


def run_gpu_grid_search_galaxies(profiles, param_grid, output_dir):
    xp = get_array_module()
    print(f"Using array backend: {'CuPy (GPU)' if xp.__name__ == 'cupy' else 'NumPy (CPU)'}")

    results = []
    total_combinations = len(list(product(*param_grid.values())))

    print(f"Running galaxy grid search with {total_combinations} parameter combinations")
    print(f"Fitting {len(profiles)} galaxies")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for i, params in enumerate(product(*param_grid.values())):
        param_dict = dict(zip(param_grid.keys(), params))

        total_chi2 = 0.0
        galaxy_results = {}

        for gal_name, gal_data in profiles.items():
            try:
                chi2 = fit_galaxy_gpu(xp, gal_data, params)
                total_chi2 += chi2
                galaxy_results[gal_name] = {"chi2": float(chi2)}
            except Exception as e:
                print(f"Error fitting {gal_name}: {e}")
                galaxy_results[gal_name] = {"chi2": 1e6}
                total_chi2 += 1e6

        results.append({"params": param_dict, "total_chi2": float(total_chi2), "galaxies": galaxy_results})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total_combinations - i - 1) / max(rate, 1e-9)
            print(
                f"Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%) "
                f"Rate: {rate:.1f}/s ETA: {eta:.0f}s"
            )

    results.sort(key=lambda x: x["total_chi2"])

    with open(output_path / "gpu_grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    best = results[:20]
    rows = []
    for rank, res in enumerate(best, start=1):
        row = {"rank": rank, "total_chi2": res["total_chi2"], **res["params"]}
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(output_path / "best_fits_summary.csv", index=False)

    elapsed_total = time.time() - start_time
    print(f"\nGalaxy grid search completed in {elapsed_total:.1f}s")
    print(f"Best fit chi2: {results[0]['total_chi2']:.2f}")
    print(f"Best parameters: {results[0]['params']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated grid search for galaxies")
    parser.add_argument("--data-dir", default="data/sparc/processed", help="Path to processed SPARC data")
    parser.add_argument(
        "--output-dir",
        default="data/sparc/processed/squared_gravity/gpu_fits",
        help="Output directory",
    )
    parser.add_argument("--gamma1", nargs=3, type=float, default=[0.2, 1.0, 2], help="gamma1 min max count")
    parser.add_argument("--gamma2", nargs=3, type=float, default=[0.1, 0.5, 2], help="gamma2 min max count")
    parser.add_argument("--a", nargs=3, type=float, default=[1.0, 3.0, 2], help="a min max count")
    parser.add_argument("--b", nargs=3, type=float, default=[0.2, 0.8, 2], help="b min max count")
    parser.add_argument("--d", nargs=3, type=float, default=[0.2, 0.8, 2], help="d min max count")
    args = parser.parse_args()

    print("Loading SPARC galaxy profiles...")
    profiles = load_sparc_profiles(args.data_dir)
    print(f"Loaded {len(profiles)} galaxies: {list(profiles.keys())}")

    if not profiles:
        print("No galaxy profiles found! Check data directory structure.")
        return

    g1_min, g1_max, g1_n = args.gamma1
    g2_min, g2_max, g2_n = args.gamma2
    a_min, a_max, a_n = args.a
    b_min, b_max, b_n = args.b
    d_min, d_max, d_n = args.d

    param_grid = {
        "gamma1": np.linspace(g1_min, g1_max, int(g1_n)),
        "gamma2": np.linspace(g2_min, g2_max, int(g2_n)),
        "a": np.linspace(a_min, a_max, int(a_n)),
        "b": np.linspace(b_min, b_max, int(b_n)),
        "d": np.linspace(d_min, d_max, int(d_n)),
    }

    run_gpu_grid_search_galaxies(profiles, param_grid, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gaia-based Test of Environment-Dependent Gravity Enhancement
=============================================================

Tests the hypothesis that low-mass stars require higher effective gravity
than high-mass stars at the same radius in LOW-DENSITY (void-like) environments,
but NOT in high-density regions.

This tests your cooperative response / void-gate mechanism at galaxy scales.

Key Predictions:
1. In LOW-density outer disk (R > 15 kpc): Low-mass stars show higher v_φ
   (or g_eff = v_φ²/R) than high-mass stars at same (R, φ, z)
2. In HIGH-density regions: No difference (gate off)
3. Effect should be environment-dependent, not just mass-dependent

Author: AI Assistant + User
Date: 2025-01-10
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass

# Try Astropy imports
try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric
    ASTROPY_AVAILABLE = True
except ImportError:
    print("Warning: Astropy not available. Will use simplified coordinate transforms.")
    ASTROPY_AVAILABLE = False

# Try scipy for KDTree
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: Scipy not available. Will use simplified density estimation.")
    SCIPY_AVAILABLE = False


@dataclass
class GaiaTestConfig:
    """Configuration for Gaia analysis."""
    # Data file
    input_file: str = "gaia_144k.csv"
    
    # Quality cuts
    min_parallax: float = 0.2  # mas
    min_parallax_snr: float = 10.0
    max_ruwe: float = 1.4
    max_rv_error: float = 5.0  # km/s
    
    # Solar position and velocity
    R0: float = 8.178  # kpc
    z0: float = 0.0208  # kpc
    U_sun: float = 11.1  # km/s
    V_sun: float = 232.24  # km/s (12.24 + 220.0)
    W_sun: float = 7.25  # km/s
    
    # Selection for near-circular outer disk
    R_min: float = 15.0  # kpc
    R_max: float = 25.0  # kpc
    z_max: float = 1.0  # kpc
    vR_max: float = 25.0  # km/s (near-circular)
    vz_max: float = 25.0  # km/s
    
    # Pair matching
    R_bin_size: float = 0.5  # kpc
    min_pairs_per_bin: int = 10
    n_density_deciles: int = 10
    void_deciles: List[int] = None  # Will default to [0, 1, 2]
    dense_deciles: List[int] = None  # Will default to [7, 8, 9]
    
    # Density estimation (kNN)
    k_neighbors: int = 50
    
    def __post_init__(self):
        if self.void_deciles is None:
            self.void_deciles = [0, 1, 2]
        if self.dense_deciles is None:
            self.dense_deciles = [7, 8, 9]


def load_and_clean_gaia_data(config: GaiaTestConfig) -> pd.DataFrame:
    """Load Gaia data and apply quality cuts."""
    print(f"\n{'='*60}")
    print("Loading Gaia data...")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(config.input_file)
    print(f"Loaded {len(df)} stars from {config.input_file}")
    
    # Handle infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    required_cols = ["ra", "dec", "parallax", "pmra", "pmdec", "rv", "bp_rp", "Gmag"]
    df = df.dropna(subset=required_cols)
    print(f"After dropping NaNs: {len(df)} stars")
    
    # Quality cuts
    ok = (df["parallax"] > config.min_parallax)
    
    if "parallax_error" in df.columns:
        ok &= (df["parallax"] / df["parallax_error"] > config.min_parallax_snr)
    
    if "rv_error" in df.columns:
        ok &= (df["rv_error"] < config.max_rv_error)
    
    if "ruwe" in df.columns:
        ok &= (df["ruwe"] < config.max_ruwe)
    
    df = df[ok].copy()
    print(f"After quality cuts: {len(df)} stars")
    print(f"  - Parallax > {config.min_parallax} mas")
    print(f"  - Parallax S/N > {config.min_parallax_snr}")
    print(f"  - RV error < {config.max_rv_error} km/s")
    if "ruwe" in df.columns:
        print(f"  - RUWE < {config.max_ruwe}")
    
    return df


def transform_to_galactocentric(df: pd.DataFrame, config: GaiaTestConfig) -> Tuple[np.ndarray, ...]:
    """
    Transform Gaia (RA, Dec, parallax, PM, RV) to Galactocentric cylindrical coordinates.
    
    Returns
    -------
    R, phi, z, v_R, v_phi, v_z : arrays in kpc and km/s
    """
    print(f"\n{'='*60}")
    print("Transforming to Galactocentric coordinates...")
    print(f"{'='*60}\n")
    
    if ASTROPY_AVAILABLE:
        return _transform_with_astropy(df, config)
    else:
        return _transform_simplified(df, config)


def _transform_with_astropy(df: pd.DataFrame, config: GaiaTestConfig) -> Tuple[np.ndarray, ...]:
    """Use Astropy for accurate coordinate transformation."""
    R0 = config.R0 * u.kpc
    z0 = config.z0 * u.kpc
    v_sun = [config.U_sun, config.V_sun, config.W_sun] * (u.km / u.s)
    
    coord = SkyCoord(
        ra=df.ra.values * u.deg,
        dec=df.dec.values * u.deg,
        distance=(1e3 / df.parallax.values) * u.pc,
        pm_ra_cosdec=df.pmra.values * (u.mas / u.yr),
        pm_dec=df.pmdec.values * (u.mas / u.yr),
        radial_velocity=df.rv.values * (u.km / u.s),
    )
    
    galcen = coord.transform_to(
        Galactocentric(galcen_distance=R0, z_sun=z0, galcen_v_sun=v_sun)
    )
    
    # Cartesian components
    x = galcen.x.to(u.kpc).value
    y = galcen.y.to(u.kpc).value
    z = galcen.z.to(u.kpc).value
    vx = galcen.v_x.to(u.km / u.s).value
    vy = galcen.v_y.to(u.km / u.s).value
    vz = galcen.v_z.to(u.km / u.s).value
    
    # Convert to cylindrical
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    
    v_R = (x * vx + y * vy) / R
    v_phi = (x * vy - y * vx) / R  # Positive in direction of rotation
    v_z = vz
    
    print(f"Coordinate transformation complete (Astropy)")
    print(f"  R range: [{np.min(R):.1f}, {np.max(R):.1f}] kpc")
    print(f"  |z| range: [{np.min(np.abs(z)):.1f}, {np.max(np.abs(z)):.1f}] kpc")
    
    return R, phi, z, v_R, v_phi, v_z, x, y


def _transform_simplified(df: pd.DataFrame, config: GaiaTestConfig) -> Tuple[np.ndarray, ...]:
    """Simplified transformation (less accurate but functional without Astropy)."""
    print("Warning: Using simplified coordinate transform without Astropy")
    print("Results will be approximate!")
    
    # This is a placeholder - real implementation would need proper transforms
    raise NotImplementedError("Please install Astropy for accurate coordinate transforms: pip install astropy")


def estimate_stellar_masses(df: pd.DataFrame) -> np.ndarray:
    """
    Estimate stellar masses from color-magnitude diagram.
    
    Uses a crude main-sequence mass-luminosity relation.
    If 'mass' column exists, uses that instead.
    """
    if "mass" in df.columns:
        print("\nUsing existing 'mass' column from input data")
        return df["mass"].values
    
    print("\nEstimating stellar masses from color-magnitude...")
    print("Warning: Using CRUDE mass proxy! Provide real masses if available.")
    
    # Distance modulus
    dist_pc = 1e3 / df.parallax.values
    mu = 5 * np.log10(dist_pc / 10.0)
    
    # Absolute magnitude
    M_G = df.Gmag.values - mu
    color = df.bp_rp.values
    
    # Very crude main-sequence relation
    # Blue stars (bp_rp ~ 0.5) → ~1.5 Msun
    # Red stars (bp_rp ~ 2.0) → ~0.6 Msun
    # This is just a toy model - USE REAL MASSES IF AVAILABLE
    mass = np.clip(10 ** (0.5 - 0.2 * color - 0.02 * (M_G - 4.8)), 0.6, 3.0)
    
    print(f"  Mass range: [{np.min(mass):.2f}, {np.max(mass):.2f}] M_sun")
    print(f"  Median mass: {np.median(mass):.2f} M_sun")
    
    return mass


def compute_local_density(x: np.ndarray, y: np.ndarray, z: np.ndarray, config: GaiaTestConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local stellar density as proxy for environment.
    
    Returns
    -------
    rho_local : array
        Local density in stars/kpc³
    decile : array
        Density decile (0 = most void-like, 9 = densest)
    """
    print(f"\n{'='*60}")
    print("Computing local stellar density (environment proxy)...")
    print(f"{'='*60}\n")
    
    if not SCIPY_AVAILABLE:
        raise ImportError("Scipy required for density estimation: pip install scipy")
    
    xyz = np.vstack([x, y, z]).T
    tree = cKDTree(xyz)
    
    # Distance to k-th nearest neighbor
    k = config.k_neighbors
    d_k, _ = tree.query(xyz, k=k + 1)  # +1 includes the point itself
    r_k = d_k[:, -1]
    
    # Density ~ k / volume
    rho_local = k / (4 / 3 * np.pi * r_k**3)
    
    # Assign to deciles (0 = most void-like)
    decile = pd.qcut(rho_local, config.n_density_deciles, labels=False, duplicates='drop')
    
    print(f"Density statistics (stars/kpc³):")
    print(f"  Min: {np.min(rho_local):.1f}")
    print(f"  Median: {np.median(rho_local):.1f}")
    print(f"  Max: {np.max(rho_local):.1f}")
    print(f"\nDecile 0 (void): ρ < {np.percentile(rho_local, 10):.1f} stars/kpc³")
    print(f"Decile 9 (dense): ρ > {np.percentile(rho_local, 90):.1f} stars/kpc³")
    
    return rho_local, decile


def select_outer_disk_sample(R, phi, z, v_R, v_phi, v_z, mass, rho_local, decile, config: GaiaTestConfig) -> Dict:
    """Select clean sample of outer disk stars with near-circular orbits."""
    print(f"\n{'='*60}")
    print("Selecting outer disk sample...")
    print(f"{'='*60}\n")
    
    sel = (
        (R > config.R_min) & (R < config.R_max) &
        (np.abs(z) < config.z_max) &
        (np.abs(v_R) < config.vR_max) &
        (np.abs(v_z) < config.vz_max)
    )
    
    print(f"Selection criteria:")
    print(f"  {config.R_min} < R < {config.R_max} kpc")
    print(f"  |z| < {config.z_max} kpc")
    print(f"  |v_R| < {config.vR_max} km/s")
    print(f"  |v_z| < {config.vz_max} km/s")
    print(f"\nSelected {np.sum(sel)} / {len(sel)} stars ({100*np.sum(sel)/len(sel):.1f}%)")
    
    return {
        "R": R[sel],
        "phi": phi[sel],
        "z": z[sel],
        "v_R": v_R[sel],
        "v_phi": v_phi[sel],
        "v_z": v_z[sel],
        "mass": mass[sel],
        "rho_local": rho_local[sel],
        "decile": decile[sel],
    }


def match_pairs_and_compute_delta_g(data: Dict, config: GaiaTestConfig) -> pd.DataFrame:
    """
    Match low-mass and high-mass stars at same R and environment.
    Compute Δg_eff = g_eff(low-mass) - g_eff(high-mass).
    
    Key prediction: Δg_eff > 0 in void-like environments only!
    """
    print(f"\n{'='*60}")
    print("Matching pairs and computing Δg_eff...")
    print(f"{'='*60}\n")
    
    R, v_phi, mass, decile = data["R"], data["v_phi"], data["mass"], data["decile"]
    
    # Bin by R
    R_bin = (np.floor(R / config.R_bin_size) * config.R_bin_size).astype(float)
    
    pairs = []
    
    # Test void-like deciles
    print("Testing VOID-like environments (low density):")
    for d in config.void_deciles:
        in_d = (decile == d)
        if np.sum(in_d) == 0:
            continue
        
        print(f"  Decile {d}: {np.sum(in_d)} stars")
        
        for r in np.unique(R_bin[in_d]):
            idx = np.where(in_d & (R_bin == r))[0]
            if len(idx) < 2 * config.min_pairs_per_bin:
                continue
            
            # Split by mass median within this cell
            m = mass[idx]
            m_med = np.median(m)
            i_low = idx[m < m_med]
            i_high = idx[m >= m_med]
            
            n = min(len(i_low), len(i_high))
            if n < config.min_pairs_per_bin:
                continue
            
            # Match (simple: just use first n of each)
            # Could improve with KNN matching in (phi, z, v_R) space
            i_low = i_low[:n]
            i_high = i_high[:n]
            
            # Effective gravity: g_eff = v_φ² / R  [(km/s)² / kpc]
            g_low = (v_phi[i_low] ** 2) / R[i_low]
            g_high = (v_phi[i_high] ** 2) / R[i_high]
            
            pairs.append({
                "R_bin": r,
                "decile": d,
                "environment": "void",
                "delta_g_eff": np.median(g_low - g_high),
                "delta_v_phi": np.median(v_phi[i_low] - v_phi[i_high]),
                "g_eff_low": np.median(g_low),
                "g_eff_high": np.median(g_high),
                "mass_low": np.median(mass[i_low]),
                "mass_high": np.median(mass[i_high]),
                "N_pairs": n,
            })
    
    # Test dense environments (control)
    print("\nTesting DENSE environments (high density, control):")
    for d in config.dense_deciles:
        in_d = (decile == d)
        if np.sum(in_d) == 0:
            continue
        
        print(f"  Decile {d}: {np.sum(in_d)} stars")
        
        for r in np.unique(R_bin[in_d]):
            idx = np.where(in_d & (R_bin == r))[0]
            if len(idx) < 2 * config.min_pairs_per_bin:
                continue
            
            m = mass[idx]
            m_med = np.median(m)
            i_low = idx[m < m_med]
            i_high = idx[m >= m_med]
            
            n = min(len(i_low), len(i_high))
            if n < config.min_pairs_per_bin:
                continue
            
            i_low = i_low[:n]
            i_high = i_high[:n]
            
            g_low = (v_phi[i_low] ** 2) / R[i_low]
            g_high = (v_phi[i_high] ** 2) / R[i_high]
            
            pairs.append({
                "R_bin": r,
                "decile": d,
                "environment": "dense",
                "delta_g_eff": np.median(g_low - g_high),
                "delta_v_phi": np.median(v_phi[i_low] - v_phi[i_high]),
                "g_eff_low": np.median(g_low),
                "g_eff_high": np.median(g_high),
                "mass_low": np.median(mass[i_low]),
                "mass_high": np.median(mass[i_high]),
                "N_pairs": n,
            })
    
    res = pd.DataFrame(pairs).sort_values(["environment", "decile", "R_bin"])
    print(f"\nCreated {len(res)} matched bins")
    
    return res


def analyze_results(results: pd.DataFrame) -> None:
    """Analyze and display test results."""
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}\n")
    
    # Aggregate by environment
    agg = results.groupby("environment").agg({
        "delta_g_eff": ["median", "mean", "std", "count"],
        "delta_v_phi": ["median", "mean"],
    })
    
    print("Δg_eff = g_eff(low-mass) - g_eff(high-mass)")
    print("="*60)
    print(agg)
    print()
    
    # Statistical test
    void_delta_g = results[results.environment == "void"]["delta_g_eff"].values
    dense_delta_g = results[results.environment == "dense"]["delta_g_eff"].values
    
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST")
    print(f"{'='*60}\n")
    
    print("H0 (null): Δg_eff = 0 (no mass-dependent effect)")
    print("H1 (cooperative response): Δg_eff > 0 in VOID, Δg_eff ≈ 0 in DENSE\n")
    
    if len(void_delta_g) > 0:
        void_mean = np.mean(void_delta_g)
        void_std = np.std(void_delta_g)
        void_stderr = void_std / np.sqrt(len(void_delta_g))
        void_snr = void_mean / void_stderr if void_stderr > 0 else 0
        
        print(f"VOID environment:")
        print(f"  Mean Δg_eff = {void_mean:.3f} ± {void_stderr:.3f} (km/s)²/kpc")
        print(f"  SNR = {void_snr:.2f}")
        
        if void_snr > 3:
            print(f"  ✅ SIGNIFICANT enhancement detected (>{3}σ)!")
        elif void_snr > 2:
            print(f"  ⚠️  Marginal detection (~{void_snr:.1f}σ)")
        else:
            print(f"  ❌ No significant detection (<2σ)")
    
    if len(dense_delta_g) > 0:
        dense_mean = np.mean(dense_delta_g)
        dense_std = np.std(dense_delta_g)
        dense_stderr = dense_std / np.sqrt(len(dense_delta_g))
        dense_snr = abs(dense_mean) / dense_stderr if dense_stderr > 0 else 0
        
        print(f"\nDENSE environment (control):")
        print(f"  Mean Δg_eff = {dense_mean:.3f} ± {dense_stderr:.3f} (km/s)²/kpc")
        print(f"  SNR = {dense_snr:.2f}")
        
        if abs(dense_snr) < 2:
            print(f"  ✅ No effect in dense regions (as expected)")
        else:
            print(f"  ⚠️  Unexpected signal in dense regions")
    
    # Environment contrast
    if len(void_delta_g) > 0 and len(dense_delta_g) > 0:
        contrast = void_mean - dense_mean
        contrast_err = np.sqrt(void_stderr**2 + dense_stderr**2)
        contrast_snr = contrast / contrast_err if contrast_err > 0 else 0
        
        print(f"\nEnvironment contrast:")
        print(f"  Δg_void - Δg_dense = {contrast:.3f} ± {contrast_err:.3f} (km/s)²/kpc")
        print(f"  SNR = {contrast_snr:.2f}")
        
        if contrast_snr > 3:
            print(f"  ✅ STRONG environment dependence detected!")
        elif contrast_snr > 2:
            print(f"  ⚠️  Marginal environment dependence")
        else:
            print(f"  ❌ No clear environment dependence")


def plot_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Create diagnostic plots."""
    print(f"\n{'='*60}")
    print("Creating diagnostic plots...")
    print(f"{'='*60}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Δg_eff vs R by environment
    ax = axes[0, 0]
    for env in ["void", "dense"]:
        subset = results[results.environment == env]
        ax.scatter(subset.R_bin, subset.delta_g_eff, 
                  label=env.capitalize(), alpha=0.6, s=subset.N_pairs)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Galactocentric Radius (kpc)")
    ax.set_ylabel("Δg_eff = g(low-mass) - g(high-mass) [(km/s)²/kpc]")
    ax.set_title("Environment-dependent gravity enhancement")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Δv_φ vs R
    ax = axes[0, 1]
    for env in ["void", "dense"]:
        subset = results[results.environment == env]
        ax.scatter(subset.R_bin, subset.delta_v_phi,
                  label=env.capitalize(), alpha=0.6, s=subset.N_pairs)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Galactocentric Radius (kpc)")
    ax.set_ylabel("Δv_φ = v_φ(low-mass) - v_φ(high-mass) [km/s]")
    ax.set_title("Velocity difference")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Distribution of Δg_eff
    ax = axes[1, 0]
    for env in ["void", "dense"]:
        subset = results[results.environment == env]
        ax.hist(subset.delta_g_eff, bins=20, alpha=0.5, label=env.capitalize())
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Δg_eff [(km/s)²/kpc]")
    ax.set_ylabel("Number of bins")
    ax.set_title("Distribution of Δg_eff")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Mass ratio vs Δg_eff
    ax = axes[1, 1]
    results["mass_ratio"] = results.mass_high / results.mass_low
    for env in ["void", "dense"]:
        subset = results[results.environment == env]
        ax.scatter(subset.mass_ratio, subset.delta_g_eff,
                  label=env.capitalize(), alpha=0.6, s=subset.N_pairs)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Mass ratio (high/low)")
    ax.set_ylabel("Δg_eff [(km/s)²/kpc]")
    ax.set_title("Effect vs mass contrast")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "gaia_mass_environment_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {output_file}")
    
    plt.close()


def main():
    """Main analysis pipeline."""
    print(f"\n{'='*70}")
    print(" GAIA MASS-ENVIRONMENT TEST ")
    print(" Testing cooperative response / void-gate mechanism ")
    print(f"{'='*70}\n")
    
    # Configuration
    config = GaiaTestConfig()
    config.input_file = "gaia_test/gaia_144k.csv"  # Adjust path as needed
    
    # Check if input file exists
    if not Path(config.input_file).exists():
        print(f"ERROR: Input file not found: {config.input_file}")
        print("\nPlease provide Gaia data with columns:")
        print("  ra, dec, parallax, pmra, pmdec, rv, bp_rp, Gmag")
        print("  Optional: parallax_error, rv_error, ruwe, mass")
        return 1
    
    # Load and clean data
    df = load_and_clean_gaia_data(config)
    
    # Transform to Galactocentric
    R, phi, z, v_R, v_phi, v_z, x, y = transform_to_galactocentric(df, config)
    
    # Estimate masses
    mass = estimate_stellar_masses(df)
    
    # Compute local density (environment proxy)
    rho_local, decile = compute_local_density(x, y, z, config)
    
    # Select outer disk sample
    data = select_outer_disk_sample(R, phi, z, v_R, v_phi, v_z, mass, rho_local, decile, config)
    
    # Match pairs and compute Δg_eff
    results = match_pairs_and_compute_delta_g(data, config)
    
    # Save results
    output_dir = Path("gaia_test/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "matched_pairs_results.csv", index=False)
    print(f"\n✓ Saved results: {output_dir / 'matched_pairs_results.csv'}")
    
    # Analyze
    analyze_results(results)
    
    # Plot
    plot_results(results, output_dir)
    
    print(f"\n{'='*70}")
    print(" ANALYSIS COMPLETE ")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

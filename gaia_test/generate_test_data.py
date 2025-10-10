#!/usr/bin/env python3
"""
Generate Synthetic Gaia-like Data for Testing

Creates a mock catalog of stars with realistic kinematics
to test the mass-environment analysis pipeline.

This is ONLY for testing the code structure - NOT for real science!
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_gaia_data(n_stars=10000, output_file="gaia_test/gaia_test_10k.csv"):
    """
    Generate synthetic Gaia-like catalog.
    
    Creates stars in outer Milky Way disk with:
    - Realistic spatial distribution
    - Flat rotation curve
    - Small velocity dispersion
    - Mass-magnitude relation
    """
    print(f"Generating {n_stars} synthetic stars...")
    
    np.random.seed(42)
    
    # Galactocentric radius distribution (outer disk)
    R_gal = np.random.uniform(15.0, 25.0, n_stars)  # kpc
    phi_gal = np.random.uniform(0, 2*np.pi, n_stars)  # radians
    z_gal = np.random.normal(0, 0.3, n_stars)  # kpc (thin disk)
    
    # Cartesian galactocentric coordinates
    x_gal = R_gal * np.cos(phi_gal)
    y_gal = R_gal * np.sin(phi_gal)
    
    # Velocities: flat rotation curve + dispersion
    v_circ = 220.0  # km/s (flat)
    sigma_v = 30.0  # km/s dispersion
    
    v_R_gal = np.random.normal(0, sigma_v, n_stars)
    v_phi_gal = v_circ + np.random.normal(0, sigma_v, n_stars)
    v_z_gal = np.random.normal(0, sigma_v/2, n_stars)
    
    # Convert to heliocentric
    R_sun = 8.178  # kpc
    z_sun = 0.0208  # kpc
    
    x_helio = x_gal - R_sun
    y_helio = y_gal
    z_helio = z_gal - z_sun
    
    # Heliocentric cylindrical distance and coordinates
    dist_helio = np.sqrt(x_helio**2 + y_helio**2 + z_helio**2)  # kpc
    
    # RA, Dec (very approximate - just for testing)
    l = np.arctan2(y_helio, x_helio)  # Galactic longitude
    b = np.arcsin(z_helio / dist_helio)  # Galactic latitude
    
    # Convert to RA, Dec (very rough approximation)
    ra = np.degrees(l) + 180  # degrees (wrap around)
    ra = ra % 360
    dec = np.degrees(b)  # degrees
    
    # Parallax (ensure good quality for outer disk stars)
    parallax = 1.0 / (dist_helio * 1000)  # mas (1/distance in pc)
    # Better parallax for nearby outer disk stars
    parallax = np.clip(parallax, 0.02, 2.0)  # mas
    parallax_error = parallax * 0.05  # 5% error
    
    # Proper motions (very simplified - just for structure)
    # In reality, need full transformation from galactocentric to heliocentric
    pmra = np.random.normal(0, 2, n_stars)  # mas/yr
    pmdec = np.random.normal(0, 2, n_stars)  # mas/yr
    
    # Radial velocity (approximate)
    rv = v_R_gal + np.random.normal(0, 5, n_stars)  # km/s
    rv_error = np.abs(np.random.normal(2.0, 1.0, n_stars))  # km/s
    
    # Stellar properties
    # Mass: bimodal distribution (low-mass and solar-mass)
    mass_type = np.random.choice([0, 1], size=n_stars, p=[0.6, 0.4])
    mass = np.where(
        mass_type == 0,
        np.random.uniform(0.6, 0.9, n_stars),  # Low-mass stars
        np.random.uniform(0.9, 1.5, n_stars)   # Solar-mass stars
    )
    
    # Color-magnitude relation (crude)
    bp_rp = 2.0 - 0.6 * (mass - 0.6)  # Red for low-mass, blue for high-mass
    bp_rp += np.random.normal(0, 0.1, n_stars)  # Add scatter
    bp_rp = np.clip(bp_rp, 0.5, 2.5)
    
    # Apparent magnitude (distance-dependent)
    M_G = 4.8 + 2.5 * np.log10(mass / 1.0)  # Absolute magnitude
    Gmag = M_G + 5 * np.log10(dist_helio * 1000 / 10)  # Apparent magnitude
    Gmag += np.random.normal(0, 0.05, n_stars)  # Photometric error
    
    # Quality indicator (RUWE)
    ruwe = np.random.gamma(2, 0.5, n_stars)  # Typical RUWE distribution
    
    # Create DataFrame
    df = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'parallax': parallax,
        'parallax_error': parallax_error,
        'pmra': pmra,
        'pmdec': pmdec,
        'rv': rv,
        'rv_error': rv_error,
        'bp_rp': bp_rp,
        'Gmag': Gmag,
        'mass': mass,  # Include true masses for validation
        'ruwe': ruwe,
        # Store true galactocentric coords for validation
        'R_gal_true': R_gal,
        'v_phi_true': v_phi_gal,
    })
    
    # Quality cuts (simulate Gaia selection - more permissive for outer disk)
    quality = (
        (df.parallax > 0.05) &  # More permissive for distant stars
        (df.parallax / df.parallax_error > 5) &  # Lower S/N threshold
        (df.rv_error < 10.0) &  # More permissive RV errors
        (df.ruwe < 2.0) &  # More permissive RUWE
        (df.Gmag < 19)  # Fainter limit
    )
    
    df = df[quality].copy()
    print(f"After quality cuts: {len(df)} stars")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved synthetic catalog: {output_path}")
    print(f"\nColumn summary:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-stars", type=int, default=10000,
                   help="Number of stars to generate (default: 10000)")
    ap.add_argument("--output", type=str, default="gaia_test/gaia_test_10k.csv",
                   help="Output CSV file")
    args = ap.parse_args()
    
    df = generate_synthetic_gaia_data(args.n_stars, args.output)
    
    print(f"\n{'='*60}")
    print("Synthetic data ready for testing!")
    print(f"{'='*60}")
    print("\nTo test the analysis:")
    print(f"  1. Edit test_gaia_mass_environment.py")
    print(f"  2. Set config.input_file = '{args.output}'")
    print(f"  3. Run: py -u gaia_test/test_gaia_mass_environment.py")
    print("\n⚠️  WARNING: This is synthetic data - results are NOT real science!")

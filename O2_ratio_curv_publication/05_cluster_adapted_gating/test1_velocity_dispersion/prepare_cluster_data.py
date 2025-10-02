"""
prepare_cluster_data.py

Extract median velocity dispersions from cluster temperature profiles.
Prepares data for Test 1: Velocity Dispersion Gating parameter fitting.

Usage:
    python prepare_cluster_data.py

Output:
    cluster_sigma_data.csv - Median σ for each cluster

Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from velocity_dispersion_model import (
    load_cluster_temperature_profile,
    compute_median_temperature,
    compute_velocity_dispersion_from_temperature
)


def prepare_cluster_velocity_dispersions(data_dir, output_file):
    """
    Prepare velocity dispersion data for all clusters.
    
    Parameters:
    -----------
    data_dir : Path
        Path to data/clusters/ directory
    output_file : Path
        Output CSV file path
    
    Returns:
    --------
    df : DataFrame
        Cluster velocity dispersion data
    """
    # Clusters with temperature profiles and lensing data
    clusters = [
        {
            'name': 'ABELL_1689',
            'z': 0.183,
            'theta_E_obs': 47.0,  # arcsec, observed Einstein radius
            'description': 'Massive lensing cluster'
        },
        {
            'name': 'A2029',
            'z': 0.077,
            'theta_E_obs': 28.0,  # arcsec
            'description': 'Massive relaxed cluster'
        },
        {
            'name': 'A478',
            'z': 0.088,
            'theta_E_obs': 31.0,  # arcsec
            'description': 'Intermediate-mass cluster'
        },
        {
            'name': 'ABELL_0426',
            'z': 0.018,
            'theta_E_obs': None,  # No strong lensing (too nearby/disturbed)
            'description': 'Perseus - Cool-core, AGN feedback'
        }
    ]
    
    results = []
    
    print("=" * 70)
    print("Extracting Velocity Dispersions from Cluster Temperature Profiles")
    print("=" * 70)
    
    for cluster_info in clusters:
        cluster_name = cluster_info['name']
        print(f"\n{cluster_name} ({cluster_info['description']})")
        print("-" * 70)
        
        try:
            # Load temperature profile
            temp_profile = load_cluster_temperature_profile(cluster_name, data_dir)
            print(f"  Loaded temperature profile: {len(temp_profile)} points")
            print(f"  Radial range: {temp_profile['r_kpc'].min():.1f} - {temp_profile['r_kpc'].max():.1f} kpc")
            
            # Compute median temperature in virial region
            # Avoid central AGN (r > 10 kpc) and outer accretion (r < 500 kpc)
            kT_median, sigma_median = compute_median_temperature(
                temp_profile, 
                r_min_kpc=10, 
                r_max_kpc=500
            )
            
            print(f"  Median temperature (10-500 kpc): {kT_median:.2f} keV")
            print(f"  Median velocity dispersion: {sigma_median:.1f} km/s")
            
            # Also compute characteristic temperature at different radii
            kT_inner, sigma_inner = compute_median_temperature(
                temp_profile, r_min_kpc=10, r_max_kpc=100
            )
            print(f"  Inner region (10-100 kpc): kT={kT_inner:.2f} keV, σ={sigma_inner:.1f} km/s")
            
            try:
                kT_outer, sigma_outer = compute_median_temperature(
                    temp_profile, r_min_kpc=100, r_max_kpc=500
                )
                print(f"  Outer region (100-500 kpc): kT={kT_outer:.2f} keV, σ={sigma_outer:.1f} km/s")
            except ValueError:
                print(f"  Outer region: Not enough data points")
                kT_outer, sigma_outer = kT_median, sigma_median
            
            # Store results
            result = {
                'cluster': cluster_name,
                'z': cluster_info['z'],
                'theta_E_obs_arcsec': cluster_info['theta_E_obs'],
                'kT_median_keV': kT_median,
                'sigma_median_kms': sigma_median,
                'kT_inner_keV': kT_inner,
                'sigma_inner_kms': sigma_inner,
                'kT_outer_keV': kT_outer,
                'sigma_outer_kms': sigma_outer,
                'n_temp_points': len(temp_profile),
                'r_min_kpc': temp_profile['r_kpc'].min(),
                'r_max_kpc': temp_profile['r_kpc'].max(),
                'description': cluster_info['description'],
                'has_lensing': cluster_info['theta_E_obs'] is not None
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    # Filter to lensing clusters only
    df_lensing = df[df['has_lensing'] == True]
    
    print(f"\nAll clusters (n={len(df)}):")
    print(f"  Median σ: {df['sigma_median_kms'].median():.1f} km/s")
    print(f"  Range: {df['sigma_median_kms'].min():.1f} - {df['sigma_median_kms'].max():.1f} km/s")
    
    print(f"\nLensing clusters only (n={len(df_lensing)}):")
    print(f"  Median σ: {df_lensing['sigma_median_kms'].median():.1f} km/s")
    print(f"  Range: {df_lensing['sigma_median_kms'].min():.1f} - {df_lensing['sigma_median_kms'].max():.1f} km/s")
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   {len(df)} clusters, {len(df_lensing)} with lensing data")
    
    print("\n" + "=" * 70)
    
    return df


def estimate_galaxy_velocity_dispersions():
    """
    Estimate typical velocity dispersions for SPARC galaxies.
    
    Returns:
    --------
    dict : Velocity dispersion estimates
    """
    print("\n" + "=" * 70)
    print("Estimating Galaxy Velocity Dispersions (for reference)")
    print("=" * 70)
    
    # Typical values for different galaxy types
    estimates = {
        'dwarf_irregular': {
            'M_200_Msun': 1e10,
            'R_200_kpc': 50,
            'sigma_kms': None
        },
        'small_spiral': {
            'M_200_Msun': 5e10,
            'R_200_kpc': 80,
            'sigma_kms': None
        },
        'milky_way_like': {
            'M_200_Msun': 1e12,
            'R_200_kpc': 200,
            'sigma_kms': None
        },
        'massive_spiral': {
            'M_200_Msun': 5e12,
            'R_200_kpc': 300,
            'sigma_kms': None
        }
    }
    
    from velocity_dispersion_model import compute_velocity_dispersion_from_virial
    
    for gal_type, props in estimates.items():
        sigma = compute_velocity_dispersion_from_virial(
            props['M_200_Msun'], 
            props['R_200_kpc']
        )
        props['sigma_kms'] = sigma
        
        print(f"\n{gal_type.replace('_', ' ').title()}:")
        print(f"  M_200 = {props['M_200_Msun']:.1e} Msun")
        print(f"  R_200 = {props['R_200_kpc']:.0f} kpc")
        print(f"  σ = {sigma:.1f} km/s")
    
    # Summary
    sigmas = [props['sigma_kms'] for props in estimates.values()]
    print(f"\nTypical galaxy range: {min(sigmas):.0f} - {max(sigmas):.0f} km/s")
    print(f"Median: {np.median(sigmas):.0f} km/s")
    
    # For SPARC, use conservative estimate
    sparc_sigma_typical = 120.0  # km/s, between small and MW-like
    print(f"\n✅ Recommended for SPARC galaxies: σ = {sparc_sigma_typical:.0f} km/s")
    print(f"   (Will test sensitivity with σ = 80, 100, 150, 200 km/s)")
    
    print("=" * 70)
    
    return estimates


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = base_dir / "data" / "clusters"
    output_file = Path(__file__).parent / "cluster_sigma_data.csv"
    
    print(f"\nData directory: {data_dir}")
    print(f"Output file: {output_file}\n")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        print("   Please check the path or run from correct location.")
        sys.exit(1)
    
    # Prepare cluster data
    df_clusters = prepare_cluster_velocity_dispersions(data_dir, output_file)
    
    # Estimate galaxy dispersions (for reference)
    galaxy_estimates = estimate_galaxy_velocity_dispersions()
    
    print("\n✅ Data preparation complete!")
    print(f"\nNext step: Run parameter fitting")
    print(f"  python fit_sigma_model.py")

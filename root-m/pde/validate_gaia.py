#!/usr/bin/env python3
"""
validate_gaia.py

Validate Gaia data slices before large-scale processing.
Check for data quality, coverage, and physical plausibility.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

def validate_gaia_slice(slice_file: Path) -> Dict:
    """Validate a single Gaia data slice."""
    
    try:
        df = pd.read_parquet(slice_file)
    except Exception as e:
        return {'valid': False, 'error': str(e)}
    
    result = {
        'file': slice_file.name,
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Check columns
    expected_cols = ['source_id', 'X_gc_kpc', 'Y_gc_kpc', 'Z_gc_kpc', 
                    'R_kpc', 'phi_rad', 'z_kpc', 'v_R_kms', 'v_phi_kms', 
                    'v_z_kms', 'v_obs', 'sigma_v', 'quality_flag']
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        result['issues'].append(f"Missing columns: {missing_cols}")
        result['valid'] = False
        return result
    
    # Basic statistics
    result['stats']['n_stars'] = len(df)
    result['stats']['quality_good'] = (df['quality_flag'] == 1).sum() if 'quality_flag' in df.columns else 0
    
    # Check physical ranges
    if 'R_kpc' in df.columns:
        R = df['R_kpc'].values
        result['stats']['R_min'] = R.min()
        result['stats']['R_max'] = R.max()
        result['stats']['R_median'] = np.median(R)
        
        # Check for unphysical distances
        if np.any(R < 0):
            result['issues'].append("Negative radial distances")
        if np.any(R > 100):  # More than 100 kpc is suspicious
            result['issues'].append(f"Very large distances: max={R.max():.1f} kpc")
    
    # Check velocities
    for v_col in ['v_R_kms', 'v_phi_kms', 'v_z_kms']:
        if v_col in df.columns:
            v = df[v_col].values
            v_clean = v[~np.isnan(v)]
            if len(v_clean) > 0:
                result['stats'][f'{v_col}_mean'] = np.mean(v_clean)
                result['stats'][f'{v_col}_std'] = np.std(v_clean)
                
                # Check for unphysical velocities
                if np.any(np.abs(v_clean) > 1000):  # > 1000 km/s is extreme
                    result['issues'].append(f"Extreme {v_col}: max={np.max(np.abs(v_clean)):.0f} km/s")
    
    # Check galactic coordinates
    if 'X_gc_kpc' in df.columns and 'Y_gc_kpc' in df.columns:
        # Sun is at approximately X=8.2 kpc, Y=0, Z=0
        X_sun = df['X_gc_kpc'].values
        if np.median(np.abs(X_sun - 8.2)) > 50:
            result['issues'].append("Coordinates don't match solar position")
    
    return result


def validate_all_slices(gaia_dir: Path) -> List[Dict]:
    """Validate all Gaia data slices."""
    
    print("\n" + "="*60)
    print("GAIA DATA VALIDATION")
    print("="*60)
    
    slice_files = sorted(gaia_dir.glob("processed_L*.parquet"))
    
    if not slice_files:
        print(f"❌ No processed Gaia slices found in {gaia_dir}")
        return []
    
    print(f"Found {len(slice_files)} processed slices")
    
    all_results = []
    total_stars = 0
    total_good = 0
    
    for slice_file in slice_files:
        print(f"\nValidating {slice_file.name}...")
        result = validate_gaia_slice(slice_file)
        all_results.append(result)
        
        if result['valid']:
            n_stars = result['stats'].get('n_stars', 0)
            n_good = result['stats'].get('quality_good', 0)
            total_stars += n_stars
            total_good += n_good
            
            print(f"  ✅ {n_stars:,} stars ({n_good:,} good quality)")
            if result['stats'].get('R_median'):
                print(f"  Median R: {result['stats']['R_median']:.1f} kpc")
            
            if result['issues']:
                print(f"  ⚠️ Issues: {', '.join(result['issues'])}")
        else:
            print(f"  ❌ Invalid: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total slices: {len(slice_files)}")
    print(f"Valid slices: {sum(1 for r in all_results if r['valid'])}")
    print(f"Total stars: {total_stars:,}")
    print(f"Good quality: {total_good:,} ({100*total_good/max(total_stars,1):.1f}%)")
    
    return all_results


def plot_gaia_coverage(all_results: List[Dict], output_dir: Path):
    """Plot Gaia data coverage and quality."""
    
    if not all_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract longitude ranges from filenames
    longitudes = []
    star_counts = []
    quality_fractions = []
    median_distances = []
    
    for result in all_results:
        if result['valid']:
            # Parse longitude from filename like "processed_L210-240.parquet"
            fname = result['file']
            if 'L' in fname:
                parts = fname.split('L')[1].split('.')[0].split('-')
                if len(parts) == 2:
                    l_start = float(parts[0])
                    l_mid = l_start + 15  # Mid-point of 30-degree slice
                    longitudes.append(l_mid)
                    star_counts.append(result['stats'].get('n_stars', 0))
                    
                    n_total = result['stats'].get('n_stars', 1)
                    n_good = result['stats'].get('quality_good', 0)
                    quality_fractions.append(n_good / n_total)
                    
                    median_distances.append(result['stats'].get('R_median', 0))
    
    # Plot 1: Star counts by longitude
    ax = axes[0, 0]
    ax.bar(longitudes, star_counts, width=25, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Galactic Longitude [deg]')
    ax.set_ylabel('Star Count')
    ax.set_title('Gaia Coverage by Longitude')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Quality fraction by longitude
    ax = axes[0, 1]
    ax.bar(longitudes, quality_fractions, width=25, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Galactic Longitude [deg]')
    ax.set_ylabel('Quality Fraction')
    ax.set_title('Data Quality by Longitude')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Median distance by longitude
    ax = axes[1, 0]
    ax.plot(longitudes, median_distances, 'o-', color='blue', alpha=0.7)
    ax.set_xlabel('Galactic Longitude [deg]')
    ax.set_ylabel('Median Distance [kpc]')
    ax.set_title('Distance Distribution by Longitude')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Gaia Data Summary:
    
    Total Slices: {len(all_results)}
    Valid Slices: {sum(1 for r in all_results if r['valid'])}
    Total Stars: {sum(star_counts):,}
    
    Coverage:
    Min stars/slice: {min(star_counts):,}
    Max stars/slice: {max(star_counts):,}
    Mean stars/slice: {np.mean(star_counts):.0f}
    
    Quality:
    Mean quality fraction: {np.mean(quality_fractions):.2%}
    
    Distances:
    Mean median distance: {np.mean(median_distances):.1f} kpc
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace')
    
    plt.suptitle('Gaia Data Validation Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'gaia_validation.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run Gaia validation."""
    
    # Setup paths
    base_path = Path("C:/Users/henry/dev/GravityCalculator")
    gaia_dir = base_path / "data" / "gaia_sky_slices"
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Validate all slices
    all_results = validate_all_slices(gaia_dir)
    
    # Create plots
    if all_results:
        print("\nGenerating validation plots...")
        plot_gaia_coverage(all_results, output_dir)
        print(f"Plots saved to {output_dir}")
    
    # Check for specific issues
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if all_results:
        # Check if we have full sky coverage
        n_slices = len([r for r in all_results if r['valid']])
        expected_slices = 12  # 360 degrees / 30 degrees per slice
        
        if n_slices < expected_slices:
            print(f"⚠️ Incomplete sky coverage: {n_slices}/{expected_slices} slices")
            print("   Consider downloading missing longitude ranges")
        else:
            print(f"✅ Full sky coverage: {n_slices} slices")
        
        # Check data quality
        avg_quality = np.mean([r['stats'].get('quality_good', 0) / max(r['stats'].get('n_stars', 1), 1) 
                              for r in all_results if r['valid']])
        
        if avg_quality < 0.5:
            print(f"⚠️ Low average data quality: {avg_quality:.1%}")
            print("   Consider stricter quality cuts or additional data")
        else:
            print(f"✅ Good average data quality: {avg_quality:.1%}")
        
        print("\n✅ Gaia data ready for GPU processing")
    else:
        print("❌ No valid Gaia data found")
        print("   Download from: https://gea.esac.esa.int/archive/")
    
    return all_results


if __name__ == "__main__":
    results = main()
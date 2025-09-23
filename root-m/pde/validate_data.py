#!/usr/bin/env python3
"""
validate_data.py

Comprehensive data validation to ensure our raw observational data is correct.
Checks for:
- Unit consistency
- Physical plausibility
- Data completeness
- Sign conventions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Physical constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun
c_km_s = 299792.458  # km/s
MU_E = 1.17  # mean molecular weight per electron
M_P_G = 1.67262192369e-24  # proton mass in g
KPC_CM = 3.0856775814913673e21  # kpc to cm
MSUN_G = 1.988409870698051e33  # solar mass in g


def validate_cluster_data(cluster_name: str, base_path: Path) -> Dict:
    """Validate cluster data for consistency and physical plausibility."""
    
    cluster_dir = base_path / cluster_name
    results = {
        'cluster': cluster_name,
        'valid': True,
        'issues': [],
        'data': {}
    }
    
    print(f"\n{'='*60}")
    print(f"Validating {cluster_name}")
    print('='*60)
    
    # 1. Check gas profile
    gas_file = cluster_dir / "gas_profile.csv"
    if not gas_file.exists():
        results['valid'] = False
        results['issues'].append("Missing gas_profile.csv")
        return results
    
    gas_df = pd.read_csv(gas_file)
    print("\n[Gas Profile]")
    print(f"  Columns: {list(gas_df.columns)}")
    print(f"  Rows: {len(gas_df)}")
    
    # Check radius values
    if 'r_kpc' in gas_df.columns:
        r = gas_df['r_kpc'].values
        if np.any(r < 0):
            results['issues'].append("Negative radius values found")
            results['valid'] = False
        if not np.all(np.diff(r) > 0):
            results['issues'].append("Radius not monotonically increasing")
        print(f"  Radius range: {r.min():.1f} - {r.max():.1f} kpc")
    
    # Check density values
    if 'n_e_cm3' in gas_df.columns:
        ne = gas_df['n_e_cm3'].values
        if np.any(ne < 0):
            results['issues'].append("Negative electron density")
            results['valid'] = False
        if np.any(ne > 10):  # Typical max is ~0.1 cm^-3
            results['issues'].append(f"Suspiciously high n_e: max={ne.max():.2e} cm^-3")
        
        # Convert to mass density and check
        rho_gas = MU_E * M_P_G * ne * (KPC_CM**3) / MSUN_G
        print(f"  n_e range: {ne.min():.2e} - {ne.max():.2e} cm^-3")
        print(f"  ρ_gas range: {rho_gas.min():.2e} - {rho_gas.max():.2e} Msun/kpc^3")
        
        # Check for proper decline with radius
        if len(r) > 10:
            r_test = r[len(r)//2]
            ne_inner = ne[r < r_test].mean()
            ne_outer = ne[r > r_test].mean()
            if ne_outer > ne_inner:
                results['issues'].append("Density increasing with radius")
        
        results['data']['ne'] = ne
        results['data']['r'] = r
        results['data']['rho_gas'] = rho_gas
    
    # 2. Check temperature profile
    temp_file = cluster_dir / "temp_profile.csv"
    if temp_file.exists():
        temp_df = pd.read_csv(temp_file)
        print("\n[Temperature Profile]")
        print(f"  Columns: {list(temp_df.columns)}")
        
        if 'kT_keV' in temp_df.columns:
            kT = temp_df['kT_keV'].values
            if np.any(kT < 0):
                results['issues'].append("Negative temperature")
                results['valid'] = False
            if np.any(kT > 20):  # Very rare to exceed 15 keV
                results['issues'].append(f"Suspiciously high kT: max={kT.max():.1f} keV")
            print(f"  kT range: {kT.min():.2f} - {kT.max():.2f} keV")
            
            results['data']['kT'] = kT
            results['data']['r_temp'] = temp_df['r_kpc'].values if 'r_kpc' in temp_df.columns else None
    
    # 3. Check stellar profile if present
    stars_file = cluster_dir / "stars_profile.csv"
    if stars_file.exists():
        stars_df = pd.read_csv(stars_file)
        print("\n[Stellar Profile]")
        print(f"  Columns: {list(stars_df.columns)}")
        
        if 'rho_star_Msun_per_kpc3' in stars_df.columns:
            rho_star = stars_df['rho_star_Msun_per_kpc3'].values
            if np.any(rho_star < 0):
                results['issues'].append("Negative stellar density")
                results['valid'] = False
            print(f"  ρ_star range: {rho_star.min():.2e} - {rho_star.max():.2e} Msun/kpc^3")
            
            # Check star/gas ratio
            if 'rho_gas' in results['data']:
                r_star = stars_df['r_kpc'].values
                rho_gas_interp = np.interp(r_star, r, results['data']['rho_gas'])
                star_gas_ratio = rho_star / (rho_gas_interp + 1e-10)
                if np.any(star_gas_ratio > 10):  # Stars typically < 10x gas in clusters
                    results['issues'].append(f"High star/gas ratio: max={star_gas_ratio.max():.1f}")
    
    # 4. Check clumping profile if present
    clump_file = cluster_dir / "clump_profile.csv"
    if clump_file.exists():
        clump_df = pd.read_csv(clump_file)
        print("\n[Clumping Profile]")
        
        if 'C' in clump_df.columns:
            C = clump_df['C'].values
            if np.any(C < 1):
                results['issues'].append("Clumping factor < 1")
                results['valid'] = False
            if np.any(C > 10):
                results['issues'].append(f"Very high clumping: max={C.max():.1f}")
            print(f"  Clumping range: {C.min():.2f} - {C.max():.2f}")
    
    # 5. Calculate total mass and check plausibility
    if 'r' in results['data'] and 'rho_gas' in results['data']:
        r = results['data']['r']
        rho = results['data']['rho_gas']
        
        # Integrate mass
        M_enc = np.zeros_like(r)
        for i in range(1, len(r)):
            dr = r[i] - r[i-1]
            rho_avg = 0.5 * (rho[i] + rho[i-1])
            dV = 4 * np.pi * r[i]**2 * dr
            M_enc[i] = M_enc[i-1] + rho_avg * dV
        
        # Check against typical cluster masses
        M_200 = M_enc[r < 200][-1] if np.any(r < 200) else M_enc[-1]
        print(f"\n[Mass Check]")
        print(f"  M(<200 kpc) = {M_200:.2e} Msun")
        
        if M_200 < 1e11 or M_200 > 1e15:
            results['issues'].append(f"Unusual cluster mass: {M_200:.2e} Msun")
        
        results['data']['M_enc'] = M_enc
    
    # Summary
    print(f"\n[Validation Summary]")
    if results['valid']:
        print(f"  ✅ Data appears valid")
    else:
        print(f"  ❌ Critical issues found")
    
    if results['issues']:
        print(f"  ⚠️ Issues: {len(results['issues'])}")
        for issue in results['issues']:
            print(f"    - {issue}")
    
    return results


def validate_sparc_data(sparc_dir: Path) -> Dict:
    """Validate SPARC galaxy data."""
    
    results = {
        'valid': True,
        'issues': [],
        'galaxies': []
    }
    
    print(f"\n{'='*60}")
    print("Validating SPARC Data")
    print('='*60)
    
    # Check for main SPARC catalog
    catalog_file = sparc_dir / "MassModels_Lelli2016c.txt"
    if not catalog_file.exists():
        results['valid'] = False
        results['issues'].append("Missing SPARC catalog file")
        return results
    
    # Read catalog
    with open(catalog_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header to find column indices
    header_line = None
    for i, line in enumerate(lines):
        if 'Galaxy' in line and 'D' in line:
            header_line = i
            break
    
    if header_line is None:
        results['issues'].append("Could not parse SPARC catalog header")
        return results
    
    # Process galaxies
    galaxy_count = 0
    issue_count = 0
    
    for line in lines[header_line + 1:]:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) > 0:
                galaxy_name = parts[0]
                galaxy_count += 1
                
                # Check individual galaxy file
                galaxy_file = sparc_dir / f"{galaxy_name}_rotmod.txt"
                if not galaxy_file.exists():
                    issue_count += 1
                    continue
                
                # Read and validate galaxy data
                try:
                    galaxy_data = pd.read_csv(galaxy_file, sep='\\s+', comment='#', 
                                            names=['R_kpc', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul'])
                    
                    # Check for negative velocities
                    if np.any(galaxy_data['Vobs'] < 0):
                        results['issues'].append(f"{galaxy_name}: Negative observed velocity")
                        issue_count += 1
                    
                    # Check for reasonable velocity range (10-500 km/s typical)
                    max_v = galaxy_data['Vobs'].max()
                    if max_v > 500 or max_v < 10:
                        results['issues'].append(f"{galaxy_name}: Unusual max velocity {max_v:.1f} km/s")
                    
                    # Calculate baryon velocity
                    Vbar = np.sqrt(galaxy_data['Vgas']**2 + galaxy_data['Vdisk']**2 + 
                                  galaxy_data['Vbul']**2)
                    
                    # Check baryon vs observed
                    if np.any(Vbar > galaxy_data['Vobs'] * 2):
                        results['issues'].append(f"{galaxy_name}: Vbar > 2*Vobs")
                    
                    results['galaxies'].append({
                        'name': galaxy_name,
                        'max_v': max_v,
                        'n_points': len(galaxy_data)
                    })
                    
                except Exception as e:
                    results['issues'].append(f"{galaxy_name}: Parse error - {e}")
                    issue_count += 1
    
    print(f"  Galaxies found: {galaxy_count}")
    print(f"  Galaxies with issues: {issue_count}")
    print(f"  Success rate: {100*(1-issue_count/galaxy_count):.1f}%")
    
    return results


def check_gaia_data(gaia_dir: Path) -> Dict:
    """Check for Gaia DR3 data availability."""
    
    results = {
        'available': False,
        'files': [],
        'total_size': 0
    }
    
    print(f"\n{'='*60}")
    print("Checking Gaia Data")
    print('='*60)
    
    if not gaia_dir.exists():
        print(f"  ❌ Gaia directory not found: {gaia_dir}")
        return results
    
    # Look for Gaia files
    gaia_files = list(gaia_dir.glob("*.fits")) + list(gaia_dir.glob("*.csv")) + \
                 list(gaia_dir.glob("*.parquet"))
    
    if gaia_files:
        results['available'] = True
        for f in gaia_files:
            size_mb = f.stat().st_size / (1024**2)
            results['files'].append({'name': f.name, 'size_mb': size_mb})
            results['total_size'] += size_mb
        
        print(f"  ✅ Found {len(gaia_files)} Gaia files")
        print(f"  Total size: {results['total_size']:.1f} MB")
    else:
        print(f"  ⚠️ No Gaia data files found")
        print(f"  Download from: https://gea.esac.esa.int/archive/")
    
    return results


def plot_validation_results(all_results: Dict, output_dir: Path):
    """Create diagnostic plots for data validation."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Density profiles for all clusters
    ax = axes[0, 0]
    for cluster_name, result in all_results['clusters'].items():
        if 'r' in result['data'] and 'rho_gas' in result['data']:
            r = result['data']['r']
            rho = result['data']['rho_gas']
            ax.loglog(r, rho, label=cluster_name, alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ_gas [Msun/kpc³]')
    ax.set_title('Gas Density Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Temperature profiles
    ax = axes[0, 1]
    for cluster_name, result in all_results['clusters'].items():
        if 'r_temp' in result['data'] and 'kT' in result['data']:
            r = result['data']['r_temp']
            kT = result['data']['kT']
            if r is not None:
                ax.semilogy(r, kT, 'o-', label=cluster_name, alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('kT [keV]')
    ax.set_title('Temperature Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mass profiles
    ax = axes[0, 2]
    for cluster_name, result in all_results['clusters'].items():
        if 'r' in result['data'] and 'M_enc' in result['data']:
            r = result['data']['r']
            M = result['data']['M_enc']
            ax.loglog(r[r>0], M[r>0], label=cluster_name, alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('M(<r) [Msun]')
    ax.set_title('Enclosed Mass')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: SPARC galaxy statistics
    ax = axes[1, 0]
    if 'sparc' in all_results and all_results['sparc'] and all_results['sparc']['galaxies']:
        max_velocities = [g['max_v'] for g in all_results['sparc']['galaxies']]
        ax.hist(max_velocities, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Max V_obs [km/s]')
        ax.set_ylabel('Count')
        ax.set_title(f'SPARC Galaxy Velocities (N={len(max_velocities)})')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Data quality summary
    ax = axes[1, 1]
    valid_clusters = sum(1 for r in all_results['clusters'].values() if r['valid'])
    total_clusters = len(all_results['clusters'])
    issue_clusters = sum(1 for r in all_results['clusters'].values() if r['issues'])
    
    categories = ['Valid', 'With Issues', 'Invalid']
    counts = [valid_clusters, issue_clusters, total_clusters - valid_clusters]
    colors = ['green', 'orange', 'red']
    ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Cluster Data Quality')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Issue types
    ax = axes[1, 2]
    issue_types = {}
    for result in all_results['clusters'].values():
        for issue in result['issues']:
            key = issue.split(':')[0] if ':' in issue else issue
            issue_types[key] = issue_types.get(key, 0) + 1
    
    if issue_types:
        ax.barh(list(issue_types.keys()), list(issue_types.values()), alpha=0.7)
        ax.set_xlabel('Count')
        ax.set_title('Issue Types')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Data Validation Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Run comprehensive data validation."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA VALIDATION")
    print("="*60)
    
    # Setup paths
    base_path = Path("C:/Users/henry/dev/GravityCalculator")
    cluster_path = base_path / "data" / "clusters"
    sparc_path = base_path / "data" / "sparc"
    gaia_path = base_path / "data" / "gaia"
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    all_results = {
        'clusters': {},
        'sparc': None,
        'gaia': None
    }
    
    # 1. Validate cluster data
    print("\n[VALIDATING CLUSTERS]")
    clusters = ['ABELL_0426', 'ABELL_1689', 'A1795', 'A2029', 'A478']
    
    for cluster in clusters:
        cluster_dir = cluster_path / cluster
        if cluster_dir.exists():
            result = validate_cluster_data(cluster, cluster_path)
            all_results['clusters'][cluster] = result
    
    # 2. Validate SPARC data
    print("\n[VALIDATING SPARC]")
    if sparc_path.exists():
        all_results['sparc'] = validate_sparc_data(sparc_path)
    else:
        print(f"  ⚠️ SPARC directory not found: {sparc_path}")
    
    # 3. Check Gaia data
    print("\n[CHECKING GAIA]")
    all_results['gaia'] = check_gaia_data(gaia_path)
    
    # 4. Generate plots
    print("\n[GENERATING PLOTS]")
    plot_validation_results(all_results, output_dir)
    
    # 5. Save validation report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'summary': {
            'clusters_validated': len(all_results['clusters']),
            'clusters_valid': sum(1 for r in all_results['clusters'].values() if r['valid']),
            'total_issues': sum(len(r['issues']) for r in all_results['clusters'].values()),
            'sparc_available': all_results['sparc'] is not None,
            'gaia_available': all_results['gaia']['available'] if all_results['gaia'] else False
        },
        'details': all_results
    }
    
    with open(output_dir / 'validation_report.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(v) for v in obj]
            else:
                return obj
        
        json.dump(convert_arrays(report), f, indent=2)
    
    # 6. Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  Clusters validated: {report['summary']['clusters_validated']}")
    print(f"  Valid clusters: {report['summary']['clusters_valid']}")
    print(f"  Total issues: {report['summary']['total_issues']}")
    print(f"  SPARC data: {'✅ Available' if report['summary']['sparc_available'] else '❌ Not found'}")
    print(f"  Gaia data: {'✅ Available' if report['summary']['gaia_available'] else '❌ Not found'}")
    print(f"\n  Report saved to: {output_dir / 'validation_report.json'}")
    print(f"  Plots saved to: {output_dir / 'data_validation.png'}")
    
    # 7. Recommendations
    print("\n[RECOMMENDATIONS]")
    if report['summary']['total_issues'] > 0:
        print("  ⚠️ Address data issues before proceeding with analysis")
    
    if not report['summary']['gaia_available']:
        print("  📥 Download Gaia DR3 data for Milky Way constraints")
        print("     Visit: https://gea.esac.esa.int/archive/")
    
    print("  🚀 Consider GPU acceleration for large-scale processing")
    print("  📊 Implement cross-validation across all data sources")
    
    return all_results


if __name__ == "__main__":
    results = main()
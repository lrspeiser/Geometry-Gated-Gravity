#!/usr/bin/env python3
"""
fix_cluster_data.py

Fix ordering issues in cluster data files.
Some files have radius in decreasing order which causes negative masses.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shutil

def fix_cluster_profiles(cluster_name: str, base_path: Path):
    """Fix ordering in all profile files for a cluster."""
    
    cluster_dir = base_path / cluster_name
    print(f"\n{'='*60}")
    print(f"Fixing {cluster_name}")
    print('='*60)
    
    # Backup original files
    backup_dir = cluster_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    files_to_fix = [
        "gas_profile.csv",
        "temp_profile.csv",
        "stars_profile.csv",
        "clump_profile.csv"
    ]
    
    for filename in files_to_fix:
        filepath = cluster_dir / filename
        if not filepath.exists():
            print(f"  ⚠️ {filename} not found")
            continue
            
        # Read the file
        df = pd.read_csv(filepath)
        
        # Check if it has r_kpc column
        if 'r_kpc' not in df.columns:
            print(f"  ⚠️ {filename} has no r_kpc column")
            continue
        
        # Check if radius is in descending order
        r = df['r_kpc'].values
        if len(r) > 1 and r[0] > r[-1]:
            print(f"  📝 {filename}: Reversing order (was descending)")
            
            # Backup original
            backup_path = backup_dir / filename
            shutil.copy2(filepath, backup_path)
            print(f"     Backed up to {backup_path}")
            
            # Reverse the dataframe
            df_fixed = df.iloc[::-1].reset_index(drop=True)
            
            # Save fixed version
            df_fixed.to_csv(filepath, index=False)
            print(f"     Fixed and saved")
            
            # Verify
            r_new = df_fixed['r_kpc'].values
            if np.all(np.diff(r_new) > 0):
                print(f"     ✅ Radius now monotonically increasing")
            else:
                print(f"     ⚠️ Still has non-monotonic radius")
        else:
            if np.all(np.diff(r) > 0):
                print(f"  ✅ {filename}: Already in correct order")
            else:
                print(f"  ⚠️ {filename}: Non-monotonic but not simply reversed")
                
                # Find duplicates or problematic points
                problematic = []
                for i in range(1, len(r)):
                    if r[i] <= r[i-1]:
                        problematic.append(i)
                
                if problematic:
                    print(f"     Problematic indices: {problematic[:5]}...")


def verify_mass_integration(cluster_name: str, base_path: Path):
    """Verify that mass integration gives positive values."""
    
    cluster_dir = base_path / cluster_name
    gas_file = cluster_dir / "gas_profile.csv"
    
    if not gas_file.exists():
        return
    
    gas_df = pd.read_csv(gas_file)
    
    if 'r_kpc' not in gas_df.columns:
        return
        
    r = gas_df['r_kpc'].values
    
    # Convert density
    if 'n_e_cm3' in gas_df.columns:
        ne = gas_df['n_e_cm3'].values
        MU_E = 1.17
        M_P_G = 1.67262192369e-24
        KPC_CM = 3.0856775814913673e21
        MSUN_G = 1.988409870698051e33
        rho_gas = MU_E * M_P_G * ne * (KPC_CM**3) / MSUN_G
    elif 'rho_gas_Msun_per_kpc3' in gas_df.columns:
        rho_gas = gas_df['rho_gas_Msun_per_kpc3'].values
    else:
        return
    
    # Check stellar density if available
    stars_file = cluster_dir / "stars_profile.csv"
    if stars_file.exists():
        stars_df = pd.read_csv(stars_file)
        if 'rho_star_Msun_per_kpc3' in stars_df.columns:
            r_star = stars_df['r_kpc'].values
            rho_star_raw = stars_df['rho_star_Msun_per_kpc3'].values
            # Interpolate to gas grid
            rho_star = np.interp(r, r_star, rho_star_raw, left=0, right=0)
            rho_total = rho_gas + rho_star
        else:
            rho_total = rho_gas
    else:
        rho_total = rho_gas
    
    # Integrate mass
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        dr = r[i] - r[i-1]
        if dr <= 0:
            print(f"  ❌ Negative dr at index {i}: {dr}")
            continue
        rho_avg = 0.5 * (rho_total[i] + rho_total[i-1])
        dV = 4 * np.pi * r[i]**2 * dr
        M_enc[i] = M_enc[i-1] + rho_avg * dV
    
    # Check results
    print(f"\n  Mass Integration Check:")
    print(f"    M(<10 kpc) = {M_enc[r < 10][-1] if np.any(r < 10) else 0:.2e} Msun")
    print(f"    M(<100 kpc) = {M_enc[r < 100][-1] if np.any(r < 100) else 0:.2e} Msun")
    print(f"    M(<200 kpc) = {M_enc[r < 200][-1] if np.any(r < 200) else 0:.2e} Msun")
    
    if np.any(M_enc < 0):
        print(f"    ❌ Negative masses found!")
    else:
        print(f"    ✅ All masses positive")
    
    return M_enc


def main():
    """Fix all cluster data files."""
    
    print("\n" + "="*60)
    print("FIXING CLUSTER DATA FILES")
    print("="*60)
    
    base_path = Path("C:/Users/henry/dev/GravityCalculator/data/clusters")
    clusters = ['ABELL_0426', 'ABELL_1689', 'A1795', 'A2029', 'A478']
    
    for cluster in clusters:
        cluster_dir = base_path / cluster
        if not cluster_dir.exists():
            print(f"\n⚠️ {cluster} directory not found")
            continue
            
        # Fix ordering
        fix_cluster_profiles(cluster, base_path)
        
        # Verify mass integration
        verify_mass_integration(cluster, base_path)
    
    print("\n" + "="*60)
    print("DATA FIXING COMPLETE")
    print("="*60)
    print("\nBackups created in each cluster's backup/ directory")
    print("Original validation should now be re-run to confirm fixes")


if __name__ == "__main__":
    main()
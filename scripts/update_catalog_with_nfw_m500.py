"""
Update Master Catalog with NFW M_500 Conversions
=================================================

Replaces fixed M_500 values with proper NFW conversions from
Umetsu+2016 M_200c and c_200c measurements.

Author: GravityCalculator
Date: 2025-01-19
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
from pathlib import Path
from core.nfw_mass_conversion import M200c_to_M500c

# Paths
CATALOG_PATH = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'
NFW_DATA_PATH = Path(__file__).parent.parent / 'data' / 'literature' / 'nfw_params.json'
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog_nfw.csv'

print("="*70)
print("UPDATE CATALOG WITH NFW M_500 CONVERSIONS")
print("="*70)

# Load catalog
catalog = pd.read_csv(CATALOG_PATH)
print(f"\nLoaded {len(catalog)} clusters from master catalog")

# Load NFW data
with open(NFW_DATA_PATH, 'r') as f:
    nfw_data = json.load(f)

# Build lookup dict (normalize cluster names)
def normalize_name(name):
    return str(name).upper().replace(' ', '').replace('-', '').replace('_', '').replace('.', '')

nfw_lookup = {}
for cluster in nfw_data['clusters']:
    key = normalize_name(cluster['cluster_id'])
    nfw_lookup[key] = cluster

print(f"Loaded NFW parameters for {len(nfw_lookup)} clusters")

# Update catalog
updated_count = 0
results_table = []

for idx, row in catalog.iterrows():
    cluster_name = row['cluster_name']
    norm_name = normalize_name(cluster_name)
    
    if norm_name in nfw_lookup:
        nfw = nfw_lookup[norm_name]
        
        # Extract NFW parameters
        M_200c = nfw['M_200c_Msun']
        c_200c = nfw['c_200c']
        z = nfw['z_lens']
        
        # Convert to M_500c
        M_500c, R_500c, c_500c = M200c_to_M500c(M_200c, c_200c, z)
        
        # Update catalog
        M_500_old = catalog.at[idx, 'M_500_Msun']
        R_500_old = catalog.at[idx, 'R_500_kpc']
        
        catalog.at[idx, 'M_500_Msun'] = M_500c
        catalog.at[idx, 'R_500_kpc'] = R_500c
        
        # Add NFW metadata as new columns
        if 'M_200c_Msun' not in catalog.columns:
            catalog['M_200c_Msun'] = float('nan')
            catalog['c_200c'] = float('nan')
            catalog['c_500c'] = float('nan')
        
        catalog.at[idx, 'M_200c_Msun'] = M_200c
        catalog.at[idx, 'c_200c'] = c_200c
        catalog.at[idx, 'c_500c'] = c_500c
        
        updated_count += 1
        
        # Log changes
        delta_M = (M_500c - M_500_old) / M_500_old * 100
        delta_R = (R_500c - R_500_old) / R_500_old * 100
        
        results_table.append({
            'cluster': cluster_name,
            'M_500_old': M_500_old,
            'M_500_new': M_500c,
            'delta_M_pct': delta_M,
            'R_500_old': R_500_old,
            'R_500_new': R_500c,
            'delta_R_pct': delta_R,
            'c_200c': c_200c,
            'c_500c': c_500c
        })

print(f"\n[Results] Updated {updated_count} clusters with NFW conversions")
print("\nChanges in M_500 and R_500:")
print("-" * 100)

df_results = pd.DataFrame(results_table)
for _, res in df_results.iterrows():
    print(f"  {res['cluster']:<15}: "
          f"M_500: {res['M_500_old']:.2e} → {res['M_500_new']:.2e} ({res['delta_M_pct']:+.1f}%), "
          f"R_500: {res['R_500_old']:.0f} → {res['R_500_new']:.0f} kpc ({res['delta_R_pct']:+.1f}%)")

# Save updated catalog
catalog.to_csv(OUTPUT_PATH, index=False)
print(f"\n[Output] Saved updated catalog to: {OUTPUT_PATH}")

# Summary statistics
print("\nSummary Statistics:")
print(f"  Mean ΔM_500: {df_results['delta_M_pct'].mean():+.1f}%")
print(f"  Mean ΔR_500: {df_results['delta_R_pct'].mean():+.1f}%")
print(f"  c_200c range: {df_results['c_200c'].min():.1f} - {df_results['c_200c'].max():.1f}")
print(f"  c_500c range: {df_results['c_500c'].min():.2f} - {df_results['c_500c'].max():.2f}")

print("\n" + "="*70)
print("UPDATE COMPLETE")
print("="*70)

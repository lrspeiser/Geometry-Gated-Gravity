#!/usr/bin/env python3
"""Quick test of SPARC data loading"""
from pathlib import Path
import sys
sys.path.insert(0, 'many_path_model')

from validation_suite import ValidationSuite

print("Testing SPARC data loading...")
vs = ValidationSuite(Path('results/test'), load_sparc=True)

print(f"\nTotal galaxies: {len(vs.sparc_data)}")
print(f"With rotation curves: {(vs.sparc_data['r_all'].notna()).sum()}")

# Show sample
sample = vs.sparc_data[vs.sparc_data['r_all'].notna()].head(3)
for idx, row in sample.iterrows():
    print(f"\n{row['Galaxy']}: {len(row['r_all'])} points")
    print(f"  r range: {row['r_all'][0]:.2f} - {row['r_all'][-1]:.2f} kpc")
    print(f"  v range: {row['v_all'].min():.1f} - {row['v_all'].max():.1f} km/s")

print("\n✅ SPARC data loading working!")

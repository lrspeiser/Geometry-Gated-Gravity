#!/usr/bin/env python3
"""
Standalone parser for SPARC disk parameters (no dependencies on many_path_gravity).
"""
from pathlib import Path
import numpy as np
import json


# Morphological type mappings
HUBBLE_TYPES = {
    0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
    6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'
}

# Type groupings
TYPE_GROUPS = {
    'early': ['S0', 'Sa', 'Sab', 'Sb'],
    'intermediate': ['Sbc', 'Sc'],
    'late': ['Scd', 'Sd', 'Sdm', 'Sm', 'Im', 'BCD']
}


def determine_type_group(hubble_name):
    """Determine if galaxy is early/intermediate/late type."""
    for group, types in TYPE_GROUPS.items():
        if hubble_name in types:
            return group
    return 'unknown'


def load_sparc_master_simple(master_file):
    """Load SPARC master table (simplified, no dependencies)."""
    galaxies = {}
    
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    # Find data section: the actual data starts after the final long dash line
    # which appears after the Notes section, around line 97-98
    data_start = 0
    for i, line in enumerate(lines):
        # Look for a long line of dashes (length > 70)
        if line.startswith('---') and len(line.strip()) > 70:
            data_start = i + 1
    
    # Parse data lines
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('#'):
            continue
        if len(line) < 75:
            continue
        
        try:
            # Parse fixed-width format
            name = line[0:11].strip()
            if not name:
                continue
            
            # Empirically determined column positions (spec offsets don't match actual data)
            hubble_type = int(line[12:14].strip()) if len(line) > 13 and line[12:14].strip() else -1
            distance_str = line[15:21].strip() if len(line) > 20 else ''
            distance = float(distance_str) if distance_str else None
            
            # Rdisk: empirically found at position 73-78
            Rdisk_str = line[73:78].strip() if len(line) > 77 else ''
            R_d_kpc = float(Rdisk_str) if Rdisk_str else None
            
            # SBdisk: empirically found at position 78-86
            SBdisk_str = line[78:86].strip() if len(line) > 85 else ''
            SBdisk = float(SBdisk_str) if SBdisk_str else None
            
            galaxies[name] = {
                'hubble_type': hubble_type,
                'hubble_name': HUBBLE_TYPES.get(hubble_type, 'Unknown'),
                'distance_mpc': distance,
                'R_d_kpc': R_d_kpc,
                'SBdisk_solLum_pc2': SBdisk
            }
        except (ValueError, IndexError) as e:
            continue
    
    return galaxies


def estimate_sigma0(SBdisk, M_L=0.5):
    """Convert SBdisk to Sigma0."""
    if SBdisk is None or np.isnan(SBdisk) or SBdisk <= 0:
        return None
    return SBdisk * M_L


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_file', type=Path,
                       default=Path('data/SPARC_Lelli2016c.mrt'))
    parser.add_argument('--output', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'))
    args = parser.parse_args()
    
    print(f"Loading SPARC master table from: {args.master_file}")
    galaxies = load_sparc_master_simple(args.master_file)
    print(f"Loaded {len(galaxies)} galaxies")
    
    disk_params = {}
    for name, info in galaxies.items():
        hubble_name = info['hubble_name']
        SBdisk = info.get('SBdisk_solLum_pc2')
        
        disk_params[name] = {
            'name': name,
            'hubble_type': hubble_name,
            'type_group': determine_type_group(hubble_name),
            'distance_mpc': info.get('distance_mpc'),
            'R_d_kpc': info.get('R_d_kpc'),
            'SBdisk_solLum_pc2': SBdisk,
            'Sigma0': estimate_sigma0(SBdisk) if SBdisk else None
        }
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(disk_params, f, indent=2)
    print(f"[OK] Saved to: {args.output}")
    
    # Statistics
    valid_Rd = [p['R_d_kpc'] for p in disk_params.values() 
                if p['R_d_kpc'] and not np.isnan(p['R_d_kpc'])]
    valid_Sig = [p['Sigma0'] for p in disk_params.values() 
                 if p['Sigma0'] and not np.isnan(p['Sigma0'])]
    
    print(f"\nStatistics:")
    print(f"  R_d available: {len(valid_Rd)}/{len(disk_params)}")
    if valid_Rd:
        print(f"    Range: {np.min(valid_Rd):.2f} - {np.max(valid_Rd):.2f} kpc")
        print(f"    Median: {np.median(valid_Rd):.2f} kpc")
    
    print(f"  Sigma0 estimated: {len(valid_Sig)}/{len(disk_params)}")
    if valid_Sig:
        print(f"    Range: {np.min(valid_Sig):.1f} - {np.max(valid_Sig):.1f} M_sun/pc^2")
        print(f"    Median: {np.median(valid_Sig):.1f} M_sun/pc^2")
    
    # Show examples
    print(f"\nExample galaxies:")
    print("-" * 70)
    print(f"{'Name':12s} {'Type':8s} {'R_d':8s} {'Sigma0':10s}")
    print("-" * 70)
    for i, (name, p) in enumerate(disk_params.items()):
        if i >= 10:
            break
        Rd = f"{p['R_d_kpc']:.2f}" if p['R_d_kpc'] else "N/A"
        Sig = f"{p['Sigma0']:.1f}" if p['Sigma0'] else "N/A"
        print(f"{name:12s} {p['hubble_type']:8s} {Rd:8s} {Sig:10s}")


if __name__ == '__main__':
    main()

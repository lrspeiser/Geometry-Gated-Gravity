#!/usr/bin/env python3
"""
Parse disk parameters from SPARC master table for use in extended B/T laws.

Extracts:
- R_d: Disk scale length (kpc)
- Sigma0: Central surface density (M_sun/pc^2)
- mu0: Central surface brightness (mag/arcsec^2)
- L_d: Disk luminosity ([3.6] band)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from sparc_stratified_test import load_sparc_master_table, determine_type_group


def estimate_sigma0_from_SB(SBdisk_solLum_pc2, M_L=0.5):
    """
    Estimate central surface density from luminosity surface brightness.
    
    Args:
        SBdisk_solLum_pc2: Disk central surface brightness (solar luminosities/pc^2, [3.6])
        M_L: Mass-to-light ratio at [3.6] microns (default 0.5 for disks)
    
    Returns:
        Sigma0 in M_sun/pc^2
    
    Note: SPARC provides SBdisk in solLum/pc^2, which is the disk central 
          surface brightness in linear units (not magnitudes).
    """
    if SBdisk_solLum_pc2 is None or np.isnan(SBdisk_solLum_pc2) or SBdisk_solLum_pc2 <= 0:
        return None
    
    # Direct conversion: Sigma0 = SBdisk × M/L
    # SBdisk is already in L_sun/pc^2, so just multiply by M/L ratio
    Sigma0 = SBdisk_solLum_pc2 * M_L
    
    return Sigma0


def parse_sparc_disk_parameters(master_file: Path, output_file: Path = None):
    """
    Parse disk parameters from SPARC master table.
    
    Args:
        master_file: Path to SPARC_Lelli2016c.mrt
        output_file: Optional path to save JSON output
    
    Returns:
        Dictionary mapping galaxy_name -> disk_params
    """
    master_info = load_sparc_master_table(master_file)
    
    print(f"Loaded {len(master_info)} galaxies from master table")
    
    disk_params = {}
    
    for gal_name, info in master_info.items():
        hubble_name = info.get('hubble_name', 'Unknown')
        R_d_kpc = info.get('R_d_kpc', None)
        SBdisk = info.get('SBdisk_solLum_pc2', None)
        
        # Convert R_d and SBdisk to float, handling None
        try:
            R_d_kpc = float(R_d_kpc) if R_d_kpc is not None else None
        except (TypeError, ValueError):
            R_d_kpc = None
        
        try:
            SBdisk = float(SBdisk) if SBdisk is not None else None
        except (TypeError, ValueError):
            SBdisk = None
        
        # Sigma0 estimation from SBdisk
        if SBdisk is not None and not np.isnan(SBdisk) and SBdisk > 0:
            Sigma0 = estimate_sigma0_from_SB(SBdisk, M_L=0.5)
        else:
            Sigma0 = None
        
        params = {
            'name': gal_name,
            'R_d_kpc': R_d_kpc,
            'distance_mpc': info.get('distance_mpc', None),
            'hubble_type': hubble_name,
            'type_group': determine_type_group(hubble_name),
            'Sigma0': Sigma0,
            'SBdisk_solLum_pc2': SBdisk
        }
        
        disk_params[gal_name] = params
    
    # Save if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(disk_params, f, indent=2)
        print(f"[OK] Saved disk parameters to: {output_file}")
    
    # Summary statistics
    valid_Rd = [p['R_d_kpc'] for p in disk_params.values() if p['R_d_kpc'] and not np.isnan(p['R_d_kpc'])]
    valid_Sigma = [p['Sigma0'] for p in disk_params.values() if p['Sigma0'] and not np.isnan(p['Sigma0'])]
    
    print(f"\nParsed {len(disk_params)} galaxies:")
    print(f"  R_d available: {len(valid_Rd)}/{len(disk_params)}")
    if valid_Rd:
        print(f"    Range: {np.min(valid_Rd):.2f} - {np.max(valid_Rd):.2f} kpc")
        print(f"    Median: {np.median(valid_Rd):.2f} kpc")
    
    print(f"  Sigma0 estimated: {len(valid_Sigma)}/{len(disk_params)}")
    if valid_Sigma:
        print(f"    Range: {np.min(valid_Sigma):.1f} - {np.max(valid_Sigma):.1f} M_sun/pc^2")
        print(f"    Median: {np.median(valid_Sigma):.1f} M_sun/pc^2")
    
    return disk_params


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Parse SPARC disk parameters')
    parser.add_argument('--master_file', type=Path,
                       default=Path('data/SPARC_Lelli2016c.mrt'),
                       help='Path to SPARC master table')
    parser.add_argument('--output', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'),
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    disk_params = parse_sparc_disk_parameters(args.master_file, args.output)
    
    # Show a few examples
    print("\nExample galaxies:")
    print("-" * 80)
    print(f"{'Name':12s} {'Type':8s} {'R_d (kpc)':10s} {'Sigma0':12s}")
    print("-" * 80)
    
    for i, (name, params) in enumerate(disk_params.items()):
        if i >= 10:
            break
        Rd_str = f"{params['R_d_kpc']:.2f}" if params['R_d_kpc'] else "N/A"
        Sig_str = f"{params['Sigma0']:.1f}" if params['Sigma0'] else "N/A"
        hubble = params['hubble_type'] or "?"
        print(f"{name:12s} {hubble:8s} {Rd_str:10s} {Sig_str:12s}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test script to debug why optimization isn't working
"""

import numpy as np
from pathlib import Path
from solve_g3_production import (
    G3SolverProduction, G3Parameters, SolverConfig, SystemType,
    load_galaxy_data, voxelize_galaxy, extract_rotation_curve
)

def test_different_params():
    """Test if different parameters actually give different chi2 values."""
    
    # Load a test galaxy
    galaxy_name = "CamB"
    data_dir = Path("data")
    galaxy_data = load_galaxy_data(galaxy_name, data_dir)
    
    if galaxy_data is None:
        print(f"Could not load {galaxy_name}")
        return
    
    # Voxelize
    nx, ny, nz = 128, 128, 16
    rho_3d, dx = voxelize_galaxy(galaxy_data, nx, ny, nz)
    galaxy_data['dx'] = dx
    
    # Initialize solver
    solver = G3SolverProduction(nx, ny, nz, dx)
    config = SolverConfig(verbose=False, max_cycles=30, tol=1e-4)
    
    # Test different parameter sets
    param_sets = [
        {"S0": 1.5, "rc": 10.0, "name": "Default"},
        {"S0": 0.5, "rc": 5.0, "name": "Low values"},
        {"S0": 3.0, "rc": 20.0, "name": "High values"},
        {"S0": 2.0, "rc": 15.0, "name": "Medium values"},
        {"S0": 2.945, "rc": 20.518, "name": "Optimization result"},
    ]
    
    print(f"\nTesting {galaxy_name} with different parameters:")
    print("="*60)
    
    for param_set in param_sets:
        params = G3Parameters(
            S0=param_set["S0"],
            rc_kpc=param_set["rc"],
            rc_gamma=0.3,
            sigma_beta=0.5,
            g_sat_kms2_per_kpc=100.0,
            n_sat=2.0,
            use_sigma_screen=False,
            omega=1.2
        )
        
        try:
            # Solve
            result = solver.solve(rho_3d, SystemType.GALAXY_DISK, params, config)
            
            if result['converged']:
                result['dx'] = dx
                curve_data = extract_rotation_curve(result, galaxy_data)
                chi2 = curve_data['chi2_reduced']
                print(f"{param_set['name']:20s}: S0={param_set['S0']:.2f}, rc={param_set['rc']:.1f} kpc → χ²/dof = {chi2:.2f}")
            else:
                print(f"{param_set['name']:20s}: Did not converge")
                
        except Exception as e:
            print(f"{param_set['name']:20s}: Error - {e}")
    
    print("\n" + "="*60)
    print("Conclusion:")
    print("If all χ² values are the same, the solver isn't responding to parameter changes.")
    print("If they differ, the optimization objective function has a bug.")

if __name__ == "__main__":
    test_different_params()
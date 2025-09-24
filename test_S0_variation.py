#!/usr/bin/env python3
"""
Test if S0 variation affects chi2
"""

import numpy as np
from pathlib import Path
from solve_g3_production import (
    G3SolverProduction, G3Parameters, SolverConfig, SystemType,
    load_galaxy_data, voxelize_galaxy, extract_rotation_curve
)

def test_S0_variation():
    """Test if different S0 values give different chi2."""
    
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
    
    # Test different S0 values with fixed other parameters
    S0_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    print(f"\nTesting {galaxy_name} with different S0 values:")
    print("="*60)
    print(f"{'S0':>8} | {'S0_eff':>8} | {'χ²/dof':>10} | {'Converged':>10}")
    print("-"*60)
    
    for S0 in S0_values:
        params = G3Parameters(
            S0=S0,
            rc_kpc=10.0,  # This won't matter based on our finding
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
                S0_eff = result['S0_eff']
                print(f"{S0:8.2f} | {S0_eff:8.4f} | {chi2:10.2f} | {'Yes':>10}")
            else:
                print(f"{S0:8.2f} | {'N/A':>8} | {'N/A':>10} | {'No':>10}")
                
        except Exception as e:
            print(f"{S0:8.2f} | Error: {e}")
    
    print("\n" + "="*60)
    print("Test complete. If S0_eff varies but χ² doesn't, ")
    print("then S0_eff isn't being used correctly in the solver.")

if __name__ == "__main__":
    test_S0_variation()
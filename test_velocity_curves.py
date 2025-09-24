#!/usr/bin/env python3
"""
Check if velocity curves actually differ with different S0 values
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from solve_g3_production import (
    G3SolverProduction, G3Parameters, SolverConfig, SystemType,
    load_galaxy_data, voxelize_galaxy, extract_rotation_curve
)

def test_velocity_curves():
    """Test if velocity curves differ with S0."""
    
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
    
    # Test with different S0 values
    S0_values = [0.5, 1.5, 5.0]
    colors = ['red', 'blue', 'green']
    
    plt.figure(figsize=(12, 8))
    
    for S0, color in zip(S0_values, colors):
        params = G3Parameters(
            S0=S0,
            rc_kpc=10.0,
            rc_gamma=0.3,
            sigma_beta=0.5,
            g_sat_kms2_per_kpc=100.0,
            n_sat=2.0,
            use_sigma_screen=False,
            omega=1.2
        )
        
        # Solve
        result = solver.solve(rho_3d, SystemType.GALAXY_DISK, params, config)
        
        if result['converged']:
            result['dx'] = dx
            curve_data = extract_rotation_curve(result, galaxy_data)
            
            chi2 = curve_data['chi2_reduced']
            S0_eff = result['S0_eff']
            
            # Plot model
            r = curve_data['r']
            v_circ = curve_data['v_circ']
            mask = (r > 0) & (v_circ > 0)
            
            plt.plot(r[mask], v_circ[mask], '-', color=color, linewidth=2,
                    label=f'S0={S0:.1f} (S0_eff={S0_eff:.3f}, χ²={chi2:.1f})')
            
            print(f"S0={S0:.1f}: S0_eff={S0_eff:.4f}, χ²/dof={chi2:.2f}")
            if np.any(v_circ > 0):
                print(f"  v_circ range: {np.min(v_circ[v_circ>0]):.1f} - {np.max(v_circ):.1f} km/s")
            else:
                print(f"  v_circ: all zeros!")
            g_mag = result['g_magnitude']
            if np.any(g_mag > 0):
                print(f"  Acceleration range: {np.min(g_mag[g_mag>0]):.2e} - {np.max(g_mag):.2e} km/s²")
            else:
                print(f"  Acceleration: all zeros!")
        else:
            print(f"S0={S0:.1f}: Did not converge")
    
    # Plot observations
    plt.errorbar(galaxy_data['r_kpc'], galaxy_data['v_obs'], yerr=galaxy_data['v_err'],
                fmt='ko', markersize=4, alpha=0.7, label='Observed', capsize=2)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Circular velocity (km/s)')
    plt.title(f'Rotation curves for {galaxy_name} with different S0 values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(galaxy_data['r_kpc']) * 1.1)
    plt.ylim(0, max(galaxy_data['v_obs']) * 1.3)
    
    # Save plot
    plt.savefig('velocity_curves_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'velocity_curves_test.png'")
    
    plt.close()
    
    print("\nConclusion:")
    print("If all curves are identical despite different S0_eff,")
    print("then the solver has a fundamental bug.")

if __name__ == "__main__":
    test_velocity_curves()
#!/usr/bin/env python3
"""
Debug what the solver is actually computing
"""

import numpy as np
from pathlib import Path
from solve_g3_production import (
    G3SolverProduction, G3Parameters, SolverConfig, SystemType,
    load_galaxy_data, voxelize_galaxy
)

def debug_solver():
    """Debug solver output."""
    
    # Create simple test case
    nx, ny, nz = 64, 64, 8
    dx = 1.0  # kpc
    
    # Create simple disk density
    rho_3d = np.zeros((nx, ny, nz))
    x = np.linspace(-32, 32, nx)
    y = np.linspace(-32, 32, ny)
    
    for i in range(nx):
        for j in range(ny):
            r = np.sqrt(x[i]**2 + y[j]**2)
            surface_density = 500.0 * np.exp(-r/5.0)  # Exponential disk
            for k in range(nz):
                z = k - nz//2
                rho_3d[i, j, k] = surface_density * np.exp(-abs(z)/0.5) * 1e3  # Convert to M_sun/kpc^3
    
    # Initialize solver
    solver = G3SolverProduction(nx, ny, nz, dx)
    
    # Set simple parameters
    params = G3Parameters(
        S0=1.0,
        rc_kpc=10.0,
        rc_gamma=0.3,
        sigma_beta=0.5,
        g_sat_kms2_per_kpc=100.0,
        n_sat=2.0,
        use_sigma_screen=False,
        omega=1.2
    )
    
    config = SolverConfig(verbose=True, max_cycles=10)
    
    print("\nInput density statistics:")
    print(f"  Shape: {rho_3d.shape}")
    print(f"  Min: {np.min(rho_3d):.2e} M_sun/kpc^3")
    print(f"  Max: {np.max(rho_3d):.2e} M_sun/kpc^3")
    print(f"  Total mass: {np.sum(rho_3d) * dx**3:.2e} M_sun")
    
    # Solve
    print("\nSolving...")
    result = solver.solve(rho_3d, SystemType.GALAXY_DISK, params, config)
    
    print("\nSolver output statistics:")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Residual: {result['residual']:.3e}")
    print(f"  S0_eff: {result['S0_eff']:.4f}")
    
    print("\nPotential (phi):")
    phi = result['phi']
    print(f"  Shape: {phi.shape}")
    print(f"  Min: {np.min(phi):.2e}")
    print(f"  Max: {np.max(phi):.2e}")
    print(f"  Non-zero elements: {np.sum(phi != 0)}/{phi.size}")
    
    print("\nGradient magnitude:")
    g_mag = result['g_magnitude']
    print(f"  Shape: {g_mag.shape}")
    print(f"  Min: {np.min(g_mag):.2e}")
    print(f"  Max: {np.max(g_mag):.2e}")
    print(f"  Non-zero elements: {np.sum(g_mag != 0)}/{g_mag.size}")
    
    print("\nAcceleration components:")
    print(f"  gx: min={np.min(result['gx']):.2e}, max={np.max(result['gx']):.2e}")
    print(f"  gy: min={np.min(result['gy']):.2e}, max={np.max(result['gy']):.2e}")
    print(f"  gz: min={np.min(result['gz']):.2e}, max={np.max(result['gz']):.2e}")
    
    print("\n" + "="*60)
    print("Analysis:")
    if np.all(phi == 0):
        print("ERROR: Potential is all zeros!")
    if np.all(g_mag == 0):
        print("ERROR: Gradient is all zeros!")
    if result['S0_eff'] == 0:
        print("ERROR: Effective coupling is zero!")

if __name__ == "__main__":
    debug_solver()
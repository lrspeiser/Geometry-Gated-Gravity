#!/usr/bin/env python3
"""
Test the working solver with actual galaxy data
"""

import numpy as np
from pathlib import Path

# Import the working solver instead of the broken production one
from solve_g3_working import G3SolverWorking as G3SolverProduction
from solve_g3_working import G3Parameters, SolverConfig, SystemType

# Import helper functions from production (these are fine)
from solve_g3_production import (
    load_galaxy_data, voxelize_galaxy, extract_rotation_curve,
    KPC_TO_KM, MSUN_PC2_TO_MSUN_KPC2
)

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_with_real_galaxy():
    """Test the working solver with real galaxy data."""
    
    logger.info("="*70)
    logger.info("TESTING WORKING SOLVER WITH REAL GALAXY DATA")
    logger.info("="*70)
    
    # Test galaxy
    galaxy_name = "CamB"
    data_dir = Path("data")
    
    # Load galaxy data
    galaxy_data = load_galaxy_data(galaxy_name, data_dir)
    if galaxy_data is None:
        logger.error(f"Could not load data for {galaxy_name}")
        return False
    
    logger.info(f"\nLoaded {galaxy_name}:")
    logger.info(f"  Data points: {len(galaxy_data['v_obs'])}")
    logger.info(f"  R range: {galaxy_data['r_kpc'].min():.1f} - {galaxy_data['r_kpc'].max():.1f} kpc")
    logger.info(f"  V range: {galaxy_data['v_obs'].min():.1f} - {galaxy_data['v_obs'].max():.1f} km/s")
    
    # Voxelize
    nx, ny, nz = 128, 128, 16
    rho_3d, dx = voxelize_galaxy(galaxy_data, nx, ny, nz)
    galaxy_data['dx'] = dx
    
    logger.info(f"\nVoxelized density:")
    logger.info(f"  Grid: {nx}×{ny}×{nz}, dx={dx:.2f} kpc")
    logger.info(f"  Total mass: {np.sum(rho_3d) * dx**3:.2e} M_sun")
    logger.info(f"  Max density: {np.max(rho_3d):.2e} M_sun/kpc^3")
    
    # Initialize solver
    solver = G3SolverProduction(nx, ny, nz, dx)
    
    # Test with different S0 values
    logger.info("\nTesting different S0 values:")
    logger.info("-"*50)
    
    S0_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    for S0 in S0_values:
        params = G3Parameters(
            S0=S0,
            rc_kpc=10.0,
            rc_gamma=0.3,
            sigma_beta=0.5,
            g_sat_kms2_per_kpc=100.0,
            n_sat=2.0,
            omega=1.0  # Lower relaxation for stability
        )
        
        config = SolverConfig(verbose=False, max_cycles=30)
        
        # Solve
        result = solver.solve(rho_3d, SystemType.GALAXY_DISK, params, config)
        result['dx'] = dx
        
        # Extract rotation curve
        curve_data = extract_rotation_curve(result, galaxy_data)
        
        chi2 = curve_data['chi2_reduced']
        max_phi = np.max(result['phi'])
        max_g = np.max(result['g_magnitude'])
        S0_eff = result['S0_eff']
        
        logger.info(f"  S0={S0:.1f} → S0_eff={S0_eff:.3f}: "
                   f"max(φ)={max_phi:.1e}, max(g)={max_g:.1e}, χ²/dof={chi2:.1f}")
        
        results.append({
            'S0': S0,
            'S0_eff': S0_eff,
            'chi2': chi2,
            'max_phi': max_phi,
            'max_g': max_g,
            'converged': result['converged']
        })
    
    # Check if results vary with S0
    chi2_values = [r['chi2'] for r in results]
    phi_values = [r['max_phi'] for r in results]
    
    chi2_variation = max(chi2_values) - min(chi2_values)
    phi_variation = max(phi_values) / (min(phi_values) + 1e-10)
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS")
    logger.info("="*70)
    
    logger.info(f"\nχ² variation: {chi2_variation:.2f} (min={min(chi2_values):.1f}, max={max(chi2_values):.1f})")
    logger.info(f"Potential variation: {phi_variation:.1f}x")
    
    if chi2_variation > 0.1 and phi_variation > 1.5:
        logger.info("\n✓ SUCCESS: Solver responds to parameter changes!")
        logger.info("  - Different S0 values produce different potentials")
        logger.info("  - Chi-squared values vary, enabling optimization")
        
        # Find best S0
        best_idx = np.argmin(chi2_values)
        best_result = results[best_idx]
        logger.info(f"\nBest fit: S0={best_result['S0']:.1f} with χ²/dof={best_result['chi2']:.1f}")
        
        return True
    else:
        logger.info("\n✗ ISSUE: Insufficient parameter sensitivity")
        return False

def compare_with_broken_solver():
    """Compare working vs broken solver."""
    
    logger.info("\n" + "="*70)
    logger.info("COMPARING WORKING VS BROKEN SOLVER")
    logger.info("="*70)
    
    # Create simple test case
    nx, ny, nz = 32, 32, 8
    dx = 1.0
    
    # Simple Gaussian density
    x = np.linspace(-16, 16, nx)
    y = np.linspace(-16, 16, ny)
    z = np.linspace(-4, 4, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    rho = 1000.0 * np.exp(-R**2/25.0)
    
    params = G3Parameters(S0=1.5)
    config = SolverConfig(verbose=False, max_cycles=20)
    
    # Test working solver
    from solve_g3_working import G3SolverWorking
    solver_working = G3SolverWorking(nx, ny, nz, dx)
    result_working = solver_working.solve(rho, params=params, config=config)
    
    logger.info("\nWorking Solver:")
    logger.info(f"  Max potential: {np.max(result_working['phi']):.3e}")
    logger.info(f"  Max gradient: {np.max(result_working['g_magnitude']):.3e}")
    logger.info(f"  S0_eff: {result_working['S0_eff']:.3f}")
    
    # Test broken solver (if you want to confirm it's still broken)
    try:
        from solve_g3_production import G3SolverProduction as BrokenSolver
        solver_broken = BrokenSolver(nx, ny, nz, dx)
        result_broken = solver_broken.solve(rho, params=params, config=config)
        
        logger.info("\nBroken Solver (production):")
        logger.info(f"  Max potential: {np.max(result_broken['phi']):.3e}")
        logger.info(f"  Max gradient: {np.max(result_broken['g_magnitude']):.3e}")
        logger.info(f"  S0_eff: {result_broken['S0_eff']:.3f}")
        
        if np.max(result_broken['phi']) == 0:
            logger.info("  ✓ Confirmed: Production solver is still broken (returns zeros)")
    except Exception as e:
        logger.info(f"\n  Could not test broken solver: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("CONCLUSION")
    logger.info("="*70)
    logger.info("\nThe working solver (solve_g3_working.py) produces non-zero results")
    logger.info("and responds to parameter changes, unlike the broken production solver.")
    logger.info("\nUse solve_g3_working.py for all future analyses.")

if __name__ == "__main__":
    # Test with real galaxy
    success = test_with_real_galaxy()
    
    # Compare solvers
    compare_with_broken_solver()
    
    if success:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Replace all imports:")
        print("   FROM: from solve_g3_production import G3SolverProduction")
        print("   TO:   from solve_g3_working import G3SolverWorking as G3SolverProduction")
        print("\n2. Re-run your optimization analysis with the working solver")
        print("\n3. The parameter optimization should now actually work!")
    else:
        print("\nFurther debugging needed")
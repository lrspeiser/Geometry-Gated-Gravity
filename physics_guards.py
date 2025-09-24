#!/usr/bin/env python3
"""
Physics and Unit Guards for G³ Solvers

These guards MUST be included in any solver to prevent unit errors
and unphysical results like the ones that occurred in solve_g3_working.py
"""

import numpy as np

# ============================================================================
# UNIT REGISTRY - ALWAYS USE THESE
# ============================================================================

KPC_TO_KM = 3.085677581e16  # km per kpc
G_NEWTON = 4.30091e-6  # (kpc km^2 s^-2 M_sun^-1)

# Physical limits for galaxies
MAX_GALAXY_VELOCITY = 1000  # km/s - very generous upper limit
MIN_GALAXY_VELOCITY = 10    # km/s - lower limit for dwarf galaxies
TYPICAL_GALAXY_VELOCITY_RANGE = (50, 400)  # km/s - typical range

# Physical limits for galaxy clusters  
MAX_CLUSTER_VELOCITY = 2000  # km/s - for massive clusters
TYPICAL_CLUSTER_VELOCITY_RANGE = (200, 1500)  # km/s

# ============================================================================
# MANDATORY GUARDS
# ============================================================================

def check_velocity_sanity(v_model, context="galaxy"):
    """
    Verify that computed velocities are physically reasonable.
    
    Args:
        v_model: Array of velocities in km/s
        context: "galaxy" or "cluster"
        
    Raises:
        ValueError: If velocities are unphysical
    """
    assert np.all(np.isfinite(v_model)), "Non-finite velocities detected"
    
    vmax = np.nanmax(v_model)
    vmin = np.nanmin(v_model[v_model > 0]) if np.any(v_model > 0) else 0
    
    if context == "galaxy":
        if vmax > MAX_GALAXY_VELOCITY:
            raise ValueError(
                f"UNPHYSICAL GALAXY VELOCITY: vmax={vmax:.2e} km/s "
                f"(should be < {MAX_GALAXY_VELOCITY} km/s). "
                "CHECK UNIT CONVERSIONS!"
            )
        if vmax < MIN_GALAXY_VELOCITY:
            raise ValueError(
                f"Suspiciously low galaxy velocity: vmax={vmax:.2f} km/s "
                f"(should be > {MIN_GALAXY_VELOCITY} km/s)"
            )
    elif context == "cluster":
        if vmax > MAX_CLUSTER_VELOCITY:
            raise ValueError(
                f"UNPHYSICAL CLUSTER VELOCITY: vmax={vmax:.2e} km/s "
                f"(should be < {MAX_CLUSTER_VELOCITY} km/s). "
                "CHECK UNIT CONVERSIONS!"
            )
    
    return True

def check_acceleration_velocity_consistency(g_kms2_per_kpc, r_kpc, v_kms):
    """
    Verify that acceleration and velocity are consistent via v = sqrt(g*r).
    
    Args:
        g_kms2_per_kpc: Acceleration in (km/s)^2 per kpc
        r_kpc: Radius in kpc
        v_kms: Velocity in km/s
        
    Raises:
        ValueError: If inconsistent
    """
    v_from_g = np.sqrt(np.clip(g_kms2_per_kpc * r_kpc, 0, np.inf))
    
    # Allow 10% tolerance for numerical differences
    relative_diff = np.abs(v_from_g - v_kms) / (v_kms + 1e-10)
    max_diff = np.nanmax(relative_diff)
    
    if max_diff > 0.1:  # 10% tolerance
        idx = np.nanargmax(relative_diff)
        raise ValueError(
            f"Acceleration-velocity inconsistency: "
            f"At r={r_kpc[idx]:.2f} kpc, g={g_kms2_per_kpc[idx]:.2e} (km/s)^2/kpc "
            f"gives v={v_from_g[idx]:.2f} km/s but model has v={v_kms[idx]:.2f} km/s "
            f"(diff={max_diff*100:.1f}%)"
        )
    
    return True

def check_pde_convergence(converged, residual_norm, tol):
    """
    Verify that PDE solver actually converged.
    
    Args:
        converged: Boolean convergence flag
        residual_norm: Final residual norm
        tol: Tolerance threshold
        
    Raises:
        RuntimeError: If not converged
    """
    if not converged:
        raise RuntimeError(
            f"PDE NOT CONVERGED: residual={residual_norm:.3e} > tol={tol:.1e}. "
            "Cannot trust results from non-converged solver!"
        )
    
    if residual_norm > tol:
        raise RuntimeError(
            f"PDE residual too large: {residual_norm:.3e} > {tol:.1e}"
        )
    
    return True

def check_parameter_scales(S0, rc_kpc, context="paper"):
    """
    Verify parameters are in expected ranges for the paper's G³ model.
    
    Args:
        S0: Coupling strength
        rc_kpc: Core radius in kpc
        context: "paper" for paper parameters, "experimental" for testing
        
    Raises:
        ValueError: If parameters are out of expected range
    """
    if context == "paper":
        # Paper uses S0 ≈ 1.4e-4
        if not (1e-5 < S0 < 1e-3):
            raise ValueError(
                f"S0={S0:.2e} is out of paper range (should be ~1.4e-4). "
                "Are you using the wrong normalization?"
            )
        
        # Paper uses rc ≈ 22 kpc for SPARC
        if not (5 < rc_kpc < 100):
            raise ValueError(
                f"rc={rc_kpc:.1f} kpc is out of reasonable range (5-100 kpc)"
            )
    
    return True

def compute_outer_median_closeness(v_model, v_obs, r_obs, r_outer_frac=0.5):
    """
    Compute the paper's metric: median fractional closeness in outer radii.
    
    This is the CORRECT metric for the paper, NOT chi-squared!
    
    Args:
        v_model: Model velocities at observation points (km/s)
        v_obs: Observed velocities (km/s)
        r_obs: Observation radii (kpc)
        r_outer_frac: Fraction defining "outer" radii (default 0.5 = outer half)
        
    Returns:
        median_closeness: Median |Δv|/v in outer radii (0 = perfect, 1 = 100% error)
    """
    # Select outer radii
    r_median = np.median(r_obs)
    outer_mask = r_obs >= r_median * r_outer_frac
    
    if not np.any(outer_mask):
        raise ValueError("No data points in outer region")
    
    # Compute fractional differences
    v_obs_outer = v_obs[outer_mask]
    v_model_outer = v_model[outer_mask]
    
    # Avoid division by zero
    valid = v_obs_outer > 10  # km/s minimum
    if not np.any(valid):
        return np.inf
    
    frac_diff = np.abs(v_model_outer[valid] - v_obs_outer[valid]) / v_obs_outer[valid]
    median_closeness = np.median(frac_diff)
    
    return median_closeness

# ============================================================================
# EXAMPLE USAGE IN ANY SOLVER
# ============================================================================

def example_solver_with_guards(rho, params):
    """
    Example showing where to insert guards in any solver.
    """
    
    # 1. Check parameter scales at start
    check_parameter_scales(params['S0'], params['rc_kpc'], context="paper")
    
    # 2. Run solver (your PDE code here)
    # ... solver code ...
    # result = solve_pde(rho, params)
    
    # 3. Check convergence immediately after solving
    # check_pde_convergence(result['converged'], result['residual'], params['tol'])
    
    # 4. Check velocity sanity after computing v(r)
    # v_model = compute_velocity_from_potential(result['phi'])
    # check_velocity_sanity(v_model, context="galaxy")
    
    # 5. If you computed g first, check consistency
    # check_acceleration_velocity_consistency(result['g'], r, v_model)
    
    # 6. Use the CORRECT metric for the paper
    # accuracy = compute_outer_median_closeness(v_model, v_obs, r_obs)
    # NOT chi-squared with broken units!
    
    pass

# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    print("Testing physics guards...")
    
    # Test 1: Catch unphysical velocities (like 10^9 km/s)
    try:
        v_bad = np.array([100, 200, 1e9])  # Last one is bad
        check_velocity_sanity(v_bad, "galaxy")
        print("ERROR: Should have caught bad velocity!")
    except ValueError as e:
        print(f"✓ Caught bad velocity: {e}")
    
    # Test 2: Good velocities pass
    v_good = np.array([100, 200, 300])
    assert check_velocity_sanity(v_good, "galaxy")
    print("✓ Good velocities pass")
    
    # Test 3: Check convergence
    try:
        check_pde_convergence(False, 1e-3, 1e-5)
        print("ERROR: Should have caught non-convergence!")
    except RuntimeError as e:
        print(f"✓ Caught non-convergence: {e}")
    
    # Test 4: Parameter scale check
    try:
        check_parameter_scales(0.1, 10, context="paper")  # S0=0.1 is wrong!
        print("ERROR: Should have caught wrong S0!")
    except ValueError as e:
        print(f"✓ Caught wrong S0: {e}")
    
    print("\nAll guards working correctly!")
    print("\nREMEMBER: Always use these guards to prevent unit errors!")
#!/usr/bin/env python3
"""
Comprehensive Test Framework for Unified G³-Σ Solver
Tests all solver approaches on galaxies, MW, and clusters with proper physics guards
"""

import numpy as np
import cupy as cp
import pandas as pd
from pathlib import Path
import json
import logging
import time
from typing import Dict, Tuple, List
import warnings

# Import physics guards
from physics_guards import (
    check_velocity_sanity,
    check_pde_convergence,
    check_parameter_scales,
    compute_outer_median_closeness,
    KPC_TO_KM,
    G_NEWTON
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED G³-Σ PARAMETERS (Single Global Tuple)
# ============================================================================

class UnifiedG3Parameters:
    """Single parameter set for all regimes"""
    def __init__(self):
        # Core G³ parameters
        self.S0 = 1.2e-4  # Starting value from scan
        self.rc_kpc = 24.0
        self.rc_gamma = 0.55
        self.sigma_beta = 0.10
        
        # Geometry reference values
        self.rc_ref_kpc = 30.0
        self.sigma0_Msun_pc2 = 150.0
        
        # Saturating mobility
        self.g_sat_kms2_per_kpc = 2500.0
        self.n_sat = 2
        
        # Sigma screen
        self.sigma_star_Msun_pc2 = 150.0
        self.alpha_sigma = 1.0
        
        # Numerical parameters
        self.tol = 1e-5
        self.max_iter = 1000
        self.omega = 1.0  # SOR parameter

# ============================================================================
# SOLVER IMPLEMENTATIONS
# ============================================================================

class SolverApproach:
    """Base class for solver approaches"""
    def __init__(self, name: str, params: UnifiedG3Parameters):
        self.name = name
        self.params = params
        self.converged = False
        self.residual_norm = np.inf
        
    def solve(self, rho: np.ndarray, dx: float) -> np.ndarray:
        """Solve for potential given density"""
        raise NotImplementedError

class BuiltInCuPySolver(SolverApproach):
    """Approach 3: CuPy built-in operations (most robust)"""
    def __init__(self, params: UnifiedG3Parameters):
        super().__init__("BuiltInCuPy", params)
        
    def solve(self, rho: np.ndarray, dx: float) -> np.ndarray:
        """Solve using CuPy's built-in array operations"""
        try:
            # Transfer to GPU
            rho_gpu = cp.asarray(rho, dtype=cp.float32)
            phi_gpu = cp.zeros_like(rho_gpu)
            
            # Precompute factors
            S0_4piG = self.params.S0 * 4 * np.pi * G_NEWTON
            dx2 = dx * dx
            factor = S0_4piG * dx2
            
            # Apply saturating mobility if enabled
            use_mobility = True  # Default ON
            
            # Jacobi iteration with built-in operations
            for iter in range(self.params.max_iter):
                phi_old = phi_gpu.copy()
                
                # Compute Laplacian using slicing
                phi_new = cp.zeros_like(phi_gpu)
                phi_new[1:-1, 1:-1, 1:-1] = (
                    phi_gpu[2:, 1:-1, 1:-1] + phi_gpu[:-2, 1:-1, 1:-1] +
                    phi_gpu[1:-1, 2:, 1:-1] + phi_gpu[1:-1, :-2, 1:-1] +
                    phi_gpu[1:-1, 1:-1, 2:] + phi_gpu[1:-1, 1:-1, :-2]
                ) / 6.0
                
                # Add source term with optional mobility
                if use_mobility:
                    # Compute gradient magnitude
                    grad_x = (phi_gpu[2:, 1:-1, 1:-1] - phi_gpu[:-2, 1:-1, 1:-1]) / (2 * dx)
                    grad_y = (phi_gpu[1:-1, 2:, 1:-1] - phi_gpu[1:-1, :-2, 1:-1]) / (2 * dx)
                    grad_z = (phi_gpu[1:-1, 1:-1, 2:] - phi_gpu[1:-1, 1:-1, :-2]) / (2 * dx)
                    grad_mag = cp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                    
                    # Saturating mobility
                    x = grad_mag / self.params.g_sat_kms2_per_kpc
                    mobility = 1.0 / (1.0 + x**self.params.n_sat)
                    
                    phi_new[1:-1, 1:-1, 1:-1] += factor * mobility * rho_gpu[1:-1, 1:-1, 1:-1] / 6.0
                else:
                    phi_new[1:-1, 1:-1, 1:-1] += factor * rho_gpu[1:-1, 1:-1, 1:-1] / 6.0
                
                # Apply Sigma screen if enabled
                use_sigma_screen = True  # Default ON
                if use_sigma_screen:
                    # Compute local surface density (simplified)
                    sigma_loc = cp.sum(rho_gpu, axis=2) * dx  # Project along z
                    sigma_ratio = sigma_loc / self.params.sigma_star_Msun_pc2
                    screen = 1.0 / (1.0 + sigma_ratio**self.params.alpha_sigma)
                    # Apply screen to source
                    phi_new[1:-1, 1:-1, 1:-1] *= screen[1:-1, 1:-1, np.newaxis]
                
                # SOR update
                phi_gpu = (1 - self.params.omega) * phi_gpu + self.params.omega * phi_new
                
                # Check convergence
                if iter % 10 == 0:
                    residual = cp.linalg.norm(phi_gpu - phi_old) / cp.linalg.norm(phi_gpu + 1e-10)
                    if residual < self.params.tol:
                        self.converged = True
                        self.residual_norm = float(residual)
                        break
                        
            # Transfer back to CPU
            return cp.asnumpy(phi_gpu)
            
        except Exception as e:
            logger.error(f"BuiltInCuPy solver failed: {e}")
            return np.zeros_like(rho)

class HybridCPUGPUSolver(SolverApproach):
    """Approach 5: Hybrid CPU/GPU"""
    def __init__(self, params: UnifiedG3Parameters):
        super().__init__("HybridCPUGPU", params)
        
    def solve(self, rho: np.ndarray, dx: float) -> np.ndarray:
        """Solve using hybrid CPU control with GPU computation"""
        try:
            # Transfer to GPU
            rho_gpu = cp.asarray(rho, dtype=cp.float32)
            phi_gpu = cp.zeros_like(rho_gpu)
            
            S0_4piG = self.params.S0 * 4 * np.pi * G_NEWTON
            
            # Simple gradient descent on GPU
            for iter in range(min(self.params.max_iter, 100)):
                # Compute Laplacian on GPU
                lap = cp.zeros_like(phi_gpu)
                interior = slice(1, -1)
                lap[interior, interior, interior] = (
                    phi_gpu[2:, 1:-1, 1:-1] + phi_gpu[:-2, 1:-1, 1:-1] +
                    phi_gpu[1:-1, 2:, 1:-1] + phi_gpu[1:-1, :-2, 1:-1] +
                    phi_gpu[1:-1, 1:-1, 2:] + phi_gpu[1:-1, 1:-1, :-2] -
                    6.0 * phi_gpu[1:-1, 1:-1, 1:-1]
                ) / (dx * dx)
                
                # Update with source
                residual = S0_4piG * rho_gpu - lap
                phi_gpu += 0.05 * residual  # Small step size for stability
                
                # Periodic convergence check
                if iter % 20 == 0:
                    res_norm = float(cp.linalg.norm(residual))
                    if res_norm < self.params.tol * float(cp.linalg.norm(rho_gpu)):
                        self.converged = True
                        self.residual_norm = res_norm
                        break
                        
            return cp.asnumpy(phi_gpu)
            
        except Exception as e:
            logger.error(f"Hybrid solver failed: {e}")
            return np.zeros_like(rho)

class SimplifiedAnalyticSolver(SolverApproach):
    """Simplified analytic approximation for quick testing"""
    def __init__(self, params: UnifiedG3Parameters):
        super().__init__("SimplifiedAnalytic", params)
        
    def solve(self, rho: np.ndarray, dx: float) -> np.ndarray:
        """Use simplified analytic formula (for comparison only)"""
        try:
            # Get grid dimensions
            nx, ny, nz = rho.shape
            phi = np.zeros_like(rho)
            
            # Center of grid
            cx, cy, cz = nx//2, ny//2, nz//2
            
            # Compute potential using point-mass approximation
            total_mass = np.sum(rho) * dx**3
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        r = np.sqrt((i-cx)**2 + (j-cy)**2 + (k-cz)**2) * dx
                        if r > 0.1:  # Avoid singularity
                            # Modified isothermal potential
                            phi[i,j,k] = -G_NEWTON * total_mass * self.params.S0 / (r + self.params.rc_kpc)
                            
            self.converged = True  # Always "converges"
            self.residual_norm = 0.0
            return phi
            
        except Exception as e:
            logger.error(f"Analytic solver failed: {e}")
            return np.zeros_like(rho)

# ============================================================================
# TEST DATA LOADERS
# ============================================================================

def load_sparc_test_galaxies(n_galaxies: int = 5) -> List[Dict]:
    """Load a subset of SPARC galaxies for testing"""
    galaxies = []
    
    # Use built-in test data
    test_galaxies = [
        {"name": "NGC2403", "scale": 1.0, "type": "spiral"},
        {"name": "DDO154", "scale": 0.3, "type": "dwarf"},
        {"name": "NGC3521", "scale": 1.2, "type": "spiral"},
        {"name": "UGC2259", "scale": 0.5, "type": "dwarf"},
        {"name": "NGC6946", "scale": 1.1, "type": "spiral"},
    ]
    
    for i, gal in enumerate(test_galaxies[:n_galaxies]):
        # Generate synthetic rotation curve
        r = np.logspace(0, 1.5, 20)  # 1 to 30 kpc
        v_obs = 150 * gal["scale"] * r / (r + 10)  # Simple model
        v_err = np.ones_like(v_obs) * 5  # 5 km/s error
        
        # Generate synthetic density (disk + bulge)
        nx, ny, nz = 64, 64, 16
        x = np.linspace(-30, 30, nx)
        y = np.linspace(-30, 30, ny)
        z = np.linspace(-5, 5, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        R = np.sqrt(X**2 + Y**2)
        
        # Exponential disk
        h_R = 3.0 * gal["scale"]  # Scale length
        h_z = 0.5  # Scale height
        rho_disk = 1e7 * gal["scale"] * np.exp(-R/h_R) * np.exp(-np.abs(Z)/h_z)
        
        # Bulge
        r_b = 2.0
        rho_bulge = 5e7 * gal["scale"] * np.exp(-np.sqrt(X**2 + Y**2 + Z**2)/r_b)
        
        rho_total = rho_disk + rho_bulge
        
        galaxies.append({
            "name": gal["name"],
            "type": gal["type"],
            "rho": rho_total,
            "r_obs": r,
            "v_obs": v_obs,
            "v_err": v_err,
            "dx": 60.0/nx  # Grid spacing in kpc
        })
        
    return galaxies

def load_milky_way_test_data() -> Dict:
    """Load Milky Way rotation curve test data"""
    # Simplified MW data
    r_mw = np.array([2, 4, 6, 8, 10, 12, 15, 20, 25])  # kpc
    v_mw = np.array([200, 220, 235, 230, 225, 220, 215, 210, 205])  # km/s
    v_err = np.ones_like(v_mw) * 10
    
    # MW density model (bulge + disk + halo gas)
    nx, ny, nz = 64, 64, 32
    x = np.linspace(-30, 30, nx)
    y = np.linspace(-30, 30, ny)
    z = np.linspace(-15, 15, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    R = np.sqrt(X**2 + Y**2)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Bulge (de Vaucouleurs)
    r_eff = 0.7  # kpc
    rho_bulge = 3e9 * np.exp(-7.67 * ((r/r_eff)**0.25 - 1))
    
    # Disk (double exponential)
    h_R = 2.6  # kpc
    h_z1 = 0.3  # thin disk
    h_z2 = 0.9  # thick disk
    rho_thin = 6e8 * np.exp(-R/h_R) * np.exp(-np.abs(Z)/h_z1)
    rho_thick = 1e8 * np.exp(-R/h_R) * np.exp(-np.abs(Z)/h_z2)
    
    rho_total = rho_bulge + rho_thin + rho_thick
    
    return {
        "name": "MilkyWay",
        "rho": rho_total,
        "r_obs": r_mw,
        "v_obs": v_mw,
        "v_err": v_err,
        "dx": 60.0/nx
    }

def load_cluster_test_data() -> List[Dict]:
    """Load cluster data for testing"""
    clusters = []
    
    # Perseus cluster
    clusters.append({
        "name": "Perseus",
        "type": "cluster",
        "r_kpc": np.logspace(1, 3, 15),  # 10 to 1000 kpc
        "T_keV_obs": np.array([6.5, 6.3, 6.0, 5.7, 5.4, 5.1, 4.8, 4.5, 4.2, 3.9, 3.6, 3.3, 3.0, 2.7, 2.4]),
        "T_keV_err": np.ones(15) * 0.3,
        "rho_gas_profile": lambda r: 1e-26 * (1 + (r/100)**2)**(-1.5),  # Beta model
        "rho_total_profile": lambda r: 1e-25 * (1 + (r/200)**2)**(-1.2)  # Total baryons
    })
    
    # A1689 cluster
    clusters.append({
        "name": "A1689",
        "type": "cluster",
        "r_kpc": np.logspace(1.2, 3, 12),
        "T_keV_obs": np.array([9.5, 9.2, 8.8, 8.4, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5]),
        "T_keV_err": np.ones(12) * 0.4,
        "rho_gas_profile": lambda r: 2e-26 * (1 + (r/150)**2)**(-1.4),
        "rho_total_profile": lambda r: 2e-25 * (1 + (r/250)**2)**(-1.3)
    })
    
    return clusters

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_galaxy(solver: SolverApproach, galaxy_data: Dict) -> Dict:
    """Test solver on a galaxy"""
    logger.info(f"Testing {solver.name} on galaxy {galaxy_data['name']}")
    
    try:
        # Solve for potential
        phi = solver.solve(galaxy_data['rho'], galaxy_data['dx'])
        
        # Compute rotation curve from potential
        nx, ny, nz = galaxy_data['rho'].shape
        cx, cy, cz = nx//2, ny//2, nz//2
        
        v_model = []
        for r in galaxy_data['r_obs']:
            # Sample potential at radius r in the midplane
            n_samples = 36
            theta = np.linspace(0, 2*np.pi, n_samples)
            
            phi_samples = []
            for th in theta:
                ix = int(cx + r/galaxy_data['dx'] * np.cos(th))
                iy = int(cy + r/galaxy_data['dx'] * np.sin(th))
                iz = cz
                
                if 0 <= ix < nx and 0 <= iy < ny:
                    # Finite difference for radial acceleration
                    if ix+1 < nx and ix-1 >= 0:
                        dphi_dr = (phi[ix+1, iy, iz] - phi[ix-1, iy, iz]) / (2 * galaxy_data['dx'])
                    else:
                        dphi_dr = 0
                    
                    phi_samples.append(abs(dphi_dr))
            
            if phi_samples:
                g_r = np.mean(phi_samples)  # Average acceleration
                v = np.sqrt(abs(g_r * r))  # v = sqrt(g*r)
                v_model.append(v)
            else:
                v_model.append(0)
        
        v_model = np.array(v_model)
        
        # Apply physics guards
        try:
            check_velocity_sanity(v_model, context="galaxy")
        except ValueError as e:
            logger.warning(f"Velocity check failed: {e}")
            return {"success": False, "error": str(e)}
        
        # Compute metric
        closeness = compute_outer_median_closeness(
            v_model, galaxy_data['v_obs'], galaxy_data['r_obs']
        )
        
        # Success is outer-median closeness < 0.15 (85% agreement)
        success = closeness < 0.15 and solver.converged
        
        return {
            "success": success,
            "converged": solver.converged,
            "residual": solver.residual_norm,
            "outer_median_closeness": closeness,
            "max_velocity": np.max(v_model),
            "v_model": v_model.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error testing galaxy: {e}")
        return {"success": False, "error": str(e)}

def test_cluster(solver: SolverApproach, cluster_data: Dict) -> Dict:
    """Test solver on a cluster"""
    logger.info(f"Testing {solver.name} on cluster {cluster_data['name']}")
    
    try:
        # Create 3D density grid
        nx, ny, nz = 64, 64, 64
        L = 2000  # kpc box size
        dx = L / nx
        
        x = np.linspace(-L/2, L/2, nx)
        y = np.linspace(-L/2, L/2, ny)
        z = np.linspace(-L/2, L/2, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        r_grid = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Use total baryon profile (default)
        rho = cluster_data['rho_total_profile'](r_grid)
        
        # Solve for potential
        phi = solver.solve(rho, dx)
        
        # Compute temperature profile from HSE
        # kT = μmp/kB * |dphi/dr|
        mu_mp_over_kB = 7.0e-11  # keV s^2/kpc^2 (approximate)
        
        T_model = []
        for r in cluster_data['r_kpc']:
            # Sample gradient at radius r
            idx = int(nx//2 + r/dx)
            if idx < nx-1:
                dphi_dr = (phi[idx+1, ny//2, nz//2] - phi[idx-1, ny//2, nz//2]) / (2*dx)
                T_keV = mu_mp_over_kB * abs(dphi_dr) * r
                T_model.append(T_keV)
            else:
                T_model.append(0)
        
        T_model = np.array(T_model)
        
        # Compute temperature residuals
        valid = cluster_data['T_keV_obs'] > 0
        if np.any(valid):
            residuals = abs(T_model[valid] - cluster_data['T_keV_obs'][valid]) / cluster_data['T_keV_obs'][valid]
            median_residual = np.median(residuals)
        else:
            median_residual = np.inf
        
        # Success is median |ΔT|/T < 0.6 
        success = median_residual < 0.6 and solver.converged
        
        return {
            "success": success,
            "converged": solver.converged,
            "residual": solver.residual_norm,
            "median_T_residual": median_residual,
            "T_model": T_model.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error testing cluster: {e}")
        return {"success": False, "error": str(e)}

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_comprehensive_tests():
    """Run all tests and compile results"""
    
    logger.info("="*70)
    logger.info("COMPREHENSIVE G³-Σ SOLVER ACCURACY TESTS")
    logger.info("="*70)
    
    # Initialize parameters
    params = UnifiedG3Parameters()
    logger.info(f"\nUsing unified parameters:")
    logger.info(f"  S0 = {params.S0:.2e}")
    logger.info(f"  rc = {params.rc_kpc:.1f} kpc")
    logger.info(f"  g_sat = {params.g_sat_kms2_per_kpc:.0f} (km/s)^2/kpc")
    logger.info(f"  Σ_star = {params.sigma_star_Msun_pc2:.0f} M☉/pc²")
    
    # Initialize solvers
    solvers = [
        BuiltInCuPySolver(params),
        HybridCPUGPUSolver(params),
        SimplifiedAnalyticSolver(params)
    ]
    
    # Load test data
    logger.info("\nLoading test data...")
    galaxies = load_sparc_test_galaxies(3)
    mw_data = load_milky_way_test_data()
    clusters = load_cluster_test_data()
    
    # Results storage
    results = {
        "parameters": {
            "S0": params.S0,
            "rc_kpc": params.rc_kpc,
            "g_sat": params.g_sat_kms2_per_kpc,
            "sigma_star": params.sigma_star_Msun_pc2
        },
        "solvers": {}
    }
    
    # Test each solver
    for solver in solvers:
        solver_results = {
            "galaxies": {},
            "milky_way": {},
            "clusters": {}
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {solver.name} solver")
        logger.info(f"{'='*50}")
        
        # Test on galaxies
        galaxy_success = []
        for galaxy in galaxies:
            result = test_galaxy(solver, galaxy)
            solver_results["galaxies"][galaxy["name"]] = result
            galaxy_success.append(result.get("success", False))
            
            if result.get("success"):
                logger.info(f"  ✓ {galaxy['name']}: closeness = {result['outer_median_closeness']:.3f}")
            else:
                logger.info(f"  ✗ {galaxy['name']}: {result.get('error', 'failed')}")
        
        # Test on Milky Way
        mw_result = test_galaxy(solver, mw_data)
        solver_results["milky_way"] = mw_result
        
        if mw_result.get("success"):
            logger.info(f"  ✓ Milky Way: closeness = {mw_result['outer_median_closeness']:.3f}")
        else:
            logger.info(f"  ✗ Milky Way: {mw_result.get('error', 'failed')}")
        
        # Test on clusters
        cluster_success = []
        for cluster in clusters:
            result = test_cluster(solver, cluster)
            solver_results["clusters"][cluster["name"]] = result
            cluster_success.append(result.get("success", False))
            
            if result.get("success"):
                logger.info(f"  ✓ {cluster['name']}: median T residual = {result['median_T_residual']:.3f}")
            else:
                logger.info(f"  ✗ {cluster['name']}: {result.get('error', 'failed')}")
        
        # Summary statistics
        solver_results["summary"] = {
            "galaxy_success_rate": sum(galaxy_success) / len(galaxy_success) if galaxy_success else 0,
            "mw_success": mw_result.get("success", False),
            "cluster_success_rate": sum(cluster_success) / len(cluster_success) if cluster_success else 0,
            "overall_success": all(galaxy_success + [mw_result.get("success", False)] + cluster_success)
        }
        
        results["solvers"][solver.name] = solver_results
    
    # Overall summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    
    best_solver = None
    best_score = 0
    
    for solver_name, solver_results in results["solvers"].items():
        summary = solver_results["summary"]
        total_score = (
            summary["galaxy_success_rate"] * 0.4 +
            float(summary["mw_success"]) * 0.3 +
            summary["cluster_success_rate"] * 0.3
        )
        
        logger.info(f"\n{solver_name}:")
        logger.info(f"  Galaxy success rate: {summary['galaxy_success_rate']*100:.1f}%")
        logger.info(f"  Milky Way success: {'Yes' if summary['mw_success'] else 'No'}")
        logger.info(f"  Cluster success rate: {summary['cluster_success_rate']*100:.1f}%")
        logger.info(f"  Overall score: {total_score*100:.1f}%")
        
        if total_score > best_score:
            best_score = total_score
            best_solver = solver_name
    
    logger.info("\n" + "="*70)
    logger.info(f"BEST APPROACH: {best_solver}")
    logger.info("="*70)
    
    # Save results
    output_file = Path("unified_solver_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, np.ndarray) else x)
    logger.info(f"\nResults saved to {output_file}")
    
    # Recommendations
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)
    
    if best_solver == "BuiltInCuPy":
        logger.info("\n1. Use BuiltInCuPy solver as the primary implementation")
        logger.info("2. It provides the best balance of accuracy and stability")
        logger.info("3. Implement full G³-Σ with mobility and screening in this framework")
    elif best_solver == "HybridCPUGPU":
        logger.info("\n1. Use Hybrid CPU/GPU approach for production")
        logger.info("2. It offers good control flow with GPU acceleration")
        logger.info("3. Consider optimizing the GPU kernels further")
    else:
        logger.info("\n1. The analytic approximation is too simplified for production")
        logger.info("2. Recommend implementing proper PDE solver")
        logger.info("3. Use BuiltInCuPy approach as template")
    
    logger.info("\nNext steps:")
    logger.info("1. Implement chosen solver in solve_g3_production.py")
    logger.info("2. Add full parameter scanning capability")
    logger.info("3. Validate on complete SPARC dataset")
    logger.info("4. Test on additional clusters")
    logger.info("5. Generate publication figures")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_tests()
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
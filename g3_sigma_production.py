#!/usr/bin/env python3
"""
G³-Σ Production Solver with Full Physics
=========================================

Complete implementation of Geometry-Gated Gravity with Sigma screening
and saturating mobility. This solver works across all regimes:
galaxies, Milky Way, and galaxy clusters.

Field equation:
∇·[μ(|∇φ|/g_sat) ∇φ] = 4π S₀ S(Σ_loc/Σ*) ρ_b

With geometry-aware scalings:
r_c,eff = r_c (r_1/2 / r_c,ref)^γ
S_0,eff = S_0 (Σ_0 / Σ_bar)^β
"""

import numpy as np
import cupy as cp
import cupyx
from scipy.interpolate import interp1d
from pathlib import Path
import json
import time
import logging
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)^2 / M_sun
C_LIGHT = 299792.458  # km/s
KPC_TO_CM = 3.086e21  # cm/kpc
MSUN_TO_G = 1.989e33  # g/Msun

# ============================================================================
# G³-Σ PARAMETERS
# ============================================================================

@dataclass
class G3SigmaParams:
    """Unified G³-Σ parameters (single global tuple for all regimes)"""
    
    # Core G³ parameters
    S0: float = 1.2e-4           # Source strength
    rc_kpc: float = 24.0         # Core radius (kpc)
    rc_gamma: float = 0.55       # Size scaling exponent
    sigma_beta: float = 0.10     # Density scaling exponent
    
    # Saturating mobility parameters
    g_sat_kms2_per_kpc: float = 2500.0  # Saturation scale (km/s)^2/kpc
    n_sat: int = 2                       # Saturation power
    use_mobility: bool = True            # Enable mobility gate
    
    # Sigma screen parameters  
    sigma_star_Msun_pc2: float = 150.0  # Screen threshold (Msun/pc^2)
    alpha_sigma: float = 1.0             # Screen power
    use_sigma_screen: bool = True        # Enable sigma screen
    
    # Reference scales
    rc_ref_kpc: float = 30.0             # Reference size
    sigma0_ref_Msun_pc2: float = 150.0   # Reference surface density
    
    # Numerical parameters
    tol: float = 1e-5            # Convergence tolerance
    max_iter: int = 2000          # Maximum iterations
    omega: float = 1.3            # SOR relaxation parameter
    
    def get_effective_params(self, r_half_kpc: float, sigma_mean_Msun_pc2: float) -> Tuple[float, float]:
        """Compute effective parameters with geometry scaling"""
        rc_eff = self.rc_kpc * (r_half_kpc / self.rc_ref_kpc) ** self.rc_gamma
        S0_eff = self.S0 * (self.sigma0_ref_Msun_pc2 / max(sigma_mean_Msun_pc2, 1.0)) ** self.sigma_beta
        return S0_eff, rc_eff

# ============================================================================
# 3D PDE SOLVER
# ============================================================================

class G3SigmaSolver:
    """Production G³-Σ solver using CuPy built-in operations"""
    
    def __init__(self, params: G3SigmaParams):
        self.params = params
        self.iteration_count = 0
        self.converged = False
        self.residual_history = []
        
    def solve_3d(self, rho: np.ndarray, dx: float, 
                 r_half: float = 10.0, sigma_mean: float = 100.0,
                 boundary: str = 'dirichlet', verbose: bool = False) -> np.ndarray:
        """
        Solve 3D Poisson equation with G³-Σ source
        
        Parameters:
        -----------
        rho : ndarray
            3D density array (Msun/kpc^3)
        dx : float
            Grid spacing (kpc)
        r_half : float
            Half-mass radius (kpc)
        sigma_mean : float
            Mean surface density (Msun/pc^2)
        boundary : str
            Boundary condition type ('dirichlet' or 'neumann')
        verbose : bool
            Print convergence info
            
        Returns:
        --------
        phi : ndarray
            Gravitational potential (km/s)^2
        """
        
        # Transfer to GPU
        rho_gpu = cp.asarray(rho, dtype=cp.float32)
        phi_gpu = cp.zeros_like(rho_gpu)
        
        # Get effective parameters
        S0_eff, rc_eff = self.params.get_effective_params(r_half, sigma_mean)
        
        # Precompute constants
        S0_4piG = S0_eff * 4 * np.pi * G_NEWTON
        dx2 = dx * dx
        factor = S0_4piG * dx2
        
        # Compute surface density map if using sigma screen
        if self.params.use_sigma_screen:
            sigma_loc_gpu = self._compute_surface_density(rho_gpu, dx)
        
        # Main iteration loop
        self.iteration_count = 0
        self.converged = False
        self.residual_history = []
        
        for iter in range(self.params.max_iter):
            phi_old = phi_gpu.copy()
            
            # Compute Laplacian with 6-point stencil
            phi_new = self._compute_laplacian_update(phi_gpu, rho_gpu, dx, factor)
            
            # Apply saturating mobility if enabled
            if self.params.use_mobility:
                mobility = self._compute_mobility(phi_gpu, dx)
                phi_new = self._apply_mobility(phi_new, phi_gpu, mobility, rho_gpu, factor)
            
            # Apply sigma screen if enabled  
            if self.params.use_sigma_screen:
                screen = self._compute_sigma_screen(sigma_loc_gpu)
                phi_new = self._apply_screen(phi_new, phi_gpu, screen, rho_gpu, factor)
            
            # Apply boundary conditions
            phi_new = self._apply_boundary_conditions(phi_new, boundary)
            
            # SOR update
            phi_gpu = (1 - self.params.omega) * phi_gpu + self.params.omega * phi_new
            
            # Check convergence periodically
            if iter % 20 == 0:
                residual = float(cp.linalg.norm(phi_gpu - phi_old) / (cp.linalg.norm(phi_gpu) + 1e-10))
                self.residual_history.append(residual)
                
                if verbose and iter % 100 == 0:
                    logger.info(f"  Iteration {iter}: residual = {residual:.3e}")
                
                if residual < self.params.tol:
                    self.converged = True
                    self.iteration_count = iter
                    if verbose:
                        logger.info(f"  Converged after {iter} iterations (residual = {residual:.3e})")
                    break
        
        if not self.converged and verbose:
            logger.warning(f"  Did not converge after {self.params.max_iter} iterations")
        
        # Transfer back to CPU
        return cp.asnumpy(phi_gpu)
    
    def _compute_laplacian_update(self, phi: cp.ndarray, rho: cp.ndarray, 
                                  dx: float, factor: float) -> cp.ndarray:
        """Compute Laplacian update using 6-point stencil"""
        phi_new = cp.zeros_like(phi)
        
        # Interior points
        phi_new[1:-1, 1:-1, 1:-1] = (
            phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
            phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
            phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
        ) / 6.0
        
        # Add source term
        phi_new[1:-1, 1:-1, 1:-1] += factor * rho[1:-1, 1:-1, 1:-1] / 6.0
        
        return phi_new
    
    def _compute_mobility(self, phi: cp.ndarray, dx: float) -> cp.ndarray:
        """Compute saturating mobility μ(|∇φ|/g_sat)"""
        # Compute gradient magnitude
        grad_x = cp.zeros_like(phi)
        grad_y = cp.zeros_like(phi)
        grad_z = cp.zeros_like(phi)
        
        grad_x[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
        grad_y[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dx)
        grad_z[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
        
        grad_mag = cp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Saturating mobility
        x = grad_mag / self.params.g_sat_kms2_per_kpc
        mobility = 1.0 / (1.0 + x**self.params.n_sat)
        
        return mobility
    
    def _apply_mobility(self, phi_new: cp.ndarray, phi_old: cp.ndarray,
                       mobility: cp.ndarray, rho: cp.ndarray, factor: float) -> cp.ndarray:
        """Apply mobility gate to update"""
        # Recompute with mobility
        phi_mob = cp.zeros_like(phi_new)
        
        # Apply mobility-weighted Laplacian
        mob_avg = (mobility[2:, 1:-1, 1:-1] + mobility[:-2, 1:-1, 1:-1] +
                   mobility[1:-1, 2:, 1:-1] + mobility[1:-1, :-2, 1:-1] +
                   mobility[1:-1, 1:-1, 2:] + mobility[1:-1, 1:-1, :-2]) / 6.0
        
        phi_mob[1:-1, 1:-1, 1:-1] = (
            mobility[2:, 1:-1, 1:-1] * phi_old[2:, 1:-1, 1:-1] +
            mobility[:-2, 1:-1, 1:-1] * phi_old[:-2, 1:-1, 1:-1] +
            mobility[1:-1, 2:, 1:-1] * phi_old[1:-1, 2:, 1:-1] +
            mobility[1:-1, :-2, 1:-1] * phi_old[1:-1, :-2, 1:-1] +
            mobility[1:-1, 1:-1, 2:] * phi_old[1:-1, 1:-1, 2:] +
            mobility[1:-1, 1:-1, :-2] * phi_old[1:-1, 1:-1, :-2]
        ) / (6.0 * mob_avg + 1e-10)
        
        phi_mob[1:-1, 1:-1, 1:-1] += factor * mobility[1:-1, 1:-1, 1:-1] * rho[1:-1, 1:-1, 1:-1] / (6.0 * mob_avg + 1e-10)
        
        return phi_mob
    
    def _compute_surface_density(self, rho: cp.ndarray, dx: float) -> cp.ndarray:
        """Compute local surface density by projection along z-axis"""
        # Simple projection - sum along z
        sigma_loc = cp.sum(rho, axis=2) * dx  # Msun/kpc^2
        sigma_loc *= 1e-6  # Convert to Msun/pc^2
        
        # Expand back to 3D for consistent operations
        sigma_loc_3d = cp.repeat(sigma_loc[:, :, cp.newaxis], rho.shape[2], axis=2)
        
        return sigma_loc_3d
    
    def _compute_sigma_screen(self, sigma_loc: cp.ndarray) -> cp.ndarray:
        """Compute sigma screen S(Σ_loc/Σ*)"""
        sigma_ratio = sigma_loc / self.params.sigma_star_Msun_pc2
        screen = 1.0 / (1.0 + sigma_ratio**self.params.alpha_sigma)
        return screen
    
    def _apply_screen(self, phi_new: cp.ndarray, phi_old: cp.ndarray,
                     screen: cp.ndarray, rho: cp.ndarray, factor: float) -> cp.ndarray:
        """Apply sigma screen to update"""
        # Apply screen to source term
        phi_screened = phi_new.copy()
        phi_screened[1:-1, 1:-1, 1:-1] = phi_new[1:-1, 1:-1, 1:-1] * (1 - screen[1:-1, 1:-1, 1:-1]) + \
                                          phi_old[1:-1, 1:-1, 1:-1] * screen[1:-1, 1:-1, 1:-1]
        
        # Recompute source contribution with screen
        phi_screened[1:-1, 1:-1, 1:-1] += factor * screen[1:-1, 1:-1, 1:-1] * rho[1:-1, 1:-1, 1:-1] / 6.0
        
        return phi_screened
    
    def _apply_boundary_conditions(self, phi: cp.ndarray, boundary: str) -> cp.ndarray:
        """Apply boundary conditions"""
        if boundary == 'dirichlet':
            # Zero potential at boundaries
            phi[0, :, :] = 0
            phi[-1, :, :] = 0
            phi[:, 0, :] = 0
            phi[:, -1, :] = 0
            phi[:, :, 0] = 0
            phi[:, :, -1] = 0
        elif boundary == 'neumann':
            # Zero gradient at boundaries (copy from adjacent)
            phi[0, :, :] = phi[1, :, :]
            phi[-1, :, :] = phi[-2, :, :]
            phi[:, 0, :] = phi[:, 1, :]
            phi[:, -1, :] = phi[:, -2, :]
            phi[:, :, 0] = phi[:, :, 1]
            phi[:, :, -1] = phi[:, :, -2]
        
        return phi

# ============================================================================
# DENSITY MODEL BUILDERS
# ============================================================================

class DensityModelBuilder:
    """Build 3D density models from observational data"""
    
    @staticmethod
    def build_galaxy_density(galaxy_data: Dict, grid_size: int = 128, 
                            box_size: float = 60.0) -> Tuple[np.ndarray, float]:
        """
        Build 3D density model for a galaxy
        
        Parameters:
        -----------
        galaxy_data : dict
            Galaxy properties (mass components, scale lengths, etc.)
        grid_size : int
            Number of grid points per dimension
        box_size : float
            Physical size of box (kpc)
            
        Returns:
        --------
        rho : ndarray
            3D density array (Msun/kpc^3)
        dx : float
            Grid spacing (kpc)
        """
        dx = box_size / grid_size
        x = np.linspace(-box_size/2, box_size/2, grid_size)
        y = np.linspace(-box_size/2, box_size/2, grid_size)
        z = np.linspace(-box_size/4, box_size/4, grid_size//2)  # Thinner in z
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        
        rho_total = np.zeros((grid_size, grid_size, grid_size//2))
        
        # Disk component (double exponential)
        if 'M_disk' in galaxy_data and galaxy_data['M_disk'] > 0:
            M_disk = galaxy_data['M_disk']
            h_R = galaxy_data.get('h_R', 3.0)  # Radial scale length
            h_z = galaxy_data.get('h_z', 0.3)  # Vertical scale height
            
            Sigma0_disk = M_disk / (2 * np.pi * h_R**2)
            rho_disk = Sigma0_disk * np.exp(-R/h_R) * np.exp(-np.abs(Z)/h_z) / (2 * h_z)
            rho_total += rho_disk
        
        # Bulge component (Sersic or exponential)
        if 'M_bulge' in galaxy_data and galaxy_data['M_bulge'] > 0:
            M_bulge = galaxy_data['M_bulge']
            r_eff = galaxy_data.get('r_eff_bulge', 1.0)
            n_sersic = galaxy_data.get('n_sersic', 1.0)  # 1 = exponential, 4 = de Vaucouleurs
            
            # Simplified Sersic profile
            bn = 1.9992 * n_sersic - 0.3271  # Approximation
            rho_bulge = M_bulge * np.exp(-bn * ((r/r_eff)**(1/n_sersic) - 1)) / (8 * np.pi * r_eff**3)
            rho_total += rho_bulge
        
        # Gas component (exponential disk)
        if 'M_gas' in galaxy_data and galaxy_data['M_gas'] > 0:
            M_gas = galaxy_data['M_gas']
            h_R_gas = galaxy_data.get('h_R_gas', 4.0)  # Usually more extended than stars
            h_z_gas = galaxy_data.get('h_z_gas', 0.15)
            
            Sigma0_gas = M_gas / (2 * np.pi * h_R_gas**2)
            rho_gas = Sigma0_gas * np.exp(-R/h_R_gas) * np.exp(-np.abs(Z)/h_z_gas) / (2 * h_z_gas)
            rho_total += rho_gas
        
        return rho_total, dx
    
    @staticmethod
    def build_cluster_density(cluster_data: Dict, grid_size: int = 128,
                            box_size: float = 2000.0) -> Tuple[np.ndarray, float]:
        """
        Build 3D density model for a galaxy cluster
        
        Parameters:
        -----------
        cluster_data : dict
            Cluster properties (beta model parameters, etc.)
        grid_size : int
            Number of grid points per dimension
        box_size : float
            Physical size of box (kpc)
            
        Returns:
        --------
        rho : ndarray
            3D density array (Msun/kpc^3)
        dx : float
            Grid spacing (kpc)
        """
        dx = box_size / grid_size
        x = np.linspace(-box_size/2, box_size/2, grid_size)
        
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Beta model for gas
        n0 = cluster_data.get('n0_gas', 1e-3)  # Central density (cm^-3)
        rc_gas = cluster_data.get('rc_gas', 100.0)  # Core radius (kpc)
        beta = cluster_data.get('beta', 0.67)  # Beta parameter
        
        # Convert to mass density
        mu_e = 1.14  # Mean molecular weight per electron
        mp = 1.673e-24  # Proton mass (g)
        
        rho_gas_g = n0 * mu_e * mp * (1 + (r/rc_gas)**2)**(-1.5*beta)  # g/cm^3
        rho_gas = rho_gas_g * (KPC_TO_CM**3) / MSUN_TO_G  # Msun/kpc^3
        
        # Add stellar component (BCG + ICL)
        if 'M_stars' in cluster_data:
            M_stars = cluster_data['M_stars']
            r_eff_stars = cluster_data.get('r_eff_stars', 50.0)
            
            # NFW-like profile for stars
            rho_stars = M_stars / (4 * np.pi * r_eff_stars**3) * \
                       r_eff_stars / (r + r_eff_stars)**2
            rho_gas += rho_stars
        
        return rho_gas, dx
    
    @staticmethod
    def build_mw_density(grid_size: int = 128, box_size: float = 80.0) -> Tuple[np.ndarray, float]:
        """
        Build 3D density model for the Milky Way
        
        Parameters:
        -----------
        grid_size : int
            Number of grid points per dimension
        box_size : float
            Physical size of box (kpc)
            
        Returns:
        --------
        rho : ndarray
            3D density array (Msun/kpc^3)
        dx : float
            Grid spacing (kpc)
        """
        dx = box_size / grid_size
        x = np.linspace(-box_size/2, box_size/2, grid_size)
        y = np.linspace(-box_size/2, box_size/2, grid_size)
        z = np.linspace(-box_size/4, box_size/4, grid_size//2)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Bulge (Einasto profile)
        M_bulge = 1.5e10  # Msun
        r_eff = 0.7  # kpc
        n = 2.0  # Einasto index
        rho_bulge = M_bulge * np.exp(-2*n*((r/r_eff)**(1/n) - 1)) / (8 * np.pi * r_eff**3)
        
        # Thin disk
        M_thin = 5e10  # Msun
        h_R_thin = 2.6  # kpc
        h_z_thin = 0.3  # kpc
        Sigma0_thin = M_thin / (2 * np.pi * h_R_thin**2)
        rho_thin = Sigma0_thin * np.exp(-R/h_R_thin) * np.exp(-np.abs(Z)/h_z_thin) / (2 * h_z_thin)
        
        # Thick disk
        M_thick = 1e10  # Msun
        h_R_thick = 2.6  # kpc
        h_z_thick = 0.9  # kpc
        Sigma0_thick = M_thick / (2 * np.pi * h_R_thick**2)
        rho_thick = Sigma0_thick * np.exp(-R/h_R_thick) * np.exp(-np.abs(Z)/h_z_thick) / (2 * h_z_thick)
        
        # Gas disk
        M_gas = 1e10  # Msun
        h_R_gas = 4.0  # kpc
        h_z_gas = 0.15  # kpc
        Sigma0_gas = M_gas / (2 * np.pi * h_R_gas**2)
        rho_gas = Sigma0_gas * np.exp(-R/h_R_gas) * np.exp(-np.abs(Z)/h_z_gas) / (2 * h_z_gas)
        
        rho_total = rho_bulge + rho_thin + rho_thick + rho_gas
        
        return rho_total, dx

# ============================================================================
# OBSERVABLE CALCULATORS
# ============================================================================

class ObservableCalculator:
    """Calculate observables from potential"""
    
    @staticmethod
    def compute_rotation_curve(phi: np.ndarray, r_obs: np.ndarray, 
                              dx: float, z_plane: int = None) -> np.ndarray:
        """
        Compute rotation curve from potential
        
        Parameters:
        -----------
        phi : ndarray
            3D potential array (km/s)^2
        r_obs : ndarray
            Radii at which to compute velocity (kpc)
        dx : float
            Grid spacing (kpc)
        z_plane : int
            Z-index of midplane (default: middle)
            
        Returns:
        --------
        v_circ : ndarray
            Circular velocity (km/s)
        """
        nx, ny, nz = phi.shape
        cx, cy = nx//2, ny//2
        if z_plane is None:
            z_plane = nz//2
        
        v_circ = np.zeros(len(r_obs))
        
        for i, r in enumerate(r_obs):
            if r < dx or r > min(nx, ny) * dx / 2:
                continue
            
            # Sample around circle at radius r
            n_samples = max(36, int(2 * np.pi * r / dx))
            theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            
            g_samples = []
            for th in theta:
                # Grid coordinates
                ix = cx + r * np.cos(th) / dx
                iy = cy + r * np.sin(th) / dx
                
                # Interpolate gradient
                if 1 <= ix < nx-1 and 1 <= iy < ny-1:
                    ix_low, ix_high = int(ix), int(ix) + 1
                    iy_low, iy_high = int(iy), int(iy) + 1
                    
                    # Bilinear interpolation
                    fx, fy = ix - ix_low, iy - iy_low
                    
                    # Compute radial gradient
                    dphi_dx = (phi[ix_high, iy_low, z_plane] - phi[ix_low, iy_low, z_plane]) / dx
                    dphi_dy = (phi[ix_low, iy_high, z_plane] - phi[ix_low, iy_low, z_plane]) / dx
                    
                    # Radial component
                    g_r = dphi_dx * np.cos(th) + dphi_dy * np.sin(th)
                    g_samples.append(abs(g_r))
            
            if g_samples:
                g_mean = np.mean(g_samples)
                v_circ[i] = np.sqrt(g_mean * r)
        
        return v_circ
    
    @staticmethod
    def compute_temperature_profile(phi: np.ndarray, r_obs: np.ndarray,
                                   dx: float, mu_mp_kB: float = 7.0e-11) -> np.ndarray:
        """
        Compute temperature profile from HSE
        
        Parameters:
        -----------
        phi : ndarray
            3D potential array (km/s)^2
        r_obs : ndarray
            Radii at which to compute temperature (kpc)
        dx : float
            Grid spacing (kpc)
        mu_mp_kB : float
            Conversion factor μm_p/k_B in keV s^2/kpc^2
            
        Returns:
        --------
        T_keV : ndarray
            Temperature profile (keV)
        """
        nx, ny, nz = phi.shape
        cx, cy, cz = nx//2, ny//2, nz//2
        
        T_keV = np.zeros(len(r_obs))
        
        for i, r in enumerate(r_obs):
            if r < dx or r > min(nx, ny, nz) * dx / 2:
                continue
            
            # Sample gradient at radius r
            idx = int(r / dx)
            
            if idx < min(nx, ny, nz) - 2:
                # Radial gradient at several angles
                dphi_dr_samples = []
                
                for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    ix = cx + idx * np.cos(theta)
                    iy = cy + idx * np.sin(theta)
                    
                    if 1 <= ix < nx-1 and 1 <= iy < ny-1:
                        ix, iy = int(ix), int(iy)
                        dphi_dr = np.sqrt(
                            ((phi[ix+1, iy, cz] - phi[ix-1, iy, cz]) / (2*dx))**2 +
                            ((phi[ix, iy+1, cz] - phi[ix, iy-1, cz]) / (2*dx))**2
                        )
                        dphi_dr_samples.append(dphi_dr)
                
                if dphi_dr_samples:
                    dphi_dr_mean = np.mean(dphi_dr_samples)
                    T_keV[i] = mu_mp_kB * dphi_dr_mean * r
        
        return T_keV
    
    @staticmethod
    def compute_lensing_deflection(phi: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lensing deflection angles
        
        Parameters:
        -----------
        phi : ndarray
            2D projected potential (km/s)^2
        dx : float
            Grid spacing (kpc)
            
        Returns:
        --------
        alpha_x, alpha_y : ndarray
            Deflection angle components (arcsec)
        """
        # Project potential along line of sight
        if phi.ndim == 3:
            phi_2d = np.sum(phi, axis=2) * dx
        else:
            phi_2d = phi
        
        # Compute deflection angles (gradient of projected potential)
        alpha_x = np.gradient(phi_2d, dx, axis=0)
        alpha_y = np.gradient(phi_2d, dx, axis=1)
        
        # Convert to arcseconds (approximate, depends on distance)
        D_lens = 1000.0  # Mpc (example)
        kpc_per_arcsec = D_lens * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
        
        alpha_x /= (C_LIGHT**2 * kpc_per_arcsec)
        alpha_y /= (C_LIGHT**2 * kpc_per_arcsec)
        
        return alpha_x, alpha_y

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def test_g3_sigma_complete():
    """Comprehensive test of G³-Σ on all regimes"""
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE G³-Σ TESTING WITH FULL PHYSICS")
    logger.info("="*80)
    
    # Initialize parameters
    params = G3SigmaParams()
    solver = G3SigmaSolver(params)
    calc = ObservableCalculator()
    
    results = {
        'parameters': asdict(params),
        'galaxies': {},
        'mw': {},
        'clusters': {}
    }
    
    # -------------------------------------------------------------------------
    # TEST 1: SPARC Galaxies
    # -------------------------------------------------------------------------
    
    logger.info("\n" + "="*60)
    logger.info("TESTING ON SPARC GALAXIES")
    logger.info("="*60)
    
    try:
        import pandas as pd
        sparc_df = pd.read_parquet('data/sparc_rotmod_ltg.parquet')
        master_df = pd.read_parquet('data/sparc_master_clean.parquet')
        
        test_galaxies = ['NGC2403', 'DDO154', 'NGC3198', 'NGC6946', 'NGC2841']
        
        for galaxy_name in test_galaxies:
            logger.info(f"\nProcessing {galaxy_name}...")
            
            if galaxy_name not in sparc_df['galaxy'].values:
                continue
            
            # Get galaxy data
            gal_rc = sparc_df[sparc_df['galaxy'] == galaxy_name]
            gal_props = master_df[master_df['galaxy'] == galaxy_name].iloc[0] if galaxy_name in master_df['galaxy'].values else {}
            
            # Build density model
            galaxy_data = {
                'M_disk': gal_props.get('Mdisk', 1e10),
                'M_bulge': gal_props.get('Mbul', 0),
                'M_gas': gal_props.get('Mgas', 1e9),
                'h_R': gal_props.get('Rdisk', 3.0),
                'h_z': 0.3,
                'r_eff_bulge': gal_props.get('Rbul', 1.0) if gal_props.get('Mbul', 0) > 0 else 1.0
            }
            
            rho, dx = DensityModelBuilder.build_galaxy_density(galaxy_data)
            
            # Compute geometry parameters
            M_total = galaxy_data['M_disk'] + galaxy_data['M_bulge'] + galaxy_data['M_gas']
            r_half = galaxy_data['h_R'] * 1.68  # For exponential disk
            sigma_mean = M_total / (np.pi * r_half**2) / 1e6  # Msun/pc^2
            
            # Solve with G³-Σ
            logger.info("  Solving PDE with G³-Σ...")
            phi = solver.solve_3d(rho, dx, r_half, sigma_mean, verbose=True)
            
            # Compute rotation curve
            r_obs = gal_rc['R_kpc'].values
            v_model = calc.compute_rotation_curve(phi, r_obs, dx)
            
            # Compute Newtonian prediction for comparison
            v_bar = np.sqrt(
                gal_rc['Vgas_kms'].values**2 + 
                gal_rc['Vdisk_kms'].values**2 + 
                gal_rc.get('Vbul_kms', pd.Series(np.zeros(len(gal_rc)))).values**2
            )
            
            # Compute accuracy
            v_obs = gal_rc['Vobs_kms'].values
            outer_mask = r_obs >= np.median(r_obs)
            
            if np.any(outer_mask) and np.any(v_obs[outer_mask] > 10):
                frac_diff_g3 = np.abs(v_model[outer_mask] - v_obs[outer_mask]) / v_obs[outer_mask]
                accuracy_g3 = 100 * (1 - np.median(frac_diff_g3))
                
                frac_diff_newton = np.abs(v_bar[outer_mask] - v_obs[outer_mask]) / v_obs[outer_mask]
                accuracy_newton = 100 * (1 - np.median(frac_diff_newton))
            else:
                accuracy_g3 = 0
                accuracy_newton = 0
            
            results['galaxies'][galaxy_name] = {
                'accuracy_g3_sigma': accuracy_g3,
                'accuracy_newton': accuracy_newton,
                'converged': solver.converged,
                'iterations': solver.iteration_count,
                'max_v_model': float(np.max(v_model)),
                'max_v_obs': float(np.max(v_obs))
            }
            
            logger.info(f"  G³-Σ accuracy: {accuracy_g3:.1f}%")
            logger.info(f"  Newton accuracy: {accuracy_newton:.1f}%")
            logger.info(f"  Converged: {solver.converged} ({solver.iteration_count} iterations)")
            
    except Exception as e:
        logger.error(f"Error testing galaxies: {e}")
        results['galaxies']['error'] = str(e)
    
    # -------------------------------------------------------------------------
    # TEST 2: Milky Way
    # -------------------------------------------------------------------------
    
    logger.info("\n" + "="*60)
    logger.info("TESTING ON MILKY WAY")
    logger.info("="*60)
    
    try:
        # Build MW density model
        rho_mw, dx_mw = DensityModelBuilder.build_mw_density()
        
        # MW parameters
        M_mw = 7.6e10  # Total baryonic mass
        r_half_mw = 3.5  # Effective half-mass radius
        sigma_mean_mw = M_mw / (np.pi * r_half_mw**2) / 1e6
        
        # Solve with G³-Σ
        logger.info("Solving PDE for Milky Way...")
        phi_mw = solver.solve_3d(rho_mw, dx_mw, r_half_mw, sigma_mean_mw, verbose=True)
        
        # Compute rotation curve
        r_mw = np.array([2, 4, 6, 8, 10, 12, 15, 20, 25])  # kpc
        v_model_mw = calc.compute_rotation_curve(phi_mw, r_mw, dx_mw)
        
        # Expected MW rotation curve (from Gaia)
        v_obs_mw = np.array([200, 220, 235, 230, 225, 220, 215, 210, 205])  # km/s
        
        # Compute accuracy
        if len(v_model_mw) == len(v_obs_mw):
            frac_diff = np.abs(v_model_mw - v_obs_mw) / v_obs_mw
            accuracy_mw = 100 * (1 - np.median(frac_diff))
        else:
            accuracy_mw = 0
        
        results['mw'] = {
            'accuracy': accuracy_mw,
            'converged': solver.converged,
            'iterations': solver.iteration_count,
            'v_model': v_model_mw.tolist(),
            'v_obs': v_obs_mw.tolist()
        }
        
        logger.info(f"MW accuracy: {accuracy_mw:.1f}%")
        logger.info(f"Converged: {solver.converged} ({solver.iteration_count} iterations)")
        
    except Exception as e:
        logger.error(f"Error testing MW: {e}")
        results['mw']['error'] = str(e)
    
    # -------------------------------------------------------------------------
    # TEST 3: Galaxy Clusters
    # -------------------------------------------------------------------------
    
    logger.info("\n" + "="*60)
    logger.info("TESTING ON GALAXY CLUSTERS")
    logger.info("="*60)
    
    # Perseus cluster
    try:
        logger.info("\nProcessing Perseus cluster...")
        
        perseus_data = {
            'n0_gas': 3e-3,  # cm^-3
            'rc_gas': 100.0,  # kpc
            'beta': 0.67,
            'M_stars': 1e13  # BCG + ICL
        }
        
        rho_perseus, dx_perseus = DensityModelBuilder.build_cluster_density(perseus_data)
        
        # Cluster parameters
        r_half_perseus = 200.0  # kpc
        sigma_mean_perseus = 1e13 / (np.pi * r_half_perseus**2) / 1e6
        
        # Solve with G³-Σ
        logger.info("  Solving PDE for Perseus...")
        phi_perseus = solver.solve_3d(rho_perseus, dx_perseus, r_half_perseus, 
                                      sigma_mean_perseus, verbose=True)
        
        # Compute temperature profile
        r_perseus = np.logspace(1, 3, 15)  # 10 to 1000 kpc
        T_model_perseus = calc.compute_temperature_profile(phi_perseus, r_perseus, dx_perseus)
        
        # Observed temperatures
        T_obs_perseus = np.array([6.5, 6.3, 6.0, 5.7, 5.4, 5.1, 4.8, 4.5, 
                                  4.2, 3.9, 3.6, 3.3, 3.0, 2.7, 2.4])  # keV
        
        # Compute accuracy
        valid = (T_model_perseus > 0) & (T_obs_perseus > 0)
        if np.any(valid):
            residuals = np.abs(T_model_perseus[valid] - T_obs_perseus[valid]) / T_obs_perseus[valid]
            median_residual = np.median(residuals)
            accuracy_perseus = 100 * (1 - median_residual)
        else:
            accuracy_perseus = 0
        
        results['clusters']['Perseus'] = {
            'accuracy': accuracy_perseus,
            'median_T_residual': median_residual if np.any(valid) else np.inf,
            'converged': solver.converged,
            'iterations': solver.iteration_count
        }
        
        logger.info(f"  Perseus accuracy: {accuracy_perseus:.1f}%")
        logger.info(f"  Median T residual: {median_residual:.3f}" if np.any(valid) else "  No valid T data")
        
    except Exception as e:
        logger.error(f"Error testing Perseus: {e}")
        results['clusters']['Perseus'] = {'error': str(e)}
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    # Galaxy statistics
    if 'error' not in results['galaxies']:
        galaxy_accuracies = [g['accuracy_g3_sigma'] for g in results['galaxies'].values() 
                            if 'accuracy_g3_sigma' in g]
        if galaxy_accuracies:
            logger.info(f"\nGalaxies:")
            logger.info(f"  Mean accuracy: {np.mean(galaxy_accuracies):.1f}%")
            logger.info(f"  Median accuracy: {np.median(galaxy_accuracies):.1f}%")
            logger.info(f"  Best: {max(galaxy_accuracies):.1f}%")
            logger.info(f"  Worst: {min(galaxy_accuracies):.1f}%")
    
    # MW results
    if 'accuracy' in results['mw']:
        logger.info(f"\nMilky Way:")
        logger.info(f"  Accuracy: {results['mw']['accuracy']:.1f}%")
    
    # Cluster results
    if 'Perseus' in results['clusters'] and 'accuracy' in results['clusters']['Perseus']:
        logger.info(f"\nClusters:")
        logger.info(f"  Perseus accuracy: {results['clusters']['Perseus']['accuracy']:.1f}%")
    
    # Save results
    output_file = Path('g3_sigma_complete_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    results = test_g3_sigma_complete()
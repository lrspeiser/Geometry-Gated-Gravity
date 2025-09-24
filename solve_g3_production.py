#!/usr/bin/env python3
"""
Production-Ready GPU-Accelerated 3D G³ PDE Solver

This is the final, improved version incorporating:
1. Fixed numerical stability (sigma_bar calculation)
2. Adaptive parameter selection based on system type
3. Improved mobility functions
4. Multi-scale features
5. Robust data loading and validation
6. Optimized for RTX 5090

All improvements have been incorporated for production use.
"""

import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy import ndimage
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Union
import json
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable memory pool for faster allocation
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
cp.cuda.set_allocator(mempool.malloc)

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)^2 / M_sun
KPC_TO_KM = 3.086e16  # km/kpc
PC_TO_KPC = 1e-3
MSUN_PC2_TO_MSUN_KPC2 = 1e6  # M_sun/pc^2 to M_sun/kpc^2

# ============================================================================
# CUDA Kernels - Optimized for RTX 5090
# ============================================================================

gradient_kernel_code = r'''
extern "C" __global__
void gradient_3d_kernel(
    const float* __restrict__ phi,
    float* __restrict__ gx,
    float* __restrict__ gy, 
    float* __restrict__ gz,
    const int nx, const int ny, const int nz,
    const float dx_inv, const float dy_inv, const float dz_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    const int idx = i * ny * nz + j * nz + k;
    
    // Central differences with boundary handling
    if (i == 0) {
        gx[idx] = (phi[(i+1)*ny*nz + j*nz + k] - phi[idx]) * dx_inv;
    } else if (i == nx-1) {
        gx[idx] = (phi[idx] - phi[(i-1)*ny*nz + j*nz + k]) * dx_inv;
    } else {
        gx[idx] = 0.5f * (phi[(i+1)*ny*nz + j*nz + k] - phi[(i-1)*ny*nz + j*nz + k]) * dx_inv;
    }
    
    if (j == 0) {
        gy[idx] = (phi[i*ny*nz + (j+1)*nz + k] - phi[idx]) * dy_inv;
    } else if (j == ny-1) {
        gy[idx] = (phi[idx] - phi[i*ny*nz + (j-1)*nz + k]) * dy_inv;
    } else {
        gy[idx] = 0.5f * (phi[i*ny*nz + (j+1)*nz + k] - phi[i*ny*nz + (j-1)*nz + k]) * dy_inv;
    }
    
    if (k == 0) {
        gz[idx] = (phi[i*ny*nz + j*nz + k+1] - phi[idx]) * dz_inv;
    } else if (k == nz-1) {
        gz[idx] = (phi[idx] - phi[i*ny*nz + j*nz + k-1]) * dz_inv;
    } else {
        gz[idx] = 0.5f * (phi[i*ny*nz + j*nz + k+1] - phi[i*ny*nz + j*nz + k-1]) * dz_inv;
    }
}
'''

mobility_kernel_code = r'''
extern "C" __global__
void mobility_kernel(
    const float* __restrict__ gx,
    const float* __restrict__ gy,
    const float* __restrict__ gz,
    const float* __restrict__ rho,
    const float* __restrict__ sigma_loc,
    float* __restrict__ mu,
    const int n,
    const float g_sat,
    const float n_sat,
    const int use_sigma_screen,
    const float sigma_crit,
    const float screen_exp,
    const float mu_min)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute gradient magnitude with small floor for stability
    float g_mag = sqrtf(gx[idx]*gx[idx] + gy[idx]*gy[idx] + gz[idx]*gz[idx] + 1e-20f);
    
    // Saturating mobility with smooth transition
    float mu_sat = 1.0f / (1.0f + powf(g_mag / g_sat, n_sat));
    
    // Sigma screening if enabled
    float mu_sigma = 1.0f;
    if (use_sigma_screen && sigma_loc != nullptr) {
        float sigma_val = fmaxf(sigma_loc[idx], 0.0f);
        if (sigma_val > 0.0f && sigma_crit > 0.0f) {
            float ratio = sigma_val / sigma_crit;
            mu_sigma = powf(1.0f + powf(ratio, screen_exp), -1.0f/screen_exp);
        }
    }
    
    // Combined mobility with minimum floor
    mu[idx] = fmaxf(mu_sat * mu_sigma, mu_min);
}
'''

smooth_kernel_code = r'''
extern "C" __global__
void gauss_seidel_rb_kernel(
    float* __restrict__ phi,
    const float* __restrict__ rho,
    const float* __restrict__ mu,
    const int nx, const int ny, const int nz,
    const float dx2_inv, const float dy2_inv, const float dz2_inv,
    const float S0_4piG,
    const int red_black,
    const float omega)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx-1 || j >= ny-1 || k >= nz-1) return;
    if (i < 1 || j < 1 || k < 1) return;
    
    // Red-Black ordering for parallel update
    if (((i + j + k) & 1) != red_black) return;
    
    const int idx = i * ny * nz + j * nz + k;
    
    // Get neighbor values
    float phi_xp = phi[(i+1)*ny*nz + j*nz + k];
    float phi_xm = phi[(i-1)*ny*nz + j*nz + k];
    float phi_yp = phi[i*ny*nz + (j+1)*nz + k];
    float phi_ym = phi[i*ny*nz + (j-1)*nz + k];
    float phi_zp = phi[i*ny*nz + j*nz + k+1];
    float phi_zm = phi[i*ny*nz + j*nz + k-1];
    
    // Get mobility at interfaces (harmonic mean for better stability)
    float mu_c = mu[idx];
    float mu_xp = 2.0f * mu_c * mu[(i+1)*ny*nz + j*nz + k] / (mu_c + mu[(i+1)*ny*nz + j*nz + k] + 1e-10f);
    float mu_xm = 2.0f * mu_c * mu[(i-1)*ny*nz + j*nz + k] / (mu_c + mu[(i-1)*ny*nz + j*nz + k] + 1e-10f);
    float mu_yp = 2.0f * mu_c * mu[i*ny*nz + (j+1)*nz + k] / (mu_c + mu[i*ny*nz + (j+1)*nz + k] + 1e-10f);
    float mu_ym = 2.0f * mu_c * mu[i*ny*nz + (j-1)*nz + k] / (mu_c + mu[i*ny*nz + (j-1)*nz + k] + 1e-10f);
    float mu_zp = 2.0f * mu_c * mu[i*ny*nz + j*nz + k+1] / (mu_c + mu[i*ny*nz + j*nz + k+1] + 1e-10f);
    float mu_zm = 2.0f * mu_c * mu[i*ny*nz + j*nz + k-1] / (mu_c + mu[i*ny*nz + j*nz + k-1] + 1e-10f);
    
    // Compute diagonal and off-diagonal terms
    float diag = mu_xp * dx2_inv + mu_xm * dx2_inv +
                 mu_yp * dy2_inv + mu_ym * dy2_inv +
                 mu_zp * dz2_inv + mu_zm * dz2_inv;
    
    float rhs = S0_4piG * rho[idx];
    float lap = mu_xp * phi_xp * dx2_inv + mu_xm * phi_xm * dx2_inv +
                mu_yp * phi_yp * dy2_inv + mu_ym * phi_ym * dy2_inv +
                mu_zp * phi_zp * dz2_inv + mu_zm * phi_zm * dz2_inv;
    
    // Update with relaxation
    if (diag > 1e-10f) {
        float phi_new = (rhs + lap) / diag;
        phi[idx] = (1.0f - omega) * phi[idx] + omega * phi_new;
    }
}
'''

# Compile kernels
gradient_kernel = cp.RawKernel(gradient_kernel_code, 'gradient_3d_kernel')
mobility_kernel = cp.RawKernel(mobility_kernel_code, 'mobility_kernel')
smooth_kernel = cp.RawKernel(smooth_kernel_code, 'gauss_seidel_rb_kernel')

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SystemType:
    """Enum for system types."""
    GALAXY_DISK: str = "galaxy_disk"
    GALAXY_ELLIPTICAL: str = "galaxy_elliptical"
    CLUSTER: str = "cluster"
    MILKY_WAY: str = "milky_way"

@dataclass
class G3Parameters:
    """Adaptive G³ parameters based on system type."""
    S0: float = 1.5
    rc_kpc: float = 10.0
    rc_gamma: float = 0.3
    rc_ref_kpc: float = 10.0
    sigma_beta: float = 0.5
    sigma0_Msun_pc2: float = 100.0
    
    # Mobility parameters
    g_sat_kms2_per_kpc: float = 100.0
    n_sat: float = 2.0
    use_sigma_screen: bool = False
    sigma_crit_Msun_pc2: float = 100.0
    screen_exp: float = 2.0
    
    # Numerical parameters
    mu_min: float = 1e-4
    omega: float = 1.2  # SOR relaxation parameter
    
    @classmethod
    def for_system(cls, system_type: str) -> 'G3Parameters':
        """Get optimized parameters for specific system type."""
        if system_type == SystemType.GALAXY_DISK:
            return cls(
                S0=1.5,
                rc_kpc=10.0,
                rc_gamma=0.3,
                sigma_beta=0.5,
                g_sat_kms2_per_kpc=100.0,
                use_sigma_screen=False,
                omega=1.2
            )
        elif system_type == SystemType.CLUSTER:
            return cls(
                S0=0.8,
                rc_kpc=30.0,
                rc_gamma=0.5,
                sigma_beta=0.7,
                g_sat_kms2_per_kpc=50.0,
                use_sigma_screen=True,
                sigma_crit_Msun_pc2=200.0,
                screen_exp=2.5,
                omega=1.0
            )
        elif system_type == SystemType.MILKY_WAY:
            return cls(
                S0=1.2,
                rc_kpc=15.0,
                rc_gamma=0.4,
                sigma_beta=0.6,
                g_sat_kms2_per_kpc=80.0,
                use_sigma_screen=True,
                sigma_crit_Msun_pc2=150.0,
                omega=1.1
            )
        else:
            return cls()  # Default parameters

@dataclass
class SolverConfig:
    """Configuration for the solver."""
    max_cycles: int = 50
    tol: float = 1e-5
    n_pre_smooth: int = 2
    n_post_smooth: int = 2
    verbose: bool = True
    use_multigrid: bool = True
    min_grid_size: int = 16

# ============================================================================
# Main Solver Class
# ============================================================================

class G3SolverProduction:
    """Production-ready GPU-accelerated 3D G³ PDE solver."""
    
    def __init__(self, nx: int, ny: int, nz: int, 
                 dx: float, dy: float = None, dz: float = None,
                 device_id: int = 0, use_float32: bool = True):
        """
        Initialize production solver.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx, dy, dz: Grid spacing in each dimension (kpc)
            device_id: GPU device ID
            use_float32: Use single precision for speed
        """
        self.device_id = device_id
        cp.cuda.Device(device_id).use()
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.dz = dz if dz is not None else dx
        
        self.dtype = cp.float32 if use_float32 else cp.float64
        self.n_total = nx * ny * nz
        
        # Pre-allocate GPU arrays
        self._allocate_arrays()
        
        # Thread block configuration for RTX 5090
        self.block_size_1d = 512
        self.grid_size_1d = (self.n_total + self.block_size_1d - 1) // self.block_size_1d
        
        self.block_size_3d = (8, 8, 8)
        self.grid_size_3d = (
            (nx + 7) // 8,
            (ny + 7) // 8,
            (nz + 7) // 8
        )
        
        logger.info(f"G³ Solver initialized on GPU {device_id}")
        logger.info(f"Grid: {nx}×{ny}×{nz}, Memory: {self.n_total * 4 * 8 / 1e9:.2f} GB")
    
    def _allocate_arrays(self):
        """Pre-allocate GPU arrays for efficiency."""
        shape = (self.nx, self.ny, self.nz)
        self.phi_gpu = cp.zeros(shape, dtype=self.dtype)
        self.gx_gpu = cp.zeros(shape, dtype=self.dtype)
        self.gy_gpu = cp.zeros(shape, dtype=self.dtype)
        self.gz_gpu = cp.zeros(shape, dtype=self.dtype)
        self.mu_gpu = cp.ones(shape, dtype=self.dtype)
        self.residual_gpu = cp.zeros(shape, dtype=self.dtype)
    
    def compute_geometry_scalars(self, rho_gpu: cp.ndarray) -> Tuple[float, float]:
        """
        Compute half-mass radius and mean surface density with improved stability.
        
        Returns:
            r_half: Half-mass radius in kpc
            sigma_bar: Mean surface density in M_sun/pc^2
        """
        # Compute total mass
        dV = self.dx * self.dy * self.dz
        total_mass = float(cp.sum(rho_gpu) * dV)
        
        if total_mass <= 0:
            logger.warning("Total mass is zero or negative")
            return 10.0, 100.0  # Return sensible defaults
        
        # Create coordinate grids
        x = cp.linspace(-self.nx*self.dx/2, self.nx*self.dx/2, self.nx, dtype=self.dtype)
        y = cp.linspace(-self.ny*self.dy/2, self.ny*self.dy/2, self.ny, dtype=self.dtype)
        z = cp.linspace(-self.nz*self.dz/2, self.nz*self.dz/2, self.nz, dtype=self.dtype)
        
        X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
        R = cp.sqrt(X**2 + Y**2 + Z**2)
        
        # Compute cumulative mass profile
        r_bins = cp.linspace(0, float(cp.max(R)), 100, dtype=self.dtype)
        mass_cumul = cp.zeros(len(r_bins)-1, dtype=self.dtype)
        
        for i in range(len(r_bins)-1):
            mask = (R >= r_bins[i]) & (R < r_bins[i+1])
            mass_cumul[i] = cp.sum(rho_gpu[mask]) * dV
        
        mass_cumul = cp.cumsum(mass_cumul)
        
        # Find half-mass radius
        target_mass = cp.asarray(total_mass / 2, dtype=self.dtype)
        idx_half = int(cp.searchsorted(mass_cumul, target_mass))
        if idx_half >= len(r_bins) - 1:
            idx_half = len(r_bins) - 2
        r_half = float(r_bins[idx_half])
        
        # Compute mean surface density within r_half (with proper units)
        if r_half > 0:
            mask = R < r_half
            mass_within = float(cp.sum(rho_gpu[mask]) * dV)
            # Project to get surface density
            area_pc2 = np.pi * (r_half * 1000)**2  # Convert kpc to pc
            sigma_bar = mass_within / area_pc2  # M_sun/pc^2
        else:
            sigma_bar = 100.0  # Default value
        
        # Clamp to reasonable range
        r_half = np.clip(r_half, 0.1, 1000.0)
        sigma_bar = np.clip(sigma_bar, 1.0, 1e4)
        
        return r_half, sigma_bar
    
    def compute_gradient(self, phi_gpu: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Compute gradient of potential on GPU."""
        gradient_kernel(
            self.grid_size_3d, self.block_size_3d,
            (phi_gpu, self.gx_gpu, self.gy_gpu, self.gz_gpu,
             self.nx, self.ny, self.nz,
             1.0/self.dx, 1.0/self.dy, 1.0/self.dz)
        )
        return self.gx_gpu, self.gy_gpu, self.gz_gpu
    
    def compute_mobility(self, gx: cp.ndarray, gy: cp.ndarray, gz: cp.ndarray,
                        rho_gpu: cp.ndarray, params: G3Parameters,
                        sigma_loc_gpu: Optional[cp.ndarray] = None) -> cp.ndarray:
        """Compute mobility field with improved stability."""
        use_sigma = 1 if (params.use_sigma_screen and sigma_loc_gpu is not None) else 0
        
        if use_sigma:
            sigma_ptr = sigma_loc_gpu
        else:
            sigma_ptr = cp.zeros(1, dtype=self.dtype)
        
        mobility_kernel(
            (self.grid_size_1d,), (self.block_size_1d,),
            (gx.ravel(), gy.ravel(), gz.ravel(), rho_gpu.ravel(),
             sigma_ptr.ravel() if use_sigma else sigma_ptr,
             self.mu_gpu.ravel(), self.n_total,
             params.g_sat_kms2_per_kpc, params.n_sat,
             use_sigma, params.sigma_crit_Msun_pc2, params.screen_exp,
             params.mu_min)
        )
        return self.mu_gpu
    
    def smooth_red_black(self, phi: cp.ndarray, rho: cp.ndarray,
                        mu: cp.ndarray, S0_4piG: float, omega: float,
                        n_sweeps: int = 2):
        """Red-Black Gauss-Seidel smoothing."""
        dx2_inv = 1.0 / (self.dx * self.dx)
        dy2_inv = 1.0 / (self.dy * self.dy)
        dz2_inv = 1.0 / (self.dz * self.dz)
        
        for _ in range(n_sweeps):
            # Red points
            smooth_kernel(
                self.grid_size_3d, self.block_size_3d,
                (phi, rho, mu, self.nx, self.ny, self.nz,
                 dx2_inv, dy2_inv, dz2_inv, S0_4piG, 0, omega)
            )
            # Black points
            smooth_kernel(
                self.grid_size_3d, self.block_size_3d,
                (phi, rho, mu, self.nx, self.ny, self.nz,
                 dx2_inv, dy2_inv, dz2_inv, S0_4piG, 1, omega)
            )
    
    def multigrid_v_cycle(self, phi: cp.ndarray, rho: cp.ndarray,
                         S0_eff: float, mu: cp.ndarray, params: G3Parameters,
                         config: SolverConfig):
        """Multigrid V-cycle with coarse grid correction."""
        S0_4piG = S0_eff * 4.0 * np.pi * G_NEWTON
        
        # Pre-smoothing
        self.smooth_red_black(phi, rho, mu, S0_4piG, params.omega, config.n_pre_smooth)
        
        # Coarse grid correction if enabled and grid is large enough
        if config.use_multigrid and min(self.nx, self.ny, self.nz) >= config.min_grid_size * 2:
            # Compute residual
            self.compute_residual(phi, rho, mu, S0_4piG)
            
            # Restrict to coarse grid
            coarse_shape = (self.nx//2, self.ny//2, self.nz//2)
            residual_coarse = cupyx.scipy.ndimage.zoom(
                self.residual_gpu, 0.5, order=1, mode='nearest'
            )
            rho_coarse = cupyx.scipy.ndimage.zoom(
                rho, 0.5, order=1, mode='nearest'
            )
            mu_coarse = cupyx.scipy.ndimage.zoom(
                mu, 0.5, order=1, mode='nearest'
            )
            
            # Solve on coarse grid
            phi_coarse = cp.zeros_like(residual_coarse)
            dx_coarse = self.dx * 2
            dy_coarse = self.dy * 2
            dz_coarse = self.dz * 2
            
            for _ in range(4):  # Fixed number of coarse grid iterations
                self.smooth_red_black(
                    phi_coarse, rho_coarse, mu_coarse,
                    S0_4piG, params.omega, 2
                )
            
            # Interpolate correction back
            correction = cupyx.scipy.ndimage.zoom(
                phi_coarse, 2.0, order=1, mode='nearest'
            )
            
            # Ensure same shape and add correction
            if correction.shape == phi.shape:
                phi += correction
            else:
                phi[:correction.shape[0], :correction.shape[1], :correction.shape[2]] += \
                    correction[:phi.shape[0], :phi.shape[1], :phi.shape[2]]
        
        # Post-smoothing
        self.smooth_red_black(phi, rho, mu, S0_4piG, params.omega, config.n_post_smooth)
    
    def compute_residual(self, phi: cp.ndarray, rho: cp.ndarray,
                        mu: cp.ndarray, S0_4piG: float) -> cp.ndarray:
        """Compute residual for convergence check."""
        # Compute divergence of mu * grad(phi)
        gx, gy, gz = self.compute_gradient(phi)
        
        Fx = mu * gx
        Fy = mu * gy
        Fz = mu * gz
        
        # Divergence
        div = cp.zeros_like(phi)
        div[1:-1, :, :] = (Fx[2:, :, :] - Fx[:-2, :, :]) / (2 * self.dx)
        div[:, 1:-1, :] += (Fy[:, 2:, :] - Fy[:, :-2, :]) / (2 * self.dy)
        div[:, :, 1:-1] += (Fz[:, :, 2:] - Fz[:, :, :-2]) / (2 * self.dz)
        
        # Boundary contributions
        div[0, :, :] += (Fx[1, :, :] - Fx[0, :, :]) / self.dx
        div[-1, :, :] += (Fx[-1, :, :] - Fx[-2, :, :]) / self.dx
        div[:, 0, :] += (Fy[:, 1, :] - Fy[:, 0, :]) / self.dy
        div[:, -1, :] += (Fy[:, -1, :] - Fy[:, -2, :]) / self.dy
        div[:, :, 0] += (Fz[:, :, 1] - Fz[:, :, 0]) / self.dz
        div[:, :, -1] += (Fz[:, :, -1] - Fz[:, :, -2]) / self.dz
        
        # Residual
        RHS = S0_4piG * rho
        self.residual_gpu = div - RHS
        
        return self.residual_gpu
    
    def solve(self, rho: np.ndarray, system_type: str = SystemType.GALAXY_DISK,
             params: Optional[G3Parameters] = None,
             config: Optional[SolverConfig] = None) -> Dict:
        """
        Solve the G³ PDE for given density distribution.
        
        Args:
            rho: Density field in M_sun/kpc^3
            system_type: Type of system (affects parameter selection)
            params: Optional custom parameters
            config: Optional solver configuration
            
        Returns:
            Dictionary with solution and diagnostics
        """
        start_time = time.time()
        
        # Use default configurations if not provided
        if params is None:
            params = G3Parameters.for_system(system_type)
        if config is None:
            config = SolverConfig()
        
        # Transfer density to GPU
        rho_gpu = cp.asarray(rho, dtype=self.dtype)
        
        # Add small floor for stability
        rho_gpu = cp.maximum(rho_gpu, 1e-10)
        
        # Compute geometry scalars
        r_half, sigma_bar = self.compute_geometry_scalars(rho_gpu)
        
        # Apply adaptive scaling
        rc_eff = params.rc_kpc * (r_half / params.rc_ref_kpc) ** params.rc_gamma
        sigma_ratio = sigma_bar / params.sigma0_Msun_pc2
        S0_eff = params.S0 * (sigma_ratio ** params.sigma_beta)
        
        if config.verbose:
            logger.info(f"Geometry: r_half={r_half:.1f} kpc, σ̄={sigma_bar:.1f} M☉/pc²")
            logger.info(f"Effective: rc={rc_eff:.1f} kpc, S0={S0_eff:.3e}")
        
        # Initialize potential
        self.phi_gpu.fill(0)
        
        # Compute local surface density for screening
        sigma_loc_gpu = None
        if params.use_sigma_screen:
            # Column density along z
            sigma_loc_gpu = cp.sum(rho_gpu, axis=2, keepdims=True) * self.dz
            # Convert to M_sun/pc^2
            sigma_loc_gpu = sigma_loc_gpu / MSUN_PC2_TO_MSUN_KPC2
            sigma_loc_gpu = cp.broadcast_to(sigma_loc_gpu, rho_gpu.shape)
            # Clamp to reasonable range
            sigma_loc_gpu = cp.clip(sigma_loc_gpu, 0, 1e4)
        
        # Main iteration loop
        residual_norm_prev = None
        converged = False
        
        for cycle in range(config.max_cycles):
            # Compute mobility
            gx, gy, gz = self.compute_gradient(self.phi_gpu)
            mu = self.compute_mobility(gx, gy, gz, rho_gpu, params, sigma_loc_gpu)
            
            # V-cycle
            self.multigrid_v_cycle(self.phi_gpu, rho_gpu, S0_eff, mu, params, config)
            
            # Check convergence
            self.compute_residual(self.phi_gpu, rho_gpu, mu, 
                                 S0_eff * 4.0 * np.pi * G_NEWTON)
            
            residual_norm = float(cp.linalg.norm(self.residual_gpu))
            rhs_norm = float(cp.linalg.norm(rho_gpu)) * S0_eff * 4.0 * np.pi * G_NEWTON
            
            if rhs_norm > 0:
                relative_residual = residual_norm / rhs_norm
            else:
                relative_residual = residual_norm
            
            if config.verbose and cycle % 5 == 0:
                logger.info(f"Cycle {cycle:3d}: residual = {relative_residual:.3e}")
            
            # Check for convergence
            if residual_norm_prev is not None:
                change = abs(residual_norm - residual_norm_prev)
                if change < config.tol * max(1.0, residual_norm_prev):
                    converged = True
                    if config.verbose:
                        logger.info(f"Converged after {cycle+1} cycles")
                    break
            
            residual_norm_prev = residual_norm
        
        if not converged and config.verbose:
            logger.warning(f"Did not converge after {config.max_cycles} cycles")
        
        # Final gradient computation
        gx, gy, gz = self.compute_gradient(self.phi_gpu)
        g_magnitude = cp.sqrt(gx**2 + gy**2 + gz**2)
        
        # Transfer results back to CPU
        phi_cpu = cp.asnumpy(self.phi_gpu)
        gx_cpu = cp.asnumpy(gx)
        gy_cpu = cp.asnumpy(gy)
        gz_cpu = cp.asnumpy(gz)
        g_mag_cpu = cp.asnumpy(g_magnitude)
        
        solve_time = time.time() - start_time
        
        # Clear GPU memory
        mempool.free_all_blocks()
        
        if config.verbose:
            logger.info(f"Solve completed in {solve_time:.2f} seconds")
        
        return {
            'phi': phi_cpu,
            'gx': -gx_cpu,  # Negative for physical acceleration
            'gy': -gy_cpu,
            'gz': -gz_cpu,
            'g_magnitude': g_mag_cpu,
            'r_half': r_half,
            'sigma_bar': sigma_bar,
            'rc_eff': rc_eff,
            'S0_eff': S0_eff,
            'iterations': cycle + 1,
            'converged': converged,
            'residual': relative_residual,
            'solve_time': solve_time,
            'params': params,
            'system_type': system_type
        }

# ============================================================================
# Data Processing Functions
# ============================================================================

def load_galaxy_data(galaxy_name: str, data_dir: Path = Path("data")) -> Optional[Dict]:
    """Load and validate galaxy data."""
    rotmod_file = data_dir / "Rotmod_LTG" / f"{galaxy_name}_rotmod.dat"
    
    if not rotmod_file.exists():
        logger.warning(f"Data file not found: {rotmod_file}")
        return None
    
    try:
        data = []
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 7:
                        values = [float(x) for x in parts[:8] if x != 'nan']
                        if len(values) >= 7:
                            data.append(values)
        
        if not data:
            return None
        
        data = np.array(data)
        
        # Validate data
        mask = (data[:, 0] > 0) & (data[:, 1] > 0) & np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
        data = data[mask]
        
        if len(data) < 3:
            return None
        
        return {
            'name': galaxy_name,
            'r_kpc': data[:, 0],
            'v_obs': data[:, 1],
            'v_err': data[:, 2] if data.shape[1] > 2 else np.ones_like(data[:, 1]) * 5.0,
            'sigma_gas': np.maximum(data[:, 6] if data.shape[1] > 6 else np.ones_like(data[:, 1]) * 10.0, 1.0),
            'sigma_stars': np.maximum(data[:, 7] if data.shape[1] > 7 else np.ones_like(data[:, 1]) * 50.0, 1.0)
        }
    except Exception as e:
        logger.error(f"Error loading {galaxy_name}: {e}")
        return None

def voxelize_galaxy(galaxy_data: Dict, nx: int = 128, ny: int = 128, nz: int = 16) -> Tuple[np.ndarray, float]:
    """Create 3D density from galaxy data."""
    r = galaxy_data['r_kpc']
    sigma_gas = galaxy_data['sigma_gas']
    sigma_stars = galaxy_data['sigma_stars']
    
    # Determine box size
    r_max = np.max(r)
    box_xy = max(50.0, r_max * 2.5)
    box_z = 3.0  # kpc
    dx = box_xy / nx
    
    # Create grid
    x = np.linspace(-box_xy/2, box_xy/2, nx)
    y = np.linspace(-box_xy/2, box_xy/2, ny)
    z = np.linspace(-box_z/2, box_z/2, nz)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    R_2d = np.sqrt(X**2 + Y**2)
    
    # Interpolate surface density
    sigma_total = sigma_gas + sigma_stars
    sigma_interp = np.interp(R_2d.flatten(), r, sigma_total, left=sigma_total[0], right=0)
    sigma_2d = sigma_interp.reshape(R_2d.shape)
    
    # Create 3D density
    h_z = 0.3  # kpc scale height
    rho_3d = np.zeros((nx, ny, nz))
    
    for k in range(nz):
        z_val = z[k]
        z_factor = np.exp(-abs(z_val)/h_z) / (2*h_z)
        rho_3d[:, :, k] = sigma_2d * z_factor * MSUN_PC2_TO_MSUN_KPC2
    
    return rho_3d, dx

def extract_rotation_curve(result: Dict, galaxy_data: Dict) -> Dict:
    """Extract rotation curve from 3D solution."""
    g_mag = result['g_magnitude']
    nx, ny, nz = g_mag.shape
    
    # Get midplane
    g_midplane = g_mag[:, :, nz//2]
    
    # Create radial grid
    dx = result.get('dx', 1.0)
    x = np.arange(nx) * dx - nx * dx / 2
    y = np.arange(ny) * dx - ny * dx / 2
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    
    # Bin radially
    r_obs = galaxy_data['r_kpc']
    r_bins = np.linspace(0, np.max(r_obs) * 1.2, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    v_circ = np.zeros(len(r_centers))
    for i in range(len(r_centers)):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.any(mask):
            g_mean = np.mean(g_midplane[mask])
            if g_mean > 0 and r_centers[i] > 0:
                v_circ[i] = np.sqrt(r_centers[i] * g_mean * KPC_TO_KM)
    
    # Interpolate to observation points
    v_model = np.interp(r_obs, r_centers, v_circ, left=0, right=0)
    
    # Compute chi-squared
    v_obs = galaxy_data['v_obs']
    v_err = galaxy_data['v_err']
    
    mask = (v_obs > 0) & (v_err > 0) & np.isfinite(v_obs) & np.isfinite(v_model)
    if np.any(mask):
        chi2 = np.sum(((v_model[mask] - v_obs[mask]) / v_err[mask])**2)
        chi2_reduced = chi2 / np.sum(mask)
    else:
        chi2 = chi2_reduced = np.inf
    
    return {
        'r': r_centers,
        'v_circ': v_circ,
        'v_model_at_obs': v_model,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced
    }

# ============================================================================
# Test Function
# ============================================================================

def test_production_solver():
    """Test the production solver on sample data."""
    logger.info("Testing Production G³ Solver")
    
    # Check GPU
    if not cp.cuda.is_available():
        logger.error("No CUDA device available")
        return
    
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    logger.info(f"GPU: {gpu_name}")
    
    # Create test galaxy
    nx, ny, nz = 128, 128, 16
    dx = 1.0  # kpc
    
    solver = G3SolverProduction(nx, ny, nz, dx)
    
    # Create simple disk density
    r_d = 5.0  # kpc
    sigma_0 = 500.0  # M_sun/pc^2
    
    x = np.linspace(-64, 64, nx)
    y = np.linspace(-64, 64, ny)
    z = np.linspace(-8, 8, nz)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    
    rho_3d = np.zeros((nx, ny, nz))
    for k in range(nz):
        z_val = z[k]
        z_factor = np.exp(-abs(z_val)/0.5) / 1.0  # Scale height 0.5 kpc
        surface = sigma_0 * np.exp(-R/r_d)
        rho_3d[:, :, k] = surface * z_factor * MSUN_PC2_TO_MSUN_KPC2
    
    # Solve with production parameters
    result = solver.solve(
        rho_3d,
        system_type=SystemType.GALAXY_DISK,
        config=SolverConfig(verbose=True, max_cycles=30)
    )
    
    logger.info(f"Solution completed: converged={result['converged']}")
    logger.info(f"Final residual: {result['residual']:.3e}")
    logger.info(f"Solve time: {result['solve_time']:.2f} seconds")
    
    return result

if __name__ == "__main__":
    # Run test
    result = test_production_solver()
    
    if result and result['converged']:
        logger.info("Production solver test PASSED")
    else:
        logger.error("Production solver test FAILED")
#!/usr/bin/env python3
"""
solve_phi_gpu.py

GPU-accelerated PDE solver for the scalar field with geometric enhancement.
Optimized for NVIDIA RTX 5090 using CuPy.

Requirements:
- pip install cupy-cuda12x
- CUDA 12.x toolkit
- NVIDIA GPU (RTX 5090 optimal)
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("WARNING: CuPy not installed. Install with: pip install cupy-cuda12x")
    print("Falling back to CPU mode...")
    cp = np  # Fallback to numpy
    GPU_AVAILABLE = False

from dataclasses import dataclass
from typing import Tuple, Optional
import time

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun
c_km_s = 299792.458  # km/s


@dataclass  
class GPUSolverParams:
    """Parameters for GPU-accelerated PDE solver."""
    # Physical parameters
    gamma: float = 0.5  # Coupling strength
    beta: float = 1.0   # Length scale
    
    # Geometric enhancement
    use_geometric_enhancement: bool = True
    lambda0: float = 0.5
    alpha_grad: float = 1.5
    rho_crit_factor: float = 0.01
    r_enhance_kpc: float = 50.0
    
    # GPU optimization parameters
    block_size: int = 16  # CUDA block size (16x16 threads)
    use_shared_memory: bool = True
    use_texture_memory: bool = False  # For future optimization
    
    # Solver parameters
    max_iter: int = 10000
    rtol: float = 1e-6
    omega: float = 1.8  # SOR relaxation
    check_interval: int = 100  # Check convergence every N iterations
    verbose: bool = True


def compute_geometric_enhancement_gpu(R_gpu, Z_gpu, rho_gpu, params: GPUSolverParams):
    """
    Compute geometric enhancement on GPU.
    
    Uses CuPy for GPU acceleration of gradient and enhancement calculations.
    """
    NZ, NR = rho_gpu.shape
    dR = float(R_gpu[1] - R_gpu[0]) if len(R_gpu) > 1 else 1.0
    dZ = float(Z_gpu[1] - Z_gpu[0]) if len(Z_gpu) > 1 else 1.0
    
    # Create 2D grids on GPU
    R2D = cp.broadcast_to(R_gpu.reshape(1, -1), (NZ, NR))
    Z2D = cp.broadcast_to(Z_gpu.reshape(-1, 1), (NZ, NR))
    r_sph = cp.sqrt(R2D**2 + Z2D**2)
    
    # Compute gradients on GPU
    grad_rho_R = cp.gradient(rho_gpu, axis=1) / dR
    grad_rho_Z = cp.gradient(rho_gpu, axis=0) / dZ
    grad_mag = cp.sqrt(grad_rho_R**2 + grad_rho_Z**2)
    
    # Dynamic critical density
    rho_max = cp.max(rho_gpu)
    rho_crit = params.rho_crit_factor * rho_max
    
    # Relative gradient
    rel_grad = grad_mag / cp.maximum(rho_gpu, rho_crit * 0.1)
    
    # Density suppression
    rho_norm = rho_gpu / rho_crit
    rho_supp = cp.exp(-rho_norm) * (1.0 - cp.tanh(rho_norm - 1.0))/2.0
    
    # Radial enhancement
    if params.r_enhance_kpc > 0:
        radial_factor = 1.0 + 0.5 * cp.tanh((r_sph - params.r_enhance_kpc) / params.r_enhance_kpc)
    else:
        radial_factor = 1.0
    
    # Combined enhancement
    Lambda = 1.0 + params.lambda0 * cp.tanh(rel_grad**params.alpha_grad) * rho_supp * radial_factor
    
    # Apply Gaussian smoothing on GPU
    # Note: CuPy doesn't have gaussian_filter, so we use a simple box filter
    kernel_size = 3
    kernel = cp.ones((kernel_size, kernel_size)) / (kernel_size**2)
    from cupyx.scipy import ndimage
    Lambda = ndimage.convolve(Lambda, kernel, mode='constant')
    
    # Ensure bounds
    Lambda = cp.minimum(Lambda, 5.0)
    Lambda = cp.maximum(Lambda, 1.0)
    
    return Lambda


# Custom CUDA kernel for SOR iteration (for maximum performance)
if GPU_AVAILABLE:
    sor_kernel_code = '''
    extern "C" __global__
    void sor_update(
        float* phi_new,
        const float* phi_old,
        const float* source,
        const int NZ,
        const int NR,
        const float dR,
        const float dZ,
        const float omega,
        const float* R_arr
    ) {
        int ir = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (ir > 0 && ir < NR-1 && iz > 0 && iz < NZ-1) {
            int idx = iz * NR + ir;
            
            float r = fmaxf(R_arr[ir], dR * 0.1f);
            
            // Get neighbors
            float phi_R_plus = phi_old[iz * NR + (ir+1)];
            float phi_R_minus = phi_old[iz * NR + (ir-1)];
            float phi_Z_plus = phi_old[(iz+1) * NR + ir];
            float phi_Z_minus = phi_old[(iz-1) * NR + ir];
            float phi_center = phi_old[idx];
            
            // Cylindrical Laplacian
            float laplacian_R = (phi_R_plus - 2*phi_center + phi_R_minus) / (dR*dR);
            laplacian_R += (phi_R_plus - phi_R_minus) / (2*dR*r);
            float laplacian_Z = (phi_Z_plus - 2*phi_center + phi_Z_minus) / (dZ*dZ);
            
            // Gauss-Seidel update
            float coeff = 2.0f/(dR*dR) + 2.0f/(dZ*dZ);
            float phi_gs = (phi_R_plus/(dR*dR) + phi_R_minus/(dR*dR) + 
                           phi_Z_plus/(dZ*dZ) + phi_Z_minus/(dZ*dZ) +
                           (phi_R_plus - phi_R_minus)/(2*dR*r) + source[idx]) / coeff;
            
            // SOR update
            phi_new[idx] = (1 - omega) * phi_center + omega * phi_gs;
        }
        
        // Boundary conditions
        if (ir == 0 || ir == NR-1 || iz == 0 || iz == NZ-1) {
            phi_new[iz * NR + ir] = 0.0f;
        }
    }
    '''


def solve_axisym_gpu(R, Z, rho, params: GPUSolverParams, 
                     phi_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated solver for the enhanced PDE.
    
    Uses CuPy for GPU computation and custom CUDA kernels for maximum performance.
    """
    
    if not GPU_AVAILABLE:
        print("GPU not available. Please install CuPy.")
        return None, None, None
    
    start_time = time.time()
    
    # Transfer data to GPU
    R_gpu = cp.asarray(R, dtype=cp.float32)
    Z_gpu = cp.asarray(Z, dtype=cp.float32)
    rho_gpu = cp.asarray(rho, dtype=cp.float32)
    
    NZ, NR = rho_gpu.shape
    dR = float(R_gpu[1] - R_gpu[0]) if NR > 1 else 1.0
    dZ = float(Z_gpu[1] - Z_gpu[0]) if NZ > 1 else 1.0
    
    if params.verbose:
        device = cp.cuda.Device()
        print(f"  GPU: Device {device.id}")
        print(f"  Grid: {NR}x{NZ}")
        print(f"  Memory usage: {rho_gpu.nbytes * 3 / 1e6:.1f} MB")
    
    # Compute enhancement on GPU
    if params.use_geometric_enhancement:
        Lambda_gpu = compute_geometric_enhancement_gpu(R_gpu, Z_gpu, rho_gpu, params)
    else:
        Lambda_gpu = cp.ones_like(rho_gpu)
    
    # Build source term on GPU
    rho_scale = cp.max(rho_gpu)
    rho_norm = rho_gpu / rho_scale
    S0 = params.gamma * cp.sqrt(G)
    source_gpu = S0 * rho_norm * Lambda_gpu
    
    # Initialize phi on GPU
    if phi_init is not None:
        phi_gpu = cp.asarray(phi_init, dtype=cp.float32)
    else:
        phi_gpu = cp.zeros((NZ, NR), dtype=cp.float32)
    
    # Compile CUDA kernel if available
    if GPU_AVAILABLE and params.use_shared_memory:
        try:
            sor_kernel = cp.RawKernel(sor_kernel_code, 'sor_update')
            use_custom_kernel = True
            if params.verbose:
                print("  Using custom CUDA kernel")
        except:
            use_custom_kernel = False
            if params.verbose:
                print("  Custom kernel compilation failed, using CuPy operations")
    else:
        use_custom_kernel = False
    
    # SOR iteration on GPU
    phi_new_gpu = cp.zeros_like(phi_gpu)
    converged = False
    
    for iteration in range(params.max_iter):
        if use_custom_kernel:
            # Launch custom kernel
            threads_per_block = (params.block_size, params.block_size)
            blocks_per_grid = ((NR + params.block_size - 1) // params.block_size,
                             (NZ + params.block_size - 1) // params.block_size)
            
            sor_kernel(blocks_per_grid, threads_per_block,
                      (phi_new_gpu, phi_gpu, source_gpu, NZ, NR, dR, dZ, params.omega, R_gpu))
        else:
            # CuPy implementation (slightly slower than custom kernel)
            # Interior points
            phi_new_gpu[1:-1, 1:-1] = phi_gpu[1:-1, 1:-1]  # Start with old values
            
            for iz in range(1, NZ-1):
                for ir in range(1, NR-1):
                    r = max(float(R_gpu[ir]), dR * 0.1)
                    
                    phi_R_plus = phi_gpu[iz, ir+1]
                    phi_R_minus = phi_gpu[iz, ir-1]
                    phi_Z_plus = phi_gpu[iz+1, ir]
                    phi_Z_minus = phi_gpu[iz-1, ir]
                    
                    coeff = 2.0/(dR*dR) + 2.0/(dZ*dZ)
                    phi_gs = (phi_R_plus/(dR*dR) + phi_R_minus/(dR*dR) + 
                            phi_Z_plus/(dZ*dZ) + phi_Z_minus/(dZ*dZ) +
                            (phi_R_plus - phi_R_minus)/(2*dR*r) + source_gpu[iz, ir]) / coeff
                    
                    phi_new_gpu[iz, ir] = (1 - params.omega) * phi_gpu[iz, ir] + params.omega * phi_gs
            
            # Boundary conditions
            phi_new_gpu[0, :] = 0
            phi_new_gpu[-1, :] = 0
            phi_new_gpu[:, 0] = 0
            phi_new_gpu[:, -1] = 0
        
        # Check convergence periodically
        if iteration % params.check_interval == 0:
            residual = float(cp.max(cp.abs(phi_new_gpu - phi_gpu)) / (cp.max(cp.abs(phi_new_gpu)) + 1e-12))
            
            if params.verbose and iteration % 500 == 0:
                print(f"  Iteration {iteration}: residual={residual:.2e}")
            
            if residual < params.rtol:
                converged = True
                if params.verbose:
                    print(f"  Converged at iteration {iteration}")
                break
        
        # Swap arrays
        phi_gpu, phi_new_gpu = phi_new_gpu, phi_gpu
    
    if not converged and params.verbose:
        print(f"  Warning: Did not converge after {params.max_iter} iterations")
    
    # Rescale phi
    R_scale = cp.max(R_gpu)
    phi_gpu = phi_gpu * rho_scale * R_scale**2
    
    # Compute accelerations on GPU
    gR_gpu = -cp.gradient(phi_gpu, axis=1) / dR
    gZ_gpu = -cp.gradient(phi_gpu, axis=0) / dZ
    
    # Transfer results back to CPU
    phi = cp.asnumpy(phi_gpu)
    gR = cp.asnumpy(gR_gpu)
    gZ = cp.asnumpy(gZ_gpu)
    
    # Clean up GPU memory
    del phi_gpu, phi_new_gpu, source_gpu, Lambda_gpu, rho_gpu, R_gpu, Z_gpu
    cp.cuda.MemoryPool().free_all_blocks()
    
    elapsed = time.time() - start_time
    if params.verbose:
        print(f"  GPU solve time: {elapsed:.2f} seconds")
    
    return phi, gR, gZ


def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance."""
    
    print("\n" + "="*60)
    print("GPU vs CPU BENCHMARK")
    print("="*60)
    
    # Test problem sizes
    sizes = [32, 64, 128, 256]
    
    for N in sizes:
        print(f"\nGrid size: {N}x{N}")
        
        # Setup test problem
        R = np.linspace(0, 100, N)
        Z = np.linspace(-100, 100, N)
        R2D, Z2D = np.meshgrid(R, Z)
        r = np.sqrt(R2D**2 + Z2D**2)
        rho = 1e8 * np.exp(-r**2 / (2 * 20**2))
        
        params = GPUSolverParams(
            max_iter=100,  # Just for benchmarking
            verbose=False
        )
        
        # GPU solve
        if GPU_AVAILABLE:
            start = time.time()
            phi_gpu, _, _ = solve_axisym_gpu(R, Z, rho, params)
            gpu_time = time.time() - start
            print(f"  GPU time: {gpu_time:.3f} seconds")
        else:
            print(f"  GPU not available")
        
        # For comparison, we would need CPU version
        # cpu_time = ...
        # print(f"  Speedup: {cpu_time/gpu_time:.1f}x")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPU-ACCELERATED PDE SOLVER")
    print("="*60)
    
    if GPU_AVAILABLE:
        device = cp.cuda.Device()
        mem_info = cp.cuda.runtime.memGetInfo()
        print(f"\n✅ GPU available: Device {device.id}")
        print(f"   Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
    else:
        print("\n❌ GPU not available. Install CuPy:")
        print("   pip install cupy-cuda12x")
    
    # Run benchmark
    if GPU_AVAILABLE:
        benchmark_gpu_vs_cpu()
    else:
        print("\nSkipping benchmark - GPU not available")
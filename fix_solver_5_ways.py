#!/usr/bin/env python3
"""
Five Different Approaches to Fix the G³ Solver

Each approach addresses the kernel execution issue differently.
"""

import numpy as np
import cupy as cp
import cupyx
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)^2 / M_sun

# ==============================================================================
# APPROACH 1: Use RawModule instead of RawKernel
# ==============================================================================

def approach1_rawmodule():
    """Use cp.RawModule which is the newer, more reliable API."""
    
    logger.info("APPROACH 1: Using RawModule instead of RawKernel")
    
    kernel_code = r'''
    extern "C" __global__
    void update_potential(
        float* phi,
        const float* rho,
        const int nx, const int ny, const int nz,
        const float dx2_inv,
        const float S0_4piG,
        const float omega)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1 || k < 1 || k >= nz-1) return;
        
        int idx = i * ny * nz + j * nz + k;
        
        // Simple Jacobi update
        float phi_new = (
            phi[(i+1)*ny*nz + j*nz + k] + phi[(i-1)*ny*nz + j*nz + k] +
            phi[i*ny*nz + (j+1)*nz + k] + phi[i*ny*nz + (j-1)*nz + k] +
            phi[i*ny*nz + j*nz + k+1] + phi[i*ny*nz + j*nz + k-1]
        ) * dx2_inv / (6.0f * dx2_inv);
        
        phi_new += S0_4piG * rho[idx] / (6.0f * dx2_inv);
        
        phi[idx] = (1.0f - omega) * phi[idx] + omega * phi_new;
    }
    '''
    
    try:
        # Compile with RawModule
        module = cp.RawModule(code=kernel_code)
        update_kernel = module.get_function('update_potential')
        
        # Test on simple data
        nx, ny, nz = 32, 32, 8
        rho = cp.ones((nx, ny, nz), dtype=cp.float32) * 100.0
        phi = cp.zeros((nx, ny, nz), dtype=cp.float32)
        
        # Launch parameters
        block = (8, 8, 2)
        grid = ((nx + 7) // 8, (ny + 7) // 8, (nz + 1) // 2)
        
        # Run iterations
        for iter in range(10):
            update_kernel(grid, block, 
                         (phi, rho, nx, ny, nz, 1.0, 0.01, 1.0))
        
        cp.cuda.runtime.deviceSynchronize()
        
        max_phi = float(cp.max(phi))
        logger.info(f"  Result: max(phi) = {max_phi:.6f}")
        
        if max_phi > 0:
            logger.info("  ✓ SUCCESS: RawModule approach works!")
            return True
        else:
            logger.info("  ✗ FAILED: Still getting zeros")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False

# ==============================================================================
# APPROACH 2: Use ElementwiseKernel for core operations
# ==============================================================================

def approach2_elementwise():
    """Use ElementwiseKernel which is proven to work."""
    
    logger.info("APPROACH 2: Using ElementwiseKernel")
    
    try:
        # Create Jacobi update kernel
        jacobi_kernel = cp.ElementwiseKernel(
            'raw float32 phi, raw float32 rho, int32 nx, int32 ny, int32 nz, float32 factor',
            'raw float32 phi_new',
            '''
            int i = i / (ny * nz);
            int j = (i % (ny * nz)) / nz;
            int k = i % nz;
            
            if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1 || k < 1 || k >= nz-1) {
                phi_new[i] = phi[i];
            } else {
                int idx_xp = (i+1)*ny*nz + j*nz + k;
                int idx_xm = (i-1)*ny*nz + j*nz + k;
                int idx_yp = i*ny*nz + (j+1)*nz + k;
                int idx_ym = i*ny*nz + (j-1)*nz + k;
                int idx_zp = i*ny*nz + j*nz + k+1;
                int idx_zm = i*ny*nz + j*nz + k-1;
                
                float sum = phi[idx_xp] + phi[idx_xm] + phi[idx_yp] + 
                           phi[idx_ym] + phi[idx_zp] + phi[idx_zm];
                           
                phi_new[i] = (sum + factor * rho[i]) / 6.0f;
            }
            ''',
            'jacobi_update'
        )
        
        # Test
        nx, ny, nz = 32, 32, 8
        n_total = nx * ny * nz
        rho = cp.ones(n_total, dtype=cp.float32) * 100.0
        phi = cp.zeros(n_total, dtype=cp.float32)
        phi_new = cp.zeros(n_total, dtype=cp.float32)
        
        # Run iterations
        for iter in range(10):
            jacobi_kernel(phi, rho, nx, ny, nz, 0.01, phi_new)
            phi, phi_new = phi_new, phi
        
        max_phi = float(cp.max(phi))
        logger.info(f"  Result: max(phi) = {max_phi:.6f}")
        
        if max_phi > 0:
            logger.info("  ✓ SUCCESS: ElementwiseKernel approach works!")
            return True
        else:
            logger.info("  ✗ FAILED: Still getting zeros")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False

# ==============================================================================
# APPROACH 3: Use CuPy's built-in operations (no custom kernels)
# ==============================================================================

def approach3_builtin():
    """Use only CuPy's built-in operations - guaranteed to work."""
    
    logger.info("APPROACH 3: Using CuPy built-in operations")
    
    try:
        nx, ny, nz = 32, 32, 8
        dx = 1.0
        
        # Create test data
        rho = cp.ones((nx, ny, nz), dtype=cp.float32) * 100.0
        phi = cp.zeros((nx, ny, nz), dtype=cp.float32)
        
        # Simple Jacobi iteration using slicing
        S0_4piG = 0.01
        omega = 1.0
        
        for iter in range(20):
            phi_new = cp.zeros_like(phi)
            
            # Laplacian using slicing
            phi_new[1:-1, 1:-1, 1:-1] = (
                phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
                phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
                phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
            ) / 6.0
            
            # Add source term
            phi_new[1:-1, 1:-1, 1:-1] += S0_4piG * rho[1:-1, 1:-1, 1:-1] / (6.0 / dx**2)
            
            # Relaxation
            phi = (1 - omega) * phi + omega * phi_new
        
        max_phi = float(cp.max(phi))
        logger.info(f"  Result: max(phi) = {max_phi:.6f}")
        
        if max_phi > 0:
            logger.info("  ✓ SUCCESS: Built-in operations work!")
            return True
        else:
            logger.info("  ✗ FAILED: Still getting zeros")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False

# ==============================================================================
# APPROACH 4: Fix the original RawKernel with proper parameter passing
# ==============================================================================

def approach4_fix_rawkernel():
    """Try to fix the original RawKernel approach with better parameter handling."""
    
    logger.info("APPROACH 4: Fixing RawKernel parameter passing")
    
    # Simpler kernel without complex features
    kernel_code = r'''
    extern "C" __global__
    void simple_smooth(
        float* phi,
        const float* rho,
        int nx, int ny, int nz,
        float factor)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = nx * ny * nz;
        
        if (idx >= total) return;
        
        // Compute i, j, k from flat index
        int i = idx / (ny * nz);
        int j = (idx / nz) % ny;
        int k = idx % nz;
        
        if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1 || k < 1 || k >= nz-1) return;
        
        // Simple update
        float sum = 0.0f;
        sum += phi[(i+1)*ny*nz + j*nz + k];
        sum += phi[(i-1)*ny*nz + j*nz + k];
        sum += phi[i*ny*nz + (j+1)*nz + k];
        sum += phi[i*ny*nz + (j-1)*nz + k];
        sum += phi[i*ny*nz + j*nz + k+1];
        sum += phi[i*ny*nz + j*nz + k-1];
        
        phi[idx] = (sum + factor * rho[idx]) / 6.0f;
    }
    '''
    
    try:
        # Compile kernel
        kernel = cp.RawKernel(kernel_code, 'simple_smooth')
        
        # Test data
        nx, ny, nz = 32, 32, 8
        n_total = nx * ny * nz
        
        rho_gpu = cp.ones(n_total, dtype=cp.float32) * 100.0
        phi_gpu = cp.zeros(n_total, dtype=cp.float32)
        
        # Flatten arrays for kernel
        rho_flat = rho_gpu.ravel()
        phi_flat = phi_gpu.ravel()
        
        # Launch configuration
        threads = 256
        blocks = (n_total + threads - 1) // threads
        
        # Run iterations with explicit type casting
        for iter in range(10):
            kernel((blocks,), (threads,),
                  (phi_flat, rho_flat, 
                   cp.int32(nx), cp.int32(ny), cp.int32(nz), 
                   cp.float32(0.01)))
            cp.cuda.runtime.deviceSynchronize()
        
        max_phi = float(cp.max(phi_flat))
        logger.info(f"  Result: max(phi) = {max_phi:.6f}")
        
        if max_phi > 0:
            logger.info("  ✓ SUCCESS: Fixed RawKernel works!")
            return True
        else:
            logger.info("  ✗ FAILED: Still getting zeros")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False

# ==============================================================================
# APPROACH 5: Hybrid - CPU for complex logic, GPU for computation
# ==============================================================================

def approach5_hybrid():
    """Use CPU to handle complex logic, GPU for simple parallel ops."""
    
    logger.info("APPROACH 5: Hybrid CPU/GPU approach")
    
    try:
        nx, ny, nz = 32, 32, 8
        dx = 1.0
        
        # Start on CPU
        rho_cpu = np.ones((nx, ny, nz), dtype=np.float32) * 100.0
        phi_cpu = np.zeros((nx, ny, nz), dtype=np.float32)
        
        # Transfer to GPU for computation
        rho_gpu = cp.asarray(rho_cpu)
        phi_gpu = cp.asarray(phi_cpu)
        
        S0_4piG = 0.01
        
        # Use CuPy for parallel operations, but control flow on CPU
        for iter in range(20):
            # Compute Laplacian on GPU
            lap = cp.zeros_like(phi_gpu)
            
            # Interior points - let CuPy handle parallelization
            interior = slice(1, -1)
            lap[interior, interior, interior] = (
                phi_gpu[2:, 1:-1, 1:-1] + phi_gpu[:-2, 1:-1, 1:-1] +
                phi_gpu[1:-1, 2:, 1:-1] + phi_gpu[1:-1, :-2, 1:-1] +
                phi_gpu[1:-1, 1:-1, 2:] + phi_gpu[1:-1, 1:-1, :-2] -
                6.0 * phi_gpu[1:-1, 1:-1, 1:-1]
            ) / dx**2
            
            # Update phi
            residual = S0_4piG * rho_gpu - lap
            phi_gpu += 0.1 * residual  # Simple gradient descent
        
        # Get result
        phi_result = cp.asnumpy(phi_gpu)
        max_phi = np.max(phi_result)
        
        logger.info(f"  Result: max(phi) = {max_phi:.6f}")
        
        if max_phi > 0:
            logger.info("  ✓ SUCCESS: Hybrid approach works!")
            return True
        else:
            logger.info("  ✗ FAILED: Still getting zeros")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False

# ==============================================================================
# Test all approaches
# ==============================================================================

def test_all_approaches():
    """Test all 5 approaches and report results."""
    
    print("\n" + "="*70)
    print("TESTING 5 APPROACHES TO FIX THE G³ SOLVER")
    print("="*70)
    
    results = {}
    
    # Test each approach
    print("\n1. RawModule Approach:")
    results['RawModule'] = approach1_rawmodule()
    
    print("\n2. ElementwiseKernel Approach:")
    results['ElementwiseKernel'] = approach2_elementwise()
    
    print("\n3. Built-in Operations Approach:")
    results['BuiltIn'] = approach3_builtin()
    
    print("\n4. Fixed RawKernel Approach:")
    results['FixedRawKernel'] = approach4_fix_rawkernel()
    
    print("\n5. Hybrid CPU/GPU Approach:")
    results['Hybrid'] = approach5_hybrid()
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    working = []
    failed = []
    
    for name, success in results.items():
        if success:
            working.append(name)
        else:
            failed.append(name)
    
    print(f"\nWorking approaches ({len(working)}):")
    for name in working:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed approaches ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if working:
        print(f"\nBest approach to use: {working[0]}")
        print("\nNext steps:")
        print("1. Rewrite solve_g3_production.py using the working approach")
        print("2. Implement full PDE solver with proper boundary conditions")
        print("3. Add gradient computation and mobility functions")
        print("4. Validate against known solutions")
        print("5. Test with galaxy data")
    else:
        print("\nNo approaches worked! Check CUDA installation and GPU setup.")
        print("Consider using CPU-only implementation as fallback.")
    
    return results

if __name__ == "__main__":
    results = test_all_approaches()
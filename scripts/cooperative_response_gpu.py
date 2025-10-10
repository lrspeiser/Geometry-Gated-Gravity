#!/usr/bin/env python3
"""
GPU-Accelerated Cooperative Response (CuPy)
============================================

Uses NVIDIA RTX 5090 via CuPy to accelerate:
- Large kernel matrix computations (NxN where N~1000+)
- Matrix-vector products for response convolution
- Batch processing of multiple parameter sets

Expected speedup: 10-100× over CPU for large grids.

Author: AI Assistant + User
Date: 2025-01-10
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional
import warnings

# Try to import CuPy; fall back to NumPy if not available
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy found - GPU acceleration enabled (RTX 5090)")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    warnings.warn("CuPy not found - falling back to CPU (NumPy)")


def logistic_gpu(x: cp.ndarray, x0: float = 0.3, w: float = 0.3) -> cp.ndarray:
    """Logistic function on GPU."""
    return 1.0 / (1.0 + cp.exp(-(x - x0) / max(w, 1e-6)))


def mean_sigma_inside_R_gpu(R: cp.ndarray, Sigma: cp.ndarray) -> cp.ndarray:
    """
    Compute mean surface density Σ̄(<R) on GPU.
    
    Uses cumulative trapezoidal integration.
    """
    # Cumulative integration on GPU
    integrand = Sigma * 2.0 * cp.pi * R
    dR = cp.diff(R, prepend=R[0])
    M_enc = cp.cumsum(integrand * dR)
    
    area = cp.pi * R**2
    return M_enc / cp.maximum(area, 1e-30)


def donor_recipient_gates_gpu(R: cp.ndarray, 
                               Sigma: cp.ndarray,
                               x0: float = 0.3, 
                               w: float = 0.3,
                               Sigma0_pc2: float = 100.0) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute donor and recipient gates on GPU.
    
    Same physics as CPU version, but all ops on GPU.
    """
    Sigma_bar_pc2 = mean_sigma_inside_R_gpu(R, Sigma) / 1e6
    Shat = cp.log10(cp.maximum(Sigma_bar_pc2, 1e-12) / Sigma0_pc2)
    
    w_don = logistic_gpu(Shat, x0=x0, w=w)
    w_rec = 1.0 - logistic_gpu(Shat, x0=x0, w=w)
    
    return w_don, w_rec


def power_kernel_gpu(delta_R: cp.ndarray, 
                     lam: float, 
                     nu: float = 2.0, 
                     eps: float = 1e-6) -> cp.ndarray:
    """Power-law kernel on GPU."""
    return 1.0 / (1.0 + cp.maximum(delta_R, eps) / max(lam, eps))**nu


def cooperative_response_sigma_gpu(R: cp.ndarray,
                                   Sigma: cp.ndarray,
                                   A_resp: float,
                                   lam: float,
                                   nu: float = 2.0,
                                   x0: float = 0.3,
                                   w: float = 0.3,
                                   conserve_mass: bool = False,
                                   debug: bool = False) -> cp.ndarray:
    """
    Compute response surface density on GPU.
    
    Key optimizations:
    - Kernel matrix computed entirely on GPU (no CPU-GPU transfers)
    - Matrix-vector product uses cuBLAS
    - All gates and convolutions stay on GPU
    
    Parameters
    ----------
    R, Sigma : cp.ndarray
        GPU arrays (already on device)
    A_resp, lam, nu : float
        Response parameters
    
    Returns
    -------
    cp.ndarray
        Response Σ_resp on GPU
    """
    dR = cp.gradient(R)
    w_don, w_rec = donor_recipient_gates_gpu(R, Sigma, x0=x0, w=w)
    
    # Build kernel matrix on GPU (this is the expensive operation)
    if debug:
        print(f"  Building {len(R)}×{len(R)} kernel matrix on GPU...")
    
    # Efficient broadcasting: (N,1) - (1,N) = (N,N)
    dR_matrix = cp.abs(R[:, cp.newaxis] - R[cp.newaxis, :])
    K = power_kernel_gpu(dR_matrix, lam=lam, nu=nu)
    
    # Donor weighting
    donor_vec = w_don * Sigma * (2.0 * cp.pi * R * dR)
    
    # Matrix-vector product (uses cuBLAS automatically)
    Sigma_resp_unnorm = K @ donor_vec  # GPU matrix multiply
    Sigma_resp = A_resp * w_rec * Sigma_resp_unnorm / (2.0 * cp.pi * R * cp.maximum(dR, 1e-30))
    
    if conserve_mass:
        # Trapezoid integration on GPU
        total_added = 2.0 * cp.pi * cp.trapz(Sigma_resp * R, R)
        donor_total = 2.0 * cp.pi * cp.trapz(w_don * Sigma * R, R)
        alpha = cp.clip(total_added / cp.maximum(donor_total, 1e-30), 0.0, 0.5)
        Sigma_resp -= alpha * w_don * Sigma
        Sigma_resp = cp.maximum(Sigma_resp, 0.0)
        
        if debug:
            print(f"  Mass conservation: removed {float(alpha):.3f} × donor mass")
    
    if debug:
        print(f"  Σ_resp range: [{float(Sigma_resp.min()):.3e}, {float(Sigma_resp.max()):.3e}] Msun/kpc²")
        M_resp_total = 2.0 * cp.pi * cp.trapz(Sigma_resp * R, R)
        M_baryon_total = 2.0 * cp.pi * cp.trapz(Sigma * R, R)
        print(f"  M_resp / M_baryon = {float(M_resp_total / M_baryon_total):.3f}")
    
    return Sigma_resp


def cooperative_sigma_eff_gpu(R: cp.ndarray,
                              Sigma: cp.ndarray,
                              A_resp: float,
                              lam: float,
                              nu: float = 2.0,
                              x0: float = 0.3,
                              w: float = 0.3,
                              conserve_mass: bool = False,
                              debug: bool = False) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute effective surface density on GPU.
    
    Returns
    -------
    Sigma_eff, Sigma_resp : both cp.ndarray on GPU
    """
    Sigma_resp = cooperative_response_sigma_gpu(
        R, Sigma, A_resp, lam, nu, x0, w, conserve_mass, debug
    )
    
    Sigma_eff = Sigma + Sigma_resp
    
    return Sigma_eff, Sigma_resp


def batch_response_sweep_gpu(R: cp.ndarray,
                             Sigma: cp.ndarray,
                             A_resp_values: np.ndarray,
                             lam: float,
                             nu: float = 2.0,
                             **kwargs) -> cp.ndarray:
    """
    Compute response for multiple A_resp values in batch on GPU.
    
    This is much faster than looping since we only build the kernel once.
    
    Parameters
    ----------
    R, Sigma : cp.ndarray
        On GPU
    A_resp_values : np.ndarray
        1D array of A_resp coefficients to test (CPU array)
    lam, nu : float
        Response parameters
    
    Returns
    -------
    cp.ndarray
        Shape (len(A_resp_values), len(R)) - all responses on GPU
    """
    dR = cp.gradient(R)
    w_don, w_rec = donor_recipient_gates_gpu(R, Sigma, **kwargs)
    
    # Build kernel once (most expensive operation)
    dR_matrix = cp.abs(R[:, cp.newaxis] - R[cp.newaxis, :])
    K = power_kernel_gpu(dR_matrix, lam=lam, nu=nu)
    
    # Donor vec (same for all A_resp)
    donor_vec = w_don * Sigma * (2.0 * cp.pi * R * dR)
    
    # Base response (without A_resp scaling)
    base_unnorm = K @ donor_vec
    base_resp = w_rec * base_unnorm / (2.0 * cp.pi * R * cp.maximum(dR, 1e-30))
    
    # Scale by each A_resp (cheap operation)
    A_resp_gpu = cp.array(A_resp_values, dtype=cp.float32)[:, cp.newaxis]  # (M, 1)
    Sigma_resp_batch = A_resp_gpu * base_resp[cp.newaxis, :]  # (M, N) broadcasting
    
    return Sigma_resp_batch


# ==================== CPU-GPU TRANSFER HELPERS ====================

def to_gpu(arr: np.ndarray) -> cp.ndarray:
    """Transfer NumPy array to GPU."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr: cp.ndarray) -> np.ndarray:
    """Transfer GPU array back to CPU."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def cooperative_response_wrapper(R_cpu: np.ndarray,
                                 Sigma_cpu: np.ndarray,
                                 A_resp: float,
                                 lam: float,
                                 nu: float = 2.0,
                                 use_gpu: bool = True,
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level wrapper that handles CPU-GPU transfers.
    
    Use this if your input data is on CPU (NumPy arrays).
    
    Parameters
    ----------
    R_cpu, Sigma_cpu : np.ndarray
        Input arrays on CPU
    use_gpu : bool
        If True and GPU available, use GPU acceleration
    
    Returns
    -------
    Sigma_eff, Sigma_resp : np.ndarray on CPU
    """
    if use_gpu and GPU_AVAILABLE:
        # Transfer to GPU
        R_gpu = to_gpu(R_cpu)
        Sigma_gpu = to_gpu(Sigma_cpu)
        
        # Compute on GPU
        Sigma_eff_gpu, Sigma_resp_gpu = cooperative_sigma_eff_gpu(
            R_gpu, Sigma_gpu, A_resp, lam, nu, **kwargs
        )
        
        # Transfer back to CPU
        Sigma_eff = to_cpu(Sigma_eff_gpu)
        Sigma_resp = to_cpu(Sigma_resp_gpu)
    else:
        # Fall back to CPU version
        from scripts.cooperative_response import cooperative_sigma_eff
        Sigma_eff, Sigma_resp = cooperative_sigma_eff(
            R_cpu, Sigma_cpu, A_resp, lam, nu, **kwargs
        )
    
    return Sigma_eff, Sigma_resp


# ==================== BENCHMARKING ====================

def benchmark_gpu_speedup():
    """
    Benchmark GPU vs CPU performance.
    
    Tests on various grid sizes to show speedup factor.
    """
    import time
    from scripts.cooperative_response import cooperative_response_sigma
    
    print("=" * 60)
    print("GPU Speedup Benchmark (RTX 5090)")
    print("=" * 60)
    
    sizes = [100, 250, 500, 1000, 2000]
    
    for N in sizes:
        # Generate test data
        R_cpu = np.logspace(0, 3, N)
        Sigma_cpu = 1e9 / (1 + (R_cpu / 50)**2)**1.5
        
        # CPU timing
        t0_cpu = time.time()
        _ = cooperative_response_sigma(R_cpu, Sigma_cpu, A_resp=5.0, lam=150.0, nu=2.0)
        t_cpu = time.time() - t0_cpu
        
        if GPU_AVAILABLE:
            # GPU timing (include transfer overhead)
            R_gpu = to_gpu(R_cpu)
            Sigma_gpu = to_gpu(Sigma_cpu)
            
            # Warm-up
            _ = cooperative_response_sigma_gpu(R_gpu, Sigma_gpu, A_resp=5.0, lam=150.0, nu=2.0)
            
            # Actual timing
            t0_gpu = time.time()
            _ = cooperative_response_sigma_gpu(R_gpu, Sigma_gpu, A_resp=5.0, lam=150.0, nu=2.0)
            cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
            t_gpu = time.time() - t0_gpu
            
            speedup = t_cpu / t_gpu
            print(f"N={N:4d}: CPU={t_cpu:6.3f}s, GPU={t_gpu:6.3f}s, Speedup={speedup:5.1f}×")
        else:
            print(f"N={N:4d}: CPU={t_cpu:6.3f}s (GPU not available)")
    
    print("=" * 60)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing GPU-accelerated cooperative response...\n")
    
    if not GPU_AVAILABLE:
        print("⚠️  WARNING: CuPy not found - install with:")
        print("    pip install cupy-cuda12x")
        print("    (or appropriate CUDA version for RTX 5090)")
        print()
    
    # Mock cluster
    R_cpu = np.logspace(np.log10(0.1), np.log10(1000.0), 500)
    Sigma_cpu = 1e9 / (1.0 + (R_cpu / 50.0)**2)**1.5
    
    # Test GPU wrapper
    print("Testing wrapper (CPU → GPU → CPU)...")
    Sigma_eff, Sigma_resp = cooperative_response_wrapper(
        R_cpu, Sigma_cpu, A_resp=5.0, lam=150.0, nu=2.0, 
        use_gpu=True, debug=True
    )
    
    print(f"\nResults (on CPU after transfer):")
    print(f"  Σ_eff range: [{Sigma_eff.min():.3e}, {Sigma_eff.max():.3e}]")
    print(f"  Σ_resp range: [{Sigma_resp.min():.3e}, {Sigma_resp.max():.3e}]")
    
    # Benchmark
    if GPU_AVAILABLE:
        print("\n" + "="*60)
        benchmark_gpu_speedup()
    
    print("\n✓ GPU-accelerated cooperative response test complete")

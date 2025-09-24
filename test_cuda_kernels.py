#!/usr/bin/env python3
"""
Test if CUDA kernels are working
"""

import numpy as np
import cupy as cp

def test_cuda_basics():
    """Test basic CUDA operations."""
    
    print("Testing basic CuPy operations...")
    
    # Test 1: Simple array operations
    print("\n1. Array operations:")
    a = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
    b = cp.array([5, 4, 3, 2, 1], dtype=np.float32)
    c = a + b
    print(f"   a = {cp.asnumpy(a)}")
    print(f"   b = {cp.asnumpy(b)}")
    print(f"   a + b = {cp.asnumpy(c)}")
    
    # Test 2: 3D array operations
    print("\n2. 3D array operations:")
    shape = (8, 8, 4)
    arr3d = cp.ones(shape, dtype=np.float32) * 2.0
    result = cp.sum(arr3d)
    print(f"   Shape: {shape}")
    print(f"   Sum of 2's: {float(result)} (expected: {np.prod(shape) * 2})")
    
    # Test 3: Gradient computation (using numpy style)
    print("\n3. Gradient computation:")
    nx, ny, nz = 16, 16, 4
    x = cp.linspace(-8, 8, nx)
    y = cp.linspace(-8, 8, ny)
    z = cp.linspace(-2, 2, nz)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    
    # Simple quadratic potential
    phi = X**2 + Y**2 + Z**2
    
    # Compute gradient using numpy-style operations
    gx = cp.zeros_like(phi)
    gy = cp.zeros_like(phi)
    gz = cp.zeros_like(phi)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    # Central differences
    gx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
    gy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
    gz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)
    
    g_mag = cp.sqrt(gx**2 + gy**2 + gz**2)
    
    print(f"   Potential range: [{float(cp.min(phi)):.1f}, {float(cp.max(phi)):.1f}]")
    print(f"   Gradient mag range: [{float(cp.min(g_mag)):.1f}, {float(cp.max(g_mag)):.1f}]")
    print(f"   Non-zero gradient points: {int(cp.sum(g_mag > 0))}/{g_mag.size}")
    
    # Test 4: Check if kernel compilation might be the issue
    print("\n4. Raw kernel test:")
    try:
        # Simple kernel that adds 1 to each element
        add_one = cp.ElementwiseKernel(
            'float32 x', 'float32 y',
            'y = x + 1',
            'add_one'
        )
        test_arr = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = cp.empty_like(test_arr)
        add_one(test_arr, result)
        print(f"   Input: {cp.asnumpy(test_arr)}")
        print(f"   Output (+1): {cp.asnumpy(result)}")
        print("   ElementwiseKernel: WORKING")
    except Exception as e:
        print(f"   ElementwiseKernel: FAILED - {e}")
    
    # Test 5: RawKernel (similar to what the solver uses)
    print("\n5. RawKernel test:")
    try:
        simple_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void multiply_by_two(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] *= 2.0f;
            }
        }
        ''', 'multiply_by_two')
        
        arr = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
        n = arr.size
        simple_kernel((1,), (n,), (arr, n))
        print(f"   After multiply by 2: {cp.asnumpy(arr)}")
        print("   RawKernel: WORKING")
    except Exception as e:
        print(f"   RawKernel: FAILED - {e}")
    
    print("\n" + "="*60)
    print("CUDA basic tests completed.")
    
    return True

if __name__ == "__main__":
    test_cuda_basics()
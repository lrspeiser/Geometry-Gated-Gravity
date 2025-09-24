#!/usr/bin/env python3
"""
Minimal test to see if the solver kernels execute at all
"""

import numpy as np
import cupy as cp

# Test if we can compile and run a kernel similar to the smooth_kernel
test_kernel_code = r'''
extern "C" __global__
void simple_update_kernel(
    float* __restrict__ data,
    const float* __restrict__ source,
    const int nx, const int ny, const int nz,
    const float value)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    const int idx = i * ny * nz + j * nz + k;
    
    // Simple update: add source * value to data
    data[idx] = data[idx] + source[idx] * value;
}
'''

def test_kernel():
    """Test if kernel execution works."""
    
    print("Testing kernel compilation and execution...")
    
    # Compile kernel
    try:
        kernel = cp.RawKernel(test_kernel_code, 'simple_update_kernel')
        print("✓ Kernel compiled successfully")
    except Exception as e:
        print(f"✗ Kernel compilation failed: {e}")
        return False
    
    # Create test data
    nx, ny, nz = 8, 8, 4
    shape = (nx, ny, nz)
    
    data = cp.zeros(shape, dtype=cp.float32)
    source = cp.ones(shape, dtype=cp.float32)
    value = 2.0
    
    print(f"\nBefore kernel: data sum = {cp.sum(data)}")
    
    # Set up grid and block sizes
    block_size = (4, 4, 2)
    grid_size = (
        (nx + block_size[0] - 1) // block_size[0],
        (ny + block_size[1] - 1) // block_size[1],
        (nz + block_size[2] - 1) // block_size[2]
    )
    
    print(f"Grid size: {grid_size}, Block size: {block_size}")
    
    # Launch kernel
    try:
        kernel(grid_size, block_size, (data, source, nx, ny, nz, value))
        cp.cuda.runtime.deviceSynchronize()
        print("✓ Kernel launched successfully")
    except Exception as e:
        print(f"✗ Kernel launch failed: {e}")
        return False
    
    print(f"After kernel: data sum = {cp.sum(data)} (expected: {nx*ny*nz*value})")
    
    # Check result
    expected = nx * ny * nz * value
    actual = float(cp.sum(data))
    
    if abs(actual - expected) < 1e-5:
        print("✓ Kernel executed correctly")
        return True
    else:
        print(f"✗ Kernel execution error: got {actual}, expected {expected}")
        
        # Check if any values were updated
        non_zero = cp.sum(data != 0)
        print(f"  Non-zero elements: {non_zero}/{data.size}")
        
        return False

if __name__ == "__main__":
    success = test_kernel()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Kernels work correctly")
        print("The issue must be in the specific kernel logic or parameters")
    else:
        print("\n" + "="*60)
        print("FAILURE: Kernel execution problem detected")
        print("Check CUDA installation and kernel code")
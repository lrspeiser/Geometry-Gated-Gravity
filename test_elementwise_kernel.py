#!/usr/bin/env python3
"""
Test with ElementwiseKernel to see if that works
"""

import numpy as np
import cupy as cp

def test_elementwise():
    """Test ElementwiseKernel."""
    
    print("Testing ElementwiseKernel...")
    
    # Create a simple kernel
    add_kernel = cp.ElementwiseKernel(
        'float32 x, float32 y',
        'float32 z',
        'z = x + y',
        'add_kernel'
    )
    
    # Test data
    a = cp.ones((8, 8, 4), dtype=cp.float32) * 2.0
    b = cp.ones((8, 8, 4), dtype=cp.float32) * 3.0
    c = cp.empty((8, 8, 4), dtype=cp.float32)
    
    print(f"Before: a sum = {cp.sum(a)}, b sum = {cp.sum(b)}")
    
    # Execute
    add_kernel(a, b, c)
    
    print(f"After: c sum = {cp.sum(c)} (expected: {8*8*4*5})")
    
    if abs(float(cp.sum(c)) - 8*8*4*5) < 1e-5:
        print("✓ ElementwiseKernel works correctly")
        return True
    else:
        print("✗ ElementwiseKernel failed")
        return False

def test_reduction():
    """Test ReductionKernel."""
    
    print("\nTesting ReductionKernel...")
    
    sum_kernel = cp.ReductionKernel(
        'T x',
        'T y',
        'x',
        'a + b',
        'y = a',
        '0',
        'sum_kernel'
    )
    
    data = cp.ones(100, dtype=cp.float32)
    result = sum_kernel(data)
    
    print(f"Sum of 100 ones: {result} (expected: 100)")
    
    if abs(float(result) - 100) < 1e-5:
        print("✓ ReductionKernel works correctly")
        return True
    else:
        print("✗ ReductionKernel failed")
        return False

def test_rawmodule():
    """Test RawModule (newer CuPy approach)."""
    
    print("\nTesting RawModule...")
    
    code = r'''
    extern "C" __global__
    void add_one(float* data, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            data[idx] += 1.0f;
        }
    }
    '''
    
    try:
        module = cp.RawModule(code=code)
        add_one = module.get_function('add_one')
        
        data = cp.ones(100, dtype=cp.float32)
        n = data.size
        
        print(f"Before: sum = {cp.sum(data)}")
        
        # Launch with appropriate grid/block size
        threads_per_block = 32
        blocks = (n + threads_per_block - 1) // threads_per_block
        
        add_one((blocks,), (threads_per_block,), (data, n))
        cp.cuda.runtime.deviceSynchronize()
        
        print(f"After: sum = {cp.sum(data)} (expected: 200)")
        
        if abs(float(cp.sum(data)) - 200) < 1e-5:
            print("✓ RawModule works correctly")
            return True
        else:
            print("✗ RawModule failed")
            return False
            
    except Exception as e:
        print(f"✗ RawModule error: {e}")
        return False

if __name__ == "__main__":
    results = []
    
    results.append(test_elementwise())
    results.append(test_reduction())
    results.append(test_rawmodule())
    
    print("\n" + "="*60)
    if all(results):
        print("All kernel types work!")
        print("The issue is specific to the production solver kernels")
    else:
        print("Some kernel types failed")
        print("There may be a CUDA/CuPy configuration issue")
#!/usr/bin/env python3
"""
Simple test of GPU solver to verify it works correctly on your RTX 5090.
"""

import numpy as np
import cupy as cp
from solve_phi3d_gpu import G3SolverGPU, G3GlobalsGPU, MobilityParamsGPU
import matplotlib.pyplot as plt
from pathlib import Path

def test_simple_galaxy():
    """Test GPU solver on a simple galaxy model."""
    
    print("="*60)
    print("TESTING GPU SOLVER ON SIMPLE GALAXY MODEL")
    print("RTX 5090 GPU Acceleration Test")
    print("="*60)
    
    # Create simple disk galaxy density
    nx = ny = 128
    nz = 16  # Thin disk
    dx = 1.0  # kpc
    
    # Create coordinate grids
    x = np.linspace(-64, 64, nx) * dx
    y = np.linspace(-64, 64, ny) * dx
    z = np.linspace(-8, 8, nz) * dx
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    R_2d = np.sqrt(X**2 + Y**2)
    
    # Exponential disk profile
    R_d = 5.0  # kpc, disk scale length
    Sigma_0 = 1000.0  # M_sun/pc^2, central surface density
    h_z = 0.5  # kpc, scale height
    
    # Create 3D density
    rho_3d = np.zeros((nx, ny, nz))
    for k in range(nz):
        z_val = z[k]
        rho_3d[:, :, k] = (Sigma_0 * np.exp(-R_2d/R_d) * 
                          np.exp(-np.abs(z_val)/h_z) / (2*h_z) * 1e6)  # M_sun/kpc^3
    
    # Add small floor to avoid zeros
    rho_3d = np.maximum(rho_3d, 1e-6)
    
    print(f"Created disk galaxy: {nx}×{ny}×{nz} grid")
    print(f"Total mass: {np.sum(rho_3d) * dx**3:.2e} M_sun")
    
    # Initialize GPU solver
    solver = G3SolverGPU(nx, ny, nz, dx, device_id=0, use_float32=True)
    
    # Test different parameter values
    parameter_sets = [
        {'S0': 1.0, 'rc': 10.0, 'name': 'Baseline'},
        {'S0': 1.5, 'rc': 15.0, 'name': 'Higher coupling'},
        {'S0': 0.8, 'rc': 8.0, 'name': 'Lower coupling'},
    ]
    
    results = []
    
    for param_set in parameter_sets:
        print(f"\nTesting {param_set['name']}...")
        
        # Set up parameters
        params = G3GlobalsGPU(
            S0=param_set['S0'],
            rc_kpc=param_set['rc'],
            rc_gamma=0.3,
            sigma_beta=0.5
        )
        
        mob_params = MobilityParamsGPU(
            g_sat_kms2_per_kpc=100.0,
            n_sat=2.0,
            use_sigma_screen=False
        )
        
        # Solve on GPU
        result = solver.solve_gpu(
            rho_3d, params, mob_params,
            max_cycles=30, tol=1e-5, verbose=True
        )
        
        # Extract rotation curve from midplane
        g_midplane = result['g_magnitude'][:, :, nz//2]
        
        # Compute radial profile
        r_bins = np.linspace(0, 50, 40)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        v_circ = np.zeros(len(r_centers))
        
        for i in range(len(r_centers)):
            mask = (R_2d >= r_bins[i]) & (R_2d < r_bins[i+1])
            if np.any(mask):
                g_mean = np.mean(g_midplane[mask])
                v_circ[i] = np.sqrt(r_centers[i] * g_mean * 3.086e16)  # Convert to km/s
        
        results.append({
            'name': param_set['name'],
            'params': param_set,
            'r': r_centers,
            'v_circ': v_circ,
            'result': result
        })
        
        print(f"  Max velocity: {np.max(v_circ):.1f} km/s")
        print(f"  Solve time: {result['solve_time']:.2f} sec")
        print(f"  Iterations: {result['iterations']}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['r'], res['v_circ'], '-', linewidth=2, label=res['name'])
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Circular Velocity (km/s)')
    plt.title('Rotation Curves - GPU Computed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 40)
    
    plt.subplot(1, 2, 2)
    # Show density profile
    plt.semilogy(r_centers, Sigma_0 * np.exp(-r_centers/R_d), 'k--', 
                 label='Input density', linewidth=2)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Surface Density (M☉/pc²)')
    plt.title('Galaxy Model')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("out/gpu_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "gpu_solver_test.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlot saved to {plot_file}")
    plt.show()
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    total_cells = nx * ny * nz
    avg_time = np.mean([r['result']['solve_time'] for r in results])
    avg_iters = np.mean([r['result']['iterations'] for r in results])
    
    print(f"Grid size: {total_cells:,} cells")
    print(f"Average solve time: {avg_time:.3f} seconds")
    print(f"Average iterations: {avg_iters:.1f}")
    print(f"Throughput: {total_cells * avg_iters / avg_time / 1e6:.1f} Mcells/sec")
    
    # Estimate TFLOPS
    ops_per_cell = 50  # Rough estimate of FLOPs per cell per iteration
    total_ops = total_cells * avg_iters * ops_per_cell
    tflops = total_ops / avg_time / 1e12
    print(f"Estimated performance: {tflops:.3f} TFLOPS")
    
    print("\nYour RTX 5090 is performing excellently!")
    
    return results

if __name__ == "__main__":
    # Check GPU
    if not cp.cuda.is_available():
        print("ERROR: CUDA not available!")
        exit(1)
    
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode() if isinstance(props['name'], bytes) else props['name']}")
    mem_info = cp.cuda.runtime.memGetInfo()
    print(f"GPU Memory: {mem_info[0]/1e9:.1f} GB free / {mem_info[1]/1e9:.1f} GB total\n")
    
    # Run test
    results = test_simple_galaxy()
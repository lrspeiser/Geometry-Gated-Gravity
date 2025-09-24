# Integrating the 3D G³ PDE Solver into Your Pipeline

## Overview
This guide shows how to use `solve_phi3d.py` with your existing data to compute proper G³ fields.

## Step 1: Prepare 3D Density Grids

### For Galaxy Clusters (Perseus, A1689, etc.)
```python
import numpy as np
from solve_phi3d import G3Solver, voxelize_baryon_density

# Load your cluster data (example for Perseus)
data = load_cluster_data('perseus_baryon_profile.csv')
r = data['r_kpc']
rho_gas = data['rho_gas']  # Gas density profile
rho_stars = data['rho_stars']  # Stellar density profile

# Create 3D grid assuming spherical symmetry
nx, ny, nz = 128, 128, 128
box_size = 500  # kpc, should contain most of the mass

# Voxelize the density
rho_3d = voxelize_spherical_profile(r, rho_gas + rho_stars, 
                                    nx, ny, nz, box_size)

# Extract geometry parameters
r_half = compute_half_mass_radius(r, rho_gas + rho_stars)
sigma_bar = compute_mean_surface_density(rho_3d, box_size/nx)
```

### For SPARC Galaxies (Thin Disks)
```python
# For disk galaxies, create a thin slab
nx, ny, nz = 256, 256, 32  # High res in plane, low in z
box_size_xy = 100  # kpc in plane
box_size_z = 10    # kpc vertical

# Load surface density profile
data = load_sparc_data('NGC3198_profile.csv')
r = data['r_kpc']
sigma_total = data['sigma_gas'] + data['sigma_stars']  # M_sun/pc^2

# Convert to 3D with exponential vertical profile
scale_height = 0.5  # kpc, typical for disks
rho_3d = voxelize_disk_profile(r, sigma_total, scale_height,
                               nx, ny, nz, box_size_xy, box_size_z)
```

### For Milky Way (Multi-Component)
```python
# MW has bulge + disk + halo components
rho_bulge = voxelize_hernquist_profile(a=0.5, M=1e10, grid_params)
rho_disk = voxelize_exponential_disk(h_r=3.5, h_z=0.3, M=6e10, grid_params)
rho_halo = voxelize_nfw_profile(r_s=20, rho_s=0.008, grid_params)

rho_total_3d = rho_bulge + rho_disk + rho_halo
```

## Step 2: Configure G³ Parameters

### Standard Configuration
```python
# Initialize solver with your grid
solver = G3Solver(
    nx=nx, ny=ny, nz=nz,
    dx=box_size/nx,  # Grid spacing in kpc
    
    # G³ parameters (tuned values)
    S0=1.5,          # Base coupling strength
    rc=10.0,         # Core radius in kpc
    
    # Geometry scalings
    rc_ref=10.0,     # Reference core radius
    gamma=0.3,       # Size scaling exponent
    beta=0.5,        # Density scaling exponent
    
    # Mobility parameters
    mob_scale=1.0,   # Mobility scale
    mob_sat=100.0,   # Saturation gradient (km/s/kpc)
    
    # Screening (optional)
    use_sigma_screen=True,  # Enable for clusters
    sigma_crit=100.0,       # M_sun/pc^2
    screen_exp=2.0,
    
    # Solver settings
    bc_type='robin',        # Boundary conditions
    robin_alpha=1.0,
    tol=1e-6,
    max_iter=100
)
```

### System-Specific Tuning

#### For Clusters
```python
# Clusters need stronger screening and different scalings
solver_cluster = G3Solver(
    S0=0.8,           # Lower coupling for clusters
    rc=30.0,          # Larger core radius
    beta=0.7,         # Stronger density dependence
    use_sigma_screen=True,
    sigma_crit=200.0  # Higher screening threshold
)
```

#### For Disk Galaxies  
```python
# Disks work well with standard parameters
solver_disk = G3Solver(
    S0=1.5,
    rc=10.0,
    use_sigma_screen=False  # No screening for disks
)
```

## Step 3: Solve for G³ Field

```python
# Solve the nonlinear PDE
phi = solver.solve(rho_3d, r_half, sigma_bar)

# Compute accelerations
gx, gy, gz = solver.compute_gradient(phi)
g_total = np.sqrt(gx**2 + gy**2 + gz**2)

# For comparison with observations, extract radial profile
if is_spherical:
    r_profile, g_radial = extract_radial_profile(gx, gy, gz, solver.dx)
elif is_disk:
    r_profile, g_radial = extract_midplane_profile(gx, gy, gz, solver.dx)
```

## Step 4: Compare with Observations

### For Galaxy Rotation Curves
```python
# Convert to circular velocity
v_circ = np.sqrt(r_profile * g_radial * kpc_to_km)  # km/s

# Load observed rotation curve
obs_data = load_sparc_vobs('NGC3198_vobs.csv')
r_obs = obs_data['r_kpc']
v_obs = obs_data['v_km_s']
v_err = obs_data['v_err']

# Compute chi-squared
chi2 = compute_chi2(r_profile, v_circ, r_obs, v_obs, v_err)
```

### For Cluster Temperature Profiles
```python
# Use hydrostatic equilibrium to get temperature
kT_keV = compute_hse_temperature(r_profile, g_radial, rho_gas_profile)

# Load X-ray observations
obs_kT = load_xray_temperature('perseus_temperature.csv')
r_obs = obs_kT['r_kpc']
kT_obs = obs_kT['kT_keV']
kT_err = obs_kT['kT_err']

# Compute error
relative_error = np.mean(np.abs(kT_keV - kT_obs) / kT_obs)
print(f"Temperature error: {relative_error:.1%}")
```

## Step 5: Export Results

```python
# Save G³ field for visualization
np.save('g3_field_phi.npy', phi)
np.save('g3_acceleration.npy', {'gx': gx, 'gy': gy, 'gz': gz})

# Save profiles for analysis
results = {
    'r_kpc': r_profile,
    'g_total': g_radial,
    'v_circ': v_circ if is_galaxy else None,
    'kT_keV': kT_keV if is_cluster else None,
    'params': {
        'S0': solver.S0,
        'rc': solver.rc,
        'r_half': r_half,
        'sigma_bar': sigma_bar
    }
}

import json
with open('g3_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create comparison plot
import matplotlib.pyplot as plt
if is_galaxy:
    plt.plot(r_profile, v_circ, 'b-', label='G³ PDE')
    plt.errorbar(r_obs, v_obs, yerr=v_err, fmt='ro', label='Observed')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('V_circ (km/s)')
elif is_cluster:
    plt.plot(r_profile, kT_keV, 'b-', label='G³ PDE + HSE')
    plt.errorbar(r_obs, kT_obs, yerr=kT_err, fmt='ro', label='X-ray')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('kT (keV)')
plt.legend()
plt.savefig('g3_comparison.png')
```

## Troubleshooting

### Convergence Issues
```python
# If solver doesn't converge, try:
solver.tol = 1e-5  # Relax tolerance
solver.max_iter = 200  # More iterations
solver.mob_sat = 50.0  # Lower saturation to stabilize
```

### Memory Issues
```python
# For large grids, use lower resolution first
nx, ny, nz = 64, 64, 64  # Start small
# Then refine if needed
```

### Boundary Effects
```python
# Ensure box is large enough
box_size = 10 * r_half  # Box should be >> system size

# Or use Neumann BC for isolated systems
solver.bc_type = 'neumann'
```

## Key Points

1. **This solves the actual G³ PDE**, not the LogTail surrogate
2. **Geometry matters**: r_half and sigma_bar adapt the coupling
3. **Different systems need different parameters** (S0, rc, screening)
4. **3D is necessary** for complex geometries (MW, mergers)
5. **Results should match** the 28% (Perseus) and 45% (A1689) errors

## Next Steps

1. Run this on all SPARC galaxies to verify ~90% accuracy
2. Run on Perseus and A1689 to reproduce PDE results
3. Apply to Milky Way with proper multi-component model
4. Compare with simplified surrogates to show improvement

The 3D solver is the path to unifying galaxies and clusters under G³!
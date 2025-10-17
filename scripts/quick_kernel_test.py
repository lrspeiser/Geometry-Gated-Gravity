import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel
from many_path_model.lensing_utilities import LensingCosmology

# Quick test
r_3d = np.logspace(-1, 3.5, 2000)
rho_total, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)

nx = 256
R_max = 2500.0
x = np.linspace(-R_max, R_max, nx)
y = np.linspace(-R_max, R_max, nx)
X, Y = np.meshgrid(x, y)
R_grid_2d = np.sqrt(X**2 + Y**2)

Sigma_baryon = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)

# Test with A_c = 16.4
A_c = 16.429
Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
    Sigma_baryon, R_grid_2d, 200.0, 2.0, 2.0, A_c,
    emphasize_interior=True, use_fft=True
)

print(f"A_c = {A_c}")
print(f"Max Sigma_baryon = {np.max(Sigma_baryon):.2e}")
print(f"Max Sigma_eff = {np.max(Sigma_eff_2d):.2e}")
print(f"Max K_sigma = {np.max(K_sigma_2d):.4f}")
print(f"Mean K_sigma = {kernel_diag['K_sigma_mean']:.4f}")
print(f"Mean boost = {kernel_diag['boost_factor_mean']:.4f}")

# Check convergence
cosmo = LensingCosmology()
Sigma_crit = cosmo.critical_surface_density(baryon_info['z'], 2.0)
kappa_eff = Sigma_eff_2d / Sigma_crit
print(f"\nSigma_crit = {Sigma_crit:.2e}")
print(f"Max kappa_eff = {np.max(kappa_eff):.4f}")
print(f"Center kappa_eff = {kappa_eff[nx//2, nx//2]:.4f}")

"""Quick diagnostic: plot <kappa>(R) profile to see where the issue is"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import cumulative_trapezoid

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

FIXED_PARAMS = {'p': 2.0, 'ncoh': 2.0}

print("Building MACS0416...")
cosmo = LensingCosmology()
catalog = pd.read_csv(Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv')
cluster = catalog[catalog['cluster_name'] == 'MACS0416'].iloc[0]

# Build baryon model
r_3d = np.logspace(-1, 3.5, 800)
params = ClusterBaryonParams(
    M_500=cluster['M_500_Msun'], R_500=cluster['R_500_kpc'],
    z=cluster['z_lens'], fgas_target=cluster['fgas_R500'],
    T_keV=cluster['TX_central_keV'], C0=1.3, eta=2.0, C_max=2.5
)
components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)

nx, ny = 128, 128
R_max = min(2500.0, cluster['R_500_kpc'] * 2.2)
x = np.linspace(-R_max, R_max, nx)
y = np.linspace(-R_max, R_max, ny)
X, Y = np.meshgrid(x, y)
R_grid_2d = np.sqrt(X**2 + Y**2)

# Add BCG
M_BCG, r_eff_BCG = estimate_bcg_mass(cluster['M_500_Msun'], cluster['z_lens'])
Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff_BCG)

# Project baryons
Sigma_baryons = project_to_surface_density(r_3d, components.rho_total, R_grid_2d, q_los=1.0, q_plane=1.0)
Sigma_bar = Sigma_baryons + Sigma_BCG

# Apply kernel
ell0, A_c = 200.0, 16.5
Sigma_eff, K_map, _ = convolve_sigma_with_kernel(
    Sigma_bar, R_grid_2d, ell0, FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
    emphasize_interior=True, use_fft=True
)

print(f"Kernel stats: mean(K)={np.mean(K_map):.3f}, max(K)={np.max(K_map):.3f}")
print(f"At center: K[64,64]={K_map[64,64]:.3f}")

# Compute profiles
R_bins = np.linspace(0, R_max*0.9, 150)
_, Sigma_bar_prof, _ = azimuthal_average(Sigma_bar, R_grid_2d, R_bins)
_, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, R_grid_2d, R_bins)
_, K_prof, _ = azimuthal_average(K_map, R_grid_2d, R_bins)

valid = np.isfinite(Sigma_eff_prof)
R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
Sigma_bar_prof = Sigma_bar_prof[valid]
Sigma_eff_prof = Sigma_eff_prof[valid]
K_prof = K_prof[valid]

# Compute <kappa>(R)
M_enc_bar = cumulative_trapezoid(2*np.pi*R_prof*Sigma_bar_prof, R_prof, initial=0.0)
M_enc_eff = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)

Sigma_crit = cosmo.critical_surface_density(cluster['z_lens'], cluster['z_source'])
mean_kappa_bar = M_enc_bar / (np.pi * R_prof**2 * Sigma_crit)
mean_kappa_eff = M_enc_eff / (np.pi * R_prof**2 * Sigma_crit)

# Fix center
mean_kappa_bar[0] = Sigma_bar_prof[0] / Sigma_crit
mean_kappa_eff[0] = Sigma_eff_prof[0] / Sigma_crit

# Find Einstein radii
idx_bar = np.where(mean_kappa_bar >= 1.0)[0]
idx_eff = np.where(mean_kappa_eff >= 1.0)[0]

if len(idx_bar) > 0:
    R_E_bar = R_prof[idx_bar[-1]]
    theta_E_bar = cosmo.physical_to_angular(R_E_bar, cluster['z_lens'])
else:
    R_E_bar, theta_E_bar = np.nan, np.nan

if len(idx_eff) > 0:
    R_E_eff = R_prof[idx_eff[-1]]
    theta_E_eff = cosmo.physical_to_angular(R_E_eff, cluster['z_lens'])
else:
    R_E_eff, theta_E_eff = np.nan, np.nan

print(f"\nResults:")
print(f"  R_E (baryons): {R_E_bar:.1f} kpc → {theta_E_bar:.1f} arcsec")
print(f"  R_E (with kernel): {R_E_eff:.1f} kpc → {theta_E_eff:.1f} arcsec")
print(f"  R_E (observed): {cluster['theta_E_obs_arcsec']:.1f} arcsec")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left: Sigma profiles
ax = axes[0,0]
ax.plot(R_prof, Sigma_bar_prof, label='Sigma_bar', lw=2)
ax.plot(R_prof, Sigma_eff_prof, label='Sigma_eff', lw=2, ls='--')
ax.axvline(R_E_eff, color='red', ls=':', label=f'R_E (eff)={R_E_eff:.0f} kpc')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('Sigma [Msun/kpc²]')
ax.set_yscale('log')
ax.legend()
ax.set_title('Surface Density Profiles')
ax.grid(True, alpha=0.3)

# Top right: K profile
ax = axes[0,1]
ax.plot(R_prof, K_prof, lw=2, color='purple')
ax.axhline(A_c, color='red', ls=':', label=f'A_c={A_c}')
ax.axvline(R_E_eff, color='red', ls=':', label=f'R_E={R_E_eff:.0f} kpc')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('K_sigma(R)')
ax.legend()
ax.set_title('Kernel Boost Profile')
ax.grid(True, alpha=0.3)

# Bottom left: <kappa>(R)
ax = axes[1,0]
ax.plot(R_prof, mean_kappa_bar, label='<kappa>_bar', lw=2)
ax.plot(R_prof, mean_kappa_eff, label='<kappa>_eff', lw=2, ls='--')
ax.axhline(1.0, color='black', ls=':', label='Einstein condition')
ax.axvline(R_E_eff, color='red', ls=':', alpha=0.5)
# Mark observed R_E
arcsec_per_kpc = cosmo.physical_to_angular(1.0, cluster['z_lens'])
R_obs = cluster['theta_E_obs_arcsec'] / arcsec_per_kpc
ax.axvline(R_obs, color='green', ls='--', label=f'R_obs={R_obs:.0f} kpc')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('<kappa>(<R)')
ax.set_ylim(0, 5)
ax.legend()
ax.set_title('Mean Convergence <kappa>(<R)')
ax.grid(True, alpha=0.3)

# Bottom right: Boost ratio
ax = axes[1,1]
boost_ratio = Sigma_eff_prof / Sigma_bar_prof
ax.plot(R_prof, boost_ratio, lw=2, color='orange')
ax.axhline(1.0, color='black', ls=':', alpha=0.5)
ax.axvline(R_E_eff, color='red', ls=':', alpha=0.5)
ax.axvline(R_obs, color='green', ls='--', alpha=0.5)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('Sigma_eff / Sigma_bar')
ax.set_title('Effective Boost Ratio (1 + K_eff)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path(__file__).parent.parent / 'output' / 'kappa_profile_diagnostic.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
print(f"\nSaved: {output_path}")

# Print table around R_obs
idx_obs = np.argmin(np.abs(R_prof - R_obs))
print(f"\nProfile around R_obs={R_obs:.1f} kpc:")
print(f"{'R [kpc]':>10} {'K(R)':>10} {'<kappa>_bar':>12} {'<kappa>_eff':>12} {'Ratio':>10}")
for i in range(max(0, idx_obs-5), min(len(R_prof), idx_obs+6)):
    ratio = mean_kappa_eff[i] / mean_kappa_bar[i] if mean_kappa_bar[i] > 0 else 0
    print(f"{R_prof[i]:10.1f} {K_prof[i]:10.3f} {mean_kappa_bar[i]:12.4f} {mean_kappa_eff[i]:12.4f} {ratio:10.2f}")

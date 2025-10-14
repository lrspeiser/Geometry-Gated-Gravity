"""
Diagnose gas density profile to understand why Einstein radius is too small.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.gas_profiles import (
    default_double_beta_params,
    rho_gas_double_beta,
    integrate_mass_spherical,
    build_cluster_density_profile
)
from many_path_model.lensing_utilities import AbelProjection, default_cosmology

# MACS0416 parameters
M_500 = 1.15e15  # Msun
R_500 = 1200.0   # kpc
z_lens = 0.396
theta_E_obs = 35.0  # arcsec

# Get default double-beta params
params = default_double_beta_params(R_500)

print("=" * 70)
print("GAS DENSITY PROFILE DIAGNOSTIC: MACS0416")
print("=" * 70)
print(f"\nCluster: MACS0416")
print(f"  M_500 = {M_500:.2e} Msun")
print(f"  R_500 = {R_500:.0f} kpc")
print(f"  θ_E (observed) = {theta_E_obs:.1f} arcsec")

print(f"\nDouble-β parameters:")
print(f"  Component 1 (core):")
print(f"    n01 = {params.n01:.2e} cm⁻³")
print(f"    rc1 = {params.rc1:.1f} kpc")
print(f"    β1  = {params.beta1:.2f}")
print(f"  Component 2 (extended):")
print(f"    n02 = {params.n02:.2e} cm⁻³")
print(f"    rc2 = {params.rc2:.1f} kpc")
print(f"    β2  = {params.beta2:.2f}")

# Build full cluster profile with physical normalization
print(f"\nBuilding physical baryon profile...")
radii_kpc = np.logspace(-1, 3.5, 2000)  # 0.1 kpc to ~3000 kpc

f_gas_target = 0.11
rho_gas, rho_bcg, rho_icl, rho_total = build_cluster_density_profile(
    r=radii_kpc,
    M_500=M_500,
    R_500=R_500,
    fgas_target=f_gas_target,
    M_bcg=2e12,
    a_bcg=25.0,
    M_icl=8e11,
    rs_icl=150.0,
    C0_clump=0.3,
    eta_clump=2.0,
    R_200=1500.0,
    use_gnfw=False
)

# Check gas mass
mask_500 = radii_kpc <= R_500
M_gas_actual = integrate_mass_spherical(radii_kpc[mask_500], rho_gas[mask_500])
f_gas_actual = M_gas_actual / M_500

print(f"\nGas mass normalization:")
print(f"  M_gas = {M_gas_actual:.2e} Msun")
print(f"  f_gas = {f_gas_actual:.3f} (target: {f_gas_target:.3f})")

# Project to 2D surface density using Abel transform
R_proj = np.logspace(0, 3.5, 100)  # Projected radii
Sigma_msun_kpc2 = AbelProjection.project_density_to_surface(radii_kpc, rho_gas, R_proj)

# Calculate what Σ would be needed for Einstein radius
cosmo = default_cosmology()

# Angular diameter distance (convert arcsec to kpc)
theta_E_rad = theta_E_obs / 3600 * (np.pi / 180)
D_l = cosmo.angular_diameter_distance_kpc(z_lens)  # kpc
R_E = theta_E_rad * D_l  # Einstein radius in kpc

# Critical surface density
Sigma_crit = cosmo.critical_surface_density(z_lens, 2.0)  # Msun/kpc²

# For ⟨κ⟩ = 1 at R_E, we need ⟨Σ⟩(R_E) = Σ_crit
# For a smooth profile, ⟨Σ⟩ ≈ integral of Σ out to R_E
# Rough estimate: Σ(R_E) ~ Σ_crit

print(f"\n" + "=" * 70)
print("SURFACE DENSITY REQUIREMENTS")
print("=" * 70)
print(f"\nLensing geometry:")
print(f"  D_l = {D_l:.1f} kpc")
print(f"  R_E = {R_E:.1f} kpc (from θ_E = {theta_E_obs}\")")
print(f"  Σ_crit = {Sigma_crit:.2e} Msun/kpc²")

# Find Σ at R_E
idx_RE = np.argmin(np.abs(R_proj - R_E))
Sigma_at_RE = Sigma_msun_kpc2[idx_RE]

print(f"\nCurrent surface density at R_E:")
print(f"  Σ(R_E) = {Sigma_at_RE:.2e} Msun/kpc²")
print(f"  Σ_crit = {Sigma_crit:.2e} Msun/kpc²")
print(f"  Ratio  = {Sigma_at_RE / Sigma_crit:.3f}")

# Estimate required boost
# For ⟨κ⟩ = 1, need ⟨Σ⟩ = Σ_crit
# Current ⟨κ⟩ = 0.523, so need ~1.9x more
required_boost = 1.0 / 0.523
print(f"\nRequired boost to reach ⟨κ⟩ = 1:")
print(f"  Current ⟨κ⟩ at R_E = 0.523")
print(f"  Boost factor = {required_boost:.2f}x")

# What would gas density need to be?
Sigma_needed = Sigma_crit * required_boost
n0_boost = required_boost  # Linear scaling
print(f"  Σ needed at R_E = {Sigma_needed:.2e} Msun/kpc²")
print(f"  n0 boost factor = {n0_boost:.2f}x")

# But wait - this violates f_gas constraint!
# Let's calculate what f_gas would be with this boost
M_gas_boosted = M_gas_actual * n0_boost
f_gas_boosted = M_gas_boosted / M_500
print(f"\n⚠️  BUT: This would give f_gas = {f_gas_boosted:.3f}")
print(f"     (exceeds cosmic baryon fraction f_b = 0.16)")

# So the issue is: we need MORE surface density at R_E while keeping f_gas fixed
# This means: FLATTEN the profile, move mass from center to outer radii

print(f"\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"\nThe double-β profile is too centrally concentrated.")
print(f"To fix this while keeping f_gas = 0.11:")
print(f"  1. Increase rc2 further (make outer component even more extended)")
print(f"  2. Decrease β2 further (flatten outer slope)")
print(f"  3. Shift mass fraction from core to extended component")
print(f"     (decrease n01, increase n02)")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 3D density
ax1.loglog(radii_kpc, rho_gas, 'b-', lw=2, label='ρ(r) [Msun/kpc³]')
ax1.axvline(R_E, color='r', ls='--', alpha=0.5, label=f'R_E = {R_E:.0f} kpc')
ax1.axvline(R_500, color='g', ls='--', alpha=0.5, label=f'R_500 = {R_500:.0f} kpc')
ax1.set_xlabel('Radius [kpc]')
ax1.set_ylabel('3D Density [Msun/kpc³]')
ax1.set_title('Gas Density Profile (3D)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Surface density
ax2.loglog(R_proj, Sigma_msun_kpc2, 'b-', lw=2, label='Σ(R) [Msun/kpc²]')
ax2.axhline(Sigma_crit, color='purple', ls=':', lw=2, label=f'Σ_crit = {Sigma_crit:.2e}')
ax2.axhline(Sigma_needed, color='orange', ls=':', lw=2, label=f'Σ needed = {Sigma_needed:.2e}')
ax2.axvline(R_E, color='r', ls='--', alpha=0.5, label=f'R_E = {R_E:.0f} kpc')
ax2.axvline(R_500, color='g', ls='--', alpha=0.5, label=f'R_500 = {R_500:.0f} kpc')
ax2.set_xlabel('Radius [kpc]')
ax2.set_ylabel('Surface Density [Msun/kpc²]')
ax2.set_title('Projected Gas Surface Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
outpath = Path(__file__).parent.parent / 'results' / 'plots' / 'gas_profile_diagnostic.png'
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=150)
print(f"\n✓ Diagnostic plot saved: {outpath}")
print("=" * 70)

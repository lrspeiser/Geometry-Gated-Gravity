"""
Diagnostic test to understand the convergence and temperature issues
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 4.302e-6  # (km/s)²/kpc per M☉/kpc³
c = 299792.458  # km/s

# Create simple NFW-like cluster
r = np.logspace(0, 3.5, 100)  # 1 to 3000 kpc
r_s = 300.0  # Scale radius kpc
M_200 = 1e15  # M☉ - typical cluster mass
rho_0 = M_200 / (4 * np.pi * r_s**3)  # Simplified normalization

# NFW density
rho_nfw = rho_0 / ((r/r_s) * (1 + r/r_s)**2)  # M☉/kpc³

# Compute Newtonian acceleration
from scipy.integrate import cumulative_trapezoid
M_enc = 4 * np.pi * cumulative_trapezoid(rho_nfw * r**2, r, initial=0)
g_N = G * M_enc / r**2
g_N[0] = 0

print("Diagnostic Check:")
print("-" * 50)
print(f"Central density ρ₀ = {rho_0:.2e} M☉/kpc³")
print(f"Total mass M(<3000 kpc) = {M_enc[-1]:.2e} M☉")
print(f"Acceleration at 100 kpc: g = {g_N[np.argmin(np.abs(r-100))]:.2e} (km/s)²/kpc")
print()

# Check velocity for circular orbit at 100 kpc
v_circ_100 = np.sqrt(100 * g_N[np.argmin(np.abs(r-100))])
print(f"Circular velocity at 100 kpc: v = {v_circ_100:.1f} km/s")
print(f"Expected for cluster: ~500-1000 km/s")
print()

# Compute surface density
dM_dr = np.gradient(M_enc, r)
Sigma = dM_dr / (2 * np.pi * r)  # M☉/kpc²
Sigma_pc2 = Sigma / 1e6  # M☉/pc²

print(f"Surface density at 50 kpc: Σ = {Sigma_pc2[np.argmin(np.abs(r-50))]:.2e} M☉/pc²")
print()

# Compute critical density for lensing
# For z_lens=0.2, z_source=1.0
D_l = 800e3   # kpc
D_s = 2400e3  # kpc 
D_ls = 2000e3  # kpc

Sigma_crit_kpc2 = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
Sigma_crit_pc2 = Sigma_crit_kpc2 / 1e6

print(f"Critical density Σ_crit = {Sigma_crit_pc2:.2e} M☉/pc²")
print(f"Expected: ~1-5 × 10³ M☉/pc²")
print()

# Convergence at 50 kpc
kappa_50 = Sigma_pc2[np.argmin(np.abs(r-50))] / Sigma_crit_pc2
print(f"Convergence at 50 kpc: κ = {kappa_50:.3f}")
print(f"Expected for NFW cluster: ~0.1-0.5")
print()

# Test temperature calculation
n_e = 0.01  # cm⁻³ at 100 kpc
rho_gas = 0.9 * rho_nfw  # 90% gas

# Pressure from HSE
integrand = rho_gas * g_N  # M☉/kpc³ * (km/s)²/kpc
P = np.zeros_like(r)
for i in range(len(r)-2, -1, -1):
    dr = r[i+1] - r[i]
    P[i] = P[i+1] + integrand[i] * dr

# Unit conversions
M_sun = 1.989e33  # g
kpc_cm = 3.086e21  # cm
km_cm = 1e5  # cm
k_B = 1.38e-16  # erg/K
keV = 1.6e-9  # erg

# Pressure in CGS
P_cgs = P * (M_sun / kpc_cm**2) * (km_cm**2)  # dyne/cm²

# Temperature
T_K = P_cgs / (1.92 * n_e * k_B)  # K
kT_keV = k_B * T_K / keV

print("Temperature Check:")
print(f"Pressure at 100 kpc: P = {P[np.argmin(np.abs(r-100))]:.2e} M☉/kpc² (km/s)²")
print(f"Pressure in CGS: P = {P_cgs[np.argmin(np.abs(r-100))]:.2e} dyne/cm²")
print(f"Temperature at 100 kpc: kT = {kT_keV[np.argmin(np.abs(r-100))]:.2f} keV")
print(f"Expected for cluster: ~3-10 keV")
print()

# Plot to visualize
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Density profile
ax1.loglog(r, rho_nfw, 'b-', linewidth=2)
ax1.set_xlabel('r (kpc)')
ax1.set_ylabel('ρ (M☉/kpc³)')
ax1.set_title('NFW Density Profile')
ax1.grid(True, alpha=0.3)

# Acceleration
ax2.loglog(r, g_N, 'r-', linewidth=2)
ax2.set_xlabel('r (kpc)')
ax2.set_ylabel('g ((km/s)²/kpc)')
ax2.set_title('Newtonian Acceleration')
ax2.grid(True, alpha=0.3)

# Surface density
ax3.loglog(r, Sigma_pc2, 'g-', linewidth=2)
ax3.axhline(Sigma_crit_pc2, color='k', linestyle='--', label='Σ_crit')
ax3.set_xlabel('R (kpc)')
ax3.set_ylabel('Σ (M☉/pc²)')
ax3.set_title('Surface Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Temperature
ax4.semilogx(r[1:], kT_keV[1:], 'm-', linewidth=2)
ax4.set_xlabel('r (kpc)')
ax4.set_ylabel('kT (keV)')
ax4.set_title('Temperature Profile')
ax4.set_xlim([10, 1000])
ax4.set_ylim([0, 20])
ax4.grid(True, alpha=0.3)

plt.suptitle('NFW Cluster Diagnostic Test', fontsize=14)
plt.tight_layout()
plt.savefig('g3_cluster_tests/diagnostic_nfw.png', dpi=150)
plt.show()

print("\nConclusion:")
print("If κ is too high, check Σ_crit calculation")
print("If T is too high, check pressure unit conversion")
#!/usr/bin/env python3
"""
Diagnostic: Find the 180x Einstein Radius Bug
=============================================

MACS0416 observed: θ_E = 35.0 arcsec
MACS0416 predicted: θ_E = 0.19 arcsec
Factor: ~184x too small

This script tests each component of the calculation to isolate the bug.
"""
import numpy as np
import math

# Constants
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1
c_km_s = 299792.458  # km/s
H0 = 70.0  # km/s/Mpc
Omega_m = 0.3
Omega_L = 0.7
Mpc_to_kpc = 1000.0

# MACS0416 parameters
z_lens = 0.396
z_source = 2.0
theta_E_observed_arcsec = 35.0

print("="*70)
print("DIAGNOSTIC: Einstein Radius 180x Bug")
print("="*70)
print(f"\nMACS0416:")
print(f"  z_lens = {z_lens}")
print(f"  z_source = {z_source}")
print(f"  θ_E (observed) = {theta_E_observed_arcsec} arcsec")
print()

# Step 1: Angular diameter distances
def Ez(z):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

def comoving_distance_Mpc(z, n=2048):
    if z <= 0:
        return 0.0
    zs = np.linspace(0.0, z, n)
    integrand = 1.0 / Ez(zs)
    Dc = (c_km_s / H0) * np.trapz(integrand, zs)
    return Dc

def angular_diameter_distance_kpc(z):
    Dc = comoving_distance_Mpc(z)
    Da = Dc / (1 + z)
    return Da * Mpc_to_kpc

D_d_kpc = angular_diameter_distance_kpc(z_lens)
D_s_kpc = angular_diameter_distance_kpc(z_source)

# D_ds via comoving distances
Dc_d = comoving_distance_Mpc(z_lens) * Mpc_to_kpc
Dc_s = comoving_distance_Mpc(z_source) * Mpc_to_kpc
D_ds_kpc = (Dc_s - Dc_d) / (1 + z_source)

print("Step 1: Angular Diameter Distances")
print(f"  D_d  = {D_d_kpc:.2f} kpc")
print(f"  D_s  = {D_s_kpc:.2f} kpc")
print(f"  D_ds = {D_ds_kpc:.2f} kpc")
print()

# Step 2: Critical surface density
# Formula: Σ_crit = (c²/4πG) × (D_s / D_d D_ds)
Sigma_crit_formula = (c_km_s**2) / (4 * math.pi * G) * (D_s_kpc / (D_d_kpc * D_ds_kpc))

print("Step 2: Critical Surface Density")
print(f"  Σ_crit = (c²/4πG) × (D_s / D_d D_ds)")
print(f"  Σ_crit = {Sigma_crit_formula:.2e} M☉/kpc²")
print()

# Convert to different units for sanity check
Sigma_crit_Msun_pc2 = Sigma_crit_formula / 1e6
Sigma_crit_g_cm2 = Sigma_crit_Msun_pc2 * (1.989e33 / (3.086e18)**2)

print(f"  Σ_crit = {Sigma_crit_Msun_pc2:.2e} M☉/pc²")
print(f"  Σ_crit = {Sigma_crit_g_cm2:.2e} g/cm²")
print()

# Step 3: Expected Einstein radius in kpc from observed arcsec
theta_E_rad = theta_E_observed_arcsec / 3600.0 * (math.pi / 180.0)
R_E_expected_kpc = theta_E_rad * D_d_kpc

print("Step 3: Expected Einstein Radius")
print(f"  θ_E (observed) = {theta_E_observed_arcsec} arcsec = {theta_E_rad:.6e} radians")
print(f"  R_E (expected) = θ_E × D_d = {R_E_expected_kpc:.2f} kpc")
print()

# Step 4: What surface density corresponds to κ=1 at R_E?
# κ = Σ / Σ_crit = 1  =>  Σ = Σ_crit
Sigma_needed_Msun_kpc2 = Sigma_crit_formula
Sigma_needed_Msun_pc2 = Sigma_needed_Msun_kpc2 / 1e6

print("Step 4: Surface Density Needed for κ=1")
print(f"  At R_E = {R_E_expected_kpc:.2f} kpc:")
print(f"  Σ(R_E) = Σ_crit = {Sigma_needed_Msun_kpc2:.2e} M☉/kpc²")
print(f"  Σ(R_E) = {Sigma_needed_Msun_pc2:.2e} M☉/pc²")
print()

# Step 5: Check current code's calculation
print("Step 5: Current Code's Calculation")
print(f"  (From cluster_lensing_analysis_real_sigma.py line 101-109)")

def sigma_crit_Msun_per_kpc2(z_d, z_s):
    """Current implementation"""
    if z_s <= z_d:
        return np.inf
    Dd = angular_diameter_distance_kpc(z_d)
    Ds = angular_diameter_distance_kpc(z_s)
    Dc_s = comoving_distance_Mpc(z_s) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(z_d) * Mpc_to_kpc
    Dds_ang = (Dc_s - Dc_d) / (1 + z_s)
    return (c_km_s ** 2) / (4 * math.pi * G) * (Ds / (Dd * max(Dds_ang, 1e-12)))

Sigma_crit_current = sigma_crit_Msun_per_kpc2(z_lens, z_source)
print(f"  Σ_crit (current) = {Sigma_crit_current:.2e} M☉/kpc²")
print(f"  Match? {np.isclose(Sigma_crit_current, Sigma_crit_formula)}")
print()

# Step 6: Test with typical cluster surface density
print("Step 6: Typical Cluster Surface Density")
Sigma_typical_Msun_kpc2 = 1e9  # Typical for cluster core
Sigma_typical_Msun_pc2 = Sigma_typical_Msun_kpc2 / 1e6

print(f"  Typical core: Σ ~ {Sigma_typical_Msun_kpc2:.2e} M☉/kpc²")
print(f"               Σ ~ {Sigma_typical_Msun_pc2:.2e} M☉/pc²")

kappa_typical = Sigma_typical_Msun_kpc2 / Sigma_crit_current
print(f"  κ = Σ / Σ_crit = {kappa_typical:.4f}")
print()

# Step 7: Where would κ=1 occur with typical baryon profile?
# For a cluster with M_baryon ~ 10^14 M☉ within 500 kpc
M_baryon_typical = 1e14  # M☉
R_typical = 500.0  # kpc
# Projected surface density: Σ ~ M / (π R²) (very rough)
Sigma_avg = M_baryon_typical / (math.pi * R_typical**2)

print("Step 7: Rough Estimate for Baryon-Only Einstein Radius")
print(f"  M_baryon ~ {M_baryon_typical:.2e} M☉ within {R_typical} kpc")
print(f"  Σ_avg ~ M / (π R²) ~ {Sigma_avg:.2e} M☉/kpc²")

# Radius where Σ(R) = Σ_crit (very rough, assuming power-law)
# Σ(R) ~ Σ_0 (R/R_0)^(-1), so R_E ~ R_0 (Σ_0/Σ_crit)
# With strong lensing, expect R_E ~ 20-100 kpc typically
R_E_rough = R_typical * (Sigma_avg / Sigma_crit_current)**(1/2)  # sqrt for projected
print(f"  R_E (rough) ~ {R_E_rough:.2f} kpc")

theta_E_rough_rad = R_E_rough / D_d_kpc
theta_E_rough_arcsec = theta_E_rough_rad * (180/math.pi) * 3600

print(f"  θ_E (rough) ~ {theta_E_rough_arcsec:.2f} arcsec")
print()

# Step 8: THE BUG TEST - Check if our prediction is off by 180x
print("="*70)
print("DIAGNOSIS: Where is the 180x error?")
print("="*70)

# Our code reports θ_E ~ 0.19 arcsec
theta_E_predicted = 0.19  # arcsec (from actual output)
R_E_predicted_kpc = theta_E_predicted / 3600.0 * (math.pi/180.0) * D_d_kpc

print(f"\nOur prediction: θ_E = {theta_E_predicted} arcsec")
print(f"                R_E = {R_E_predicted_kpc:.4f} kpc")
print()

# Check: At what radius did we find κ=1?
print("If θ_E = 0.19 arcsec, that corresponds to:")
print(f"  R = {R_E_predicted_kpc:.4f} kpc")
print()
print("This is WAY too small! A cluster's baryon core is >>1 kpc.")
print()

# HYPOTHESIS 1: Unit error in R → θ conversion
print("HYPOTHESIS 1: Unit error in R → θ conversion")
print(f"  Correct: θ = R / D_d")
print(f"           θ = {R_E_expected_kpc:.2f} kpc / {D_d_kpc:.2f} kpc = {theta_E_rad:.6e} rad")
print(f"           θ = {theta_E_rad * 180/math.pi * 3600:.2f} arcsec ✓")
print()

# HYPOTHESIS 2: Σ_crit off by factor?
print("HYPOTHESIS 2: Σ_crit calculation error")
ratio_crit = Sigma_crit_current / Sigma_needed_Msun_kpc2
print(f"  Σ_crit (current) / Σ_crit (expected) = {ratio_crit:.6f}")
if abs(ratio_crit - 1.0) < 0.01:
    print("  ✓ Σ_crit looks correct")
else:
    print(f"  ❌ Σ_crit is off by factor {ratio_crit:.2f}")
print()

# HYPOTHESIS 3: κ calculation uses wrong Σ?
print("HYPOTHESIS 3: κ = Σ / Σ_crit uses wrong units")
print("  Check if Σ is in M☉/kpc² and Σ_crit is in M☉/kpc²")
print("  Current code: line 453-454")
print("    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)")
print("    kappa_bar = Sigma_bar / Sigma_crit")
print("  This SHOULD be correct if both in M☉/kpc²")
print()

# HYPOTHESIS 4: The slip factor S(R) is too small
print("HYPOTHESIS 4: Slip factor S(R) reduces κ by 180x")
print("  If S(R) ~ 1 + small correction, but it should be S(R) >> 1?")
print("  Check: g_tail / g_bar ratio at R_E")
print("  Currently: S(R) = g_total / g_bar ~ 1 + O(0.01)?")
print("  Needed: S(R) ~ 180 at R_E?")
print()

# HYPOTHESIS 5: Missing factor in deflection calculation
print("HYPOTHESIS 5: Missing factor in κ_mean → θ_E")
print("  Current: finds R where κ_mean(R) = 1")
print("  Then: θ_E = R / D_d × (rad → arcsec)")
print("  This is correct for convergence definition")
print()

# CONCLUSION
print("="*70)
print("LIKELY CULPRIT")
print("="*70)
print("""
The code finds R_E ~ 1 kpc where κ_mean = 1.

But observed θ_E = 35" → R_E ~ 186 kpc at z=0.396.

Ratio: 186 / 1 = 186x ≈ 180x error!

DIAGNOSIS: The slip factor S(R) is TOO WEAK!

The G³ tail is not amplifying the lensing enough. We need:
  κ_eff = S(R) × κ_GR

Where S(R) needs to reach ~180 at R ~ 200 kpc for MACS0416.

Currently, the universal parameters give S_∞ ~ 1 + small correction.

FIX: Either:
  1. S_∞ amplitude needs to be >>1 (not just 1 + correction)
  2. Or the entire slip factor normalization is wrong
  3. Or there's a missing factor in how we apply the slip

CHECK: Look at g_tail / g_bar at R ~ 100-200 kpc.
       Should be ~180, but is probably ~0.01.
""")
print()

# Final diagnostic: What S_∞ do we need?
print("REQUIRED SLIP FACTOR:")
print(f"  At R_E = {R_E_expected_kpc:.2f} kpc:")
print(f"  S_needed = θ_E(obs) / θ_E(pred) = {theta_E_observed_arcsec / theta_E_predicted:.1f}")
print()
print("This means the slip factor S(R) must amplify by ~180x,")
print("NOT just S(R) ~ 1 + small correction!")
print("="*70)

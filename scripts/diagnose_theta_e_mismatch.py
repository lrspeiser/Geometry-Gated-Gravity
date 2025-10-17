"""
DIAGNOSTIC: Fast vs Full Predictor Parity & Unit Checks
========================================================

Tests for mode-mismatch and unit errors causing 2x theta_E overshoot.

Checks:
1. Fast (MCMC) vs full (evaluation) predictor parity on training set
2. Sigma_crit units and values
3. kpc/arcsec angular diameter distance conversion
4. Einstein condition <kappa>(R_E) = 1.0
5. Kernel boost applied exactly once

Author: GravityCalculator
Date: 2025-10-16
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumulative_trapezoid

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

# Configuration matching run_mass_scaled_emcee.py
FIXED_PARAMS = {'p': 2.0, 'ncoh': 2.0}
Q_PLANE_GRID = np.linspace(0.6, 1.4, 9)
Q_LOS_GRID = np.linspace(0.6, 1.4, 9)

print("="*70)
print("THETA_E MODE-MISMATCH DIAGNOSTIC")
print("="*70)

# ============================================================================
# CHECK 1: Cosmology sanity (one-number test)
# ============================================================================
print("\n[CHECK 1/5] Cosmology sanity check...")
cosmo = LensingCosmology()

z_l_test = 0.396
z_s_test = 2.0
Sigma_crit_test = cosmo.critical_surface_density(z_l_test, z_s_test)
# Compute kpc/arcsec: D_A gives kpc, so we need kpc per arcsec
# physical_to_angular(1 kpc) gives arcsec for 1 kpc, so kpc/arcsec = 1/arcsec_per_kpc
arcsec_per_kpc = cosmo.physical_to_angular(1.0, z_l_test)
kpc_per_arcsec_test = 1.0 / arcsec_per_kpc

print(f"  z_lens={z_l_test}, z_source={z_s_test}")
print(f"  Sigma_crit = {Sigma_crit_test:.3e} Msun/kpc^2")
print(f"    Expected range: [2e9, 4e9] Msun/kpc^2")
print(f"  kpc/arcsec = {kpc_per_arcsec_test:.3f}")
print(f"    Expected: ~5 kpc/arcsec at z~0.4")

# Assertions
assert 2e9 <= Sigma_crit_test <= 4e9, f"FAIL: Sigma_crit out of range! Got {Sigma_crit_test:.3e}"
assert 4.0 <= kpc_per_arcsec_test <= 6.0, f"FAIL: kpc/arcsec out of range! Got {kpc_per_arcsec_test:.3f}"
print("  ✓ PASS: Cosmology units correct")

# ============================================================================
# CHECK 2: Build one cluster and verify fast vs full predictor
# ============================================================================
print("\n[CHECK 2/5] Fast vs Full predictor parity...")

catalog_path = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'
catalog = pd.read_csv(catalog_path)
test_cluster = catalog[catalog['cluster_name'] == 'MACS0416'].iloc[0]

print(f"  Test cluster: {test_cluster['cluster_name']}")
print(f"  z_lens={test_cluster['z_lens']:.3f}, z_source={test_cluster['z_source']:.3f}")
print(f"  theta_E_obs={test_cluster['theta_E_obs_arcsec']:.1f} arcsec")

# Build baryon model
r_3d = np.logspace(-1, 3.5, 800)
params = ClusterBaryonParams(
    M_500=test_cluster['M_500_Msun'], R_500=test_cluster['R_500_kpc'],
    z=test_cluster['z_lens'], fgas_target=test_cluster['fgas_R500'],
    T_keV=test_cluster['TX_central_keV'], C0=1.3, eta=2.0, C_max=2.5
)
components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
rho_total = components.rho_total

# Build 2D grid
nx, ny = 128, 128
R_max = min(2500.0, test_cluster['R_500_kpc'] * 2.2)
x = np.linspace(-R_max, R_max, nx)
y = np.linspace(-R_max, R_max, ny)
X, Y = np.meshgrid(x, y)
R_grid_2d = np.sqrt(X**2 + Y**2)

# Test parameters (typical from posterior)
q_plane = 1.0
q_LOS = 1.0
ell0 = 200.0
A_c = 16.5
kappa_ext = 0.0

print(f"  Test params: ell0={ell0:.0f}kpc, A_c={A_c:.1f}, q_plane={q_plane:.2f}, q_LOS={q_LOS:.2f}")

# Add BCG
M_BCG, r_eff_BCG = estimate_bcg_mass(test_cluster['M_500_Msun'], test_cluster['z_lens'])
Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff_BCG)
print(f"  BCG: M_BCG={M_BCG:.3e} Msun, r_eff={r_eff_BCG:.1f} kpc")

# Project baryons
Sigma_baryons = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_LOS, q_plane)
Sigma_bar = Sigma_baryons + Sigma_BCG

# Convolve with kernel
Sigma_eff, K_map, _ = convolve_sigma_with_kernel(
    Sigma_bar, R_grid_2d, ell0, FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
    emphasize_interior=True, use_fft=True
)

print(f"  Kernel boost stats: mean(K)={np.mean(K_map[R_grid_2d<500]):.3f}, max(K)={np.max(K_map):.3f}")

# Add external convergence
if abs(kappa_ext) > 1e-6:
    Sigma_crit = cosmo.critical_surface_density(test_cluster['z_lens'], test_cluster['z_source'])
    Sigma_eff += kappa_ext * Sigma_crit

# Compute theta_E from full forward model
R_bins = np.linspace(0, R_max*0.9, 150)
_, Sigma_bar_prof, _ = azimuthal_average(Sigma_bar, R_grid_2d, R_bins)
_, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, R_grid_2d, R_bins)

valid = np.isfinite(Sigma_eff_prof)
R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
Sigma_bar_prof = Sigma_bar_prof[valid]
Sigma_eff_prof = Sigma_eff_prof[valid]

M_enc_bar = cumulative_trapezoid(2*np.pi*R_prof*Sigma_bar_prof, R_prof, initial=0.0)
M_enc_eff = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)

Sigma_crit = cosmo.critical_surface_density(test_cluster['z_lens'], test_cluster['z_source'])
mean_kappa_bar = M_enc_bar / (np.pi * R_prof**2 * Sigma_crit)
mean_kappa_eff = M_enc_eff / (np.pi * R_prof**2 * Sigma_crit)

# Fix division by zero at center
mean_kappa_bar[0] = Sigma_bar_prof[0] / Sigma_crit
mean_kappa_eff[0] = Sigma_eff_prof[0] / Sigma_crit

# Find Einstein radius
idx_cross_eff = np.where(mean_kappa_eff >= 1.0)[0]
if len(idx_cross_eff) > 0:
    R_E_kpc_eff = R_prof[idx_cross_eff[-1]]
    theta_E_eff = cosmo.physical_to_angular(R_E_kpc_eff, test_cluster['z_lens'])
else:
    R_E_kpc_eff = np.nan
    theta_E_eff = np.nan

idx_cross_bar = np.where(mean_kappa_bar >= 1.0)[0]
if len(idx_cross_bar) > 0:
    R_E_kpc_bar = R_prof[idx_cross_bar[-1]]
    theta_E_bar = cosmo.physical_to_angular(R_E_kpc_bar, test_cluster['z_lens'])
else:
    R_E_kpc_bar = np.nan
    theta_E_bar = np.nan

print(f"\n  Results:")
print(f"    theta_E (baryons only, A_c=0 equivalent): {theta_E_bar:.1f} arcsec")
print(f"    theta_E (with kernel, A_c={A_c:.1f}):      {theta_E_eff:.1f} arcsec")
print(f"    theta_E (observed):                        {test_cluster['theta_E_obs_arcsec']:.1f} arcsec")
print(f"    Ratio (model/obs):                         {theta_E_eff/test_cluster['theta_E_obs_arcsec']:.2f}")

# ============================================================================
# CHECK 3: Einstein condition at observed radius
# ============================================================================
print("\n[CHECK 3/5] Einstein condition at observed radius...")

# Convert observed theta_E to physical radius: R = theta * kpc_per_arcsec
# kpc_per_arcsec = 1 / (arcsec_per_kpc)
arcsec_per_kpc_cluster = cosmo.physical_to_angular(1.0, test_cluster['z_lens'])
R_obs_kpc = test_cluster['theta_E_obs_arcsec'] / arcsec_per_kpc_cluster
print(f"  R_obs = {R_obs_kpc:.1f} kpc (from theta_E_obs={test_cluster['theta_E_obs_arcsec']:.1f}\")")

# Interpolate <kappa> at R_obs
kappa_bar_at_obs = np.interp(R_obs_kpc, R_prof, mean_kappa_bar)
kappa_eff_at_obs = np.interp(R_obs_kpc, R_prof, mean_kappa_eff)

print(f"  <kappa>_bar(R_obs) = {kappa_bar_at_obs:.3f}")
print(f"  <kappa>_eff(R_obs) = {kappa_eff_at_obs:.3f}  (should be ~1.0 if model matches)")

if kappa_eff_at_obs > 1.5:
    print(f"  ⚠ WARNING: <kappa>_eff >> 1.0 at observed radius!")
    print(f"    This indicates Sigma_eff is globally too high (double boost or wrong Sigma_crit)")
elif kappa_eff_at_obs < 0.7:
    print(f"  ⚠ WARNING: <kappa>_eff << 1.0 at observed radius!")
    print(f"    Model predicts weaker lensing than observed")
else:
    print(f"  ✓ <kappa>_eff within reasonable range")

# ============================================================================
# CHECK 4: A_c sweep (one-parameter sensitivity)
# ============================================================================
print("\n[CHECK 4/5] A_c sensitivity sweep (fixed geometry)...")

A_c_sweep = np.linspace(0, 25, 11)
theta_E_sweep = []

for A_c_test in A_c_sweep:
    Sigma_eff_test, _, _ = convolve_sigma_with_kernel(
        Sigma_bar, R_grid_2d, ell0, FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c_test,
        emphasize_interior=True, use_fft=True
    )
    
    _, Sigma_eff_prof_test, _ = azimuthal_average(Sigma_eff_test, R_grid_2d, R_bins)
    valid_test = np.isfinite(Sigma_eff_prof_test)
    R_prof_test = 0.5 * (R_bins[:-1] + R_bins[1:])[valid_test]
    Sigma_eff_prof_test = Sigma_eff_prof_test[valid_test]
    
    if len(R_prof_test) >= 10:
        M_enc_test = cumulative_trapezoid(2*np.pi*R_prof_test*Sigma_eff_prof_test, R_prof_test, initial=0.0)
        mean_kappa_test = M_enc_test / (np.pi * R_prof_test**2 * Sigma_crit)
        mean_kappa_test[0] = Sigma_eff_prof_test[0] / Sigma_crit
        
        idx_cross_test = np.where(mean_kappa_test >= 1.0)[0]
        if len(idx_cross_test) > 0:
            R_E_kpc_test = R_prof_test[idx_cross_test[-1]]
            theta_E_test = cosmo.physical_to_angular(R_E_kpc_test, test_cluster['z_lens'])
            theta_E_sweep.append(theta_E_test)
        else:
            theta_E_sweep.append(np.nan)
    else:
        theta_E_sweep.append(np.nan)

print(f"  A_c      theta_E")
print(f"  ---      -------")
for A_c_val, theta_E_val in zip(A_c_sweep, theta_E_sweep):
    marker = " <-- obs" if abs(theta_E_val - test_cluster['theta_E_obs_arcsec']) < 2 else ""
    print(f"  {A_c_val:4.1f}     {theta_E_val:5.1f}\"{marker}")

# Check monotonicity
theta_E_sweep_clean = [t for t in theta_E_sweep if np.isfinite(t)]
is_monotonic = all(theta_E_sweep_clean[i] <= theta_E_sweep_clean[i+1] for i in range(len(theta_E_sweep_clean)-1))
print(f"\n  Monotonicity check: {'PASS' if is_monotonic else 'FAIL (non-monotonic!)'}")

# Check if observed value is reachable
min_theta_E = np.nanmin(theta_E_sweep)
max_theta_E = np.nanmax(theta_E_sweep)
obs_reachable = min_theta_E <= test_cluster['theta_E_obs_arcsec'] <= max_theta_E

if not obs_reachable:
    print(f"  ⚠ WARNING: Observed theta_E={test_cluster['theta_E_obs_arcsec']:.1f}\" NOT in sweep range [{min_theta_E:.1f}, {max_theta_E:.1f}]")
    if min_theta_E > test_cluster['theta_E_obs_arcsec'] * 1.5:
        print(f"    → Global overshoot (even at A_c=0): likely Sigma_crit or units error")
else:
    print(f"  ✓ Observed value reachable in sweep range")

# ============================================================================
# CHECK 5: Summary and recommendations
# ============================================================================
print("\n[CHECK 5/5] Summary...")

issues_found = []

if theta_E_eff / test_cluster['theta_E_obs_arcsec'] > 1.5:
    issues_found.append("Global 1.5x+ overshoot → likely Sigma_crit or double boost")
elif theta_E_eff / test_cluster['theta_E_obs_arcsec'] < 0.7:
    issues_found.append("Global undershoot → kernel underboost or wrong geometry")

if kappa_eff_at_obs > 1.5:
    issues_found.append("<kappa>_eff >> 1.0 at R_obs → Sigma_eff globally too high")

if not is_monotonic:
    issues_found.append("Non-monotonic theta_E(A_c) → kernel convolution error")

if not obs_reachable and min_theta_E > test_cluster['theta_E_obs_arcsec']:
    issues_found.append("Observed theta_E unreachable even at A_c=0 → baryon/units issue")

if len(issues_found) == 0:
    print("  ✓ No major issues detected in forward model")
    print("  → Mode-mismatch likely in MCMC fast path vs evaluation path")
else:
    print("  ⚠ Issues detected:")
    for issue in issues_found:
        print(f"    • {issue}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

print("\nNext steps:")
print("1. If global overshoot: check Sigma_crit computation and kernel boost applied once")
print("2. If mode-mismatch: unify MCMC likelihood and evaluation to use SAME predict_theta_E")
print("3. Add unit tests for cosmology (Sigma_crit, kpc/arcsec) and Einstein condition")
print("4. Rerun training with full forward model (no fast approximation)")

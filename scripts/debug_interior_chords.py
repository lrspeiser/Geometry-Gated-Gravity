#!/usr/bin/env python3
"""
Debug Interior Chord Contributions
===================================

Understand why interior chords aren't contributing properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.cluster_kernel_3d_shell import (
    Shell3DKernelParams,
    chord_length_through_sphere,
    coherence_damping,
    interior_contribution
)
from many_path_model.lensing_utilities import AbelProjection

# Uniform sphere
R_sphere = 300.0  # kpc
rho_0 = 1e6  # Msun/kpc³
R_test = 150.0  # Test at R_sphere/2

# Create grid
r_grid = np.linspace(0.1, R_sphere * 2, 1000)
rho_3d = np.zeros_like(r_grid)
rho_3d[r_grid < R_sphere] = rho_0

# Kernel params (from test 2)
params = Shell3DKernelParams(
    A_c=1.0,
    r_gate=5.0,
    n_gate=4,
    ell0=150.0,
    p_density=1.0,  # Linear!
    L1=1000.0,
    q_taper=2.0,
    w_interior=1.0,
    w_exterior=1.0,
    coherence_mode='power_law',
    n_coh=1.5
)

# Split into interior shells
mask_int = r_grid < R_test
r_int = r_grid[mask_int]
rho_int = rho_3d[mask_int]

print("=" * 70)
print("DEBUG: Interior Chord Contributions")
print("=" * 70)
print()
print(f"R_test = {R_test} kpc")
print(f"ell0 = {params.ell0} kpc")
print(f"p_density = {params.p_density}")
print(f"Number of interior shells: {len(r_int)}")
print()

# Compute baseline surface density
projector = AbelProjection()
Sigma_on_rgrid = projector.project_density_to_surface(r_grid, rho_3d, r_grid)
Sigma_baseline = np.interp(R_test, r_grid, Sigma_on_rgrid)

print(f"Sigma_baseline at R={R_test} kpc: {Sigma_baseline:.2e} Msun/kpc²")
print()

# Manually compute interior contribution with debug output
total_weight = 0.0

print("Sample of interior shells:")
print(f"{'i':<6} {'r [kpc]':<12} {'L_chord':<12} {'C_damp':<12} {'contrib':<15}")
print("─" * 70)

for i in range(len(r_int)):
    if i == 0:
        continue
    
    r_s = r_int[i]
    
    # Chord length
    L_chord = chord_length_through_sphere(R_test, r_s)
    
    if L_chord == 0:
        continue
    
    # Coherence damping
    C_damp = coherence_damping(L_chord, params.ell0, params.coherence_mode, params.n_coh)
    
    # Density weighting
    rho_weighted = rho_int[i]**params.p_density
    
    # Shell contribution
    dr = r_int[i] - r_int[i-1]
    shell_area = 4 * np.pi * r_s**2
    contrib = L_chord * C_damp * rho_weighted * shell_area * dr
    
    total_weight += contrib
    
    # Print sample
    if i % 50 == 0 or i < 10:
        print(f"{i:<6} {r_s:<12.2f} {L_chord:<12.2f} {C_damp:<12.4f} {contrib:<15.2e}")

print()
print(f"Total accumulated weight: {total_weight:.4e}")
print()

# Compute normalization
if params.p_density != 1.0:
    rho_ref = Sigma_baseline / R_test
    norm_factor = (rho_ref**params.p_density) * R_test**4
else:
    norm_factor = Sigma_baseline * R_test**3

print(f"Normalization factor: {norm_factor:.4e}")
print()

K_int = total_weight / norm_factor
print(f"K_interior = {K_int:.6f}")
print()

# Compare to function call
K_int_func = interior_contribution(R_test, r_int, rho_int, params, Sigma_baseline)
print(f"K_interior (from function): {K_int_func:.6f}")
print()

# Analyze coherence damping impact
print("=" * 70)
print("Coherence Damping Analysis")
print("=" * 70)
print()

L_chords = []
C_damps = []
for i in range(1, len(r_int)):
    r_s = r_int[i]
    L = chord_length_through_sphere(R_test, r_s)
    if L > 0:
        C = coherence_damping(L, params.ell0, params.coherence_mode, params.n_coh)
        L_chords.append(L)
        C_damps.append(C)

L_chords = np.array(L_chords)
C_damps = np.array(C_damps)

print(f"Chord lengths: min={L_chords.min():.1f}, max={L_chords.max():.1f}, mean={L_chords.mean():.1f} kpc")
print(f"Coherence factors: min={C_damps.min():.4f}, max={C_damps.max():.4f}, mean={C_damps.mean():.4f}")
print()

if C_damps.mean() < 0.01:
    print("⚠️  WARNING: Coherence damping is suppressing interior chords too much!")
    print(f"   Mean damping factor: {C_damps.mean():.6f} (~{C_damps.mean()*100:.2f}%)")
    print(f"   This is the reason interior contribution is zero!")
else:
    print(f"✓ Coherence damping looks reasonable: ~{C_damps.mean()*100:.1f}% on average")

print()
print("=" * 70)

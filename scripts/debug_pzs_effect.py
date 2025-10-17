"""Debug: check effective Sigma_crit with different P(z_s)"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from pathlib import Path
from many_path_model.lensing_utilities import LensingCosmology

cosmo = LensingCosmology()
z_l = 0.544  # MACS1149

# Test 1: Single median z_s = 2.0
Sigma_crit_single = cosmo.effective_critical_density_with_distribution(z_l, z_source_single=2.0)
print(f"Single z_s=2.0: Sigma_crit = {Sigma_crit_single:.3e} Msun/kpc^2")

# Test 2: Default lognormal
Sigma_crit_lognormal = cosmo.effective_critical_density_with_distribution(z_l)
print(f"Lognormal default: Sigma_crit = {Sigma_crit_lognormal:.3e} Msun/kpc^2")

# Test 3: My custom arc redshifts
config_path = Path(__file__).parent.parent / 'data' / 'clusters' / 'macs1149_config.json'
with open(config_path) as f:
    config = json.load(f)

arc_data = config['arc_redshifts']
z_grid = np.array(arc_data['redshifts'])
weights = np.array(arc_data['weights'])
P_z_s = weights / np.sum(weights)

print(f"\nCustom arc P(z_s):")
for z, w, p in zip(z_grid, weights, P_z_s):
    print(f"  z={z:.2f}, weight={w}, P={p:.3f}")

Sigma_crit_custom = cosmo.effective_critical_density_with_distribution(z_l, z_source_grid=z_grid, P_z_s=P_z_s)
print(f"\nCustom arcs: Sigma_crit = {Sigma_crit_custom:.3e} Msun/kpc^2")

# Compute theta_E scaling
ratio_single = Sigma_crit_single / Sigma_crit_custom
ratio_lognormal = Sigma_crit_lognormal / Sigma_crit_custom

print(f"\nSigma_crit ratios (custom / reference):")
print(f"  custom vs single(2.0): {1/ratio_single:.3f} → theta_E scales by ~{np.sqrt(1/ratio_single):.3f}")
print(f"  custom vs lognormal:   {1/ratio_lognormal:.3f} → theta_E scales by ~{np.sqrt(1/ratio_lognormal):.3f}")

# Check individual D_LS/D_S
print(f"\nLensing efficiency D_LS/D_S:")
for z_s in [1.5, 2.0, 2.5, 3.0, 3.5]:
    D_s = cosmo.angular_diameter_distance_kpc(z_s)
    D_ls = cosmo.angular_diameter_distance_between(z_l, z_s)
    efficiency = D_ls / D_s
    print(f"  z_s={z_s:.1f}: D_LS/D_S = {efficiency:.4f}")

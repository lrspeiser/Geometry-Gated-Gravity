#!/usr/bin/env python3
"""Quick test to verify deflection functions return reasonable values."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lensing_utils import alpha_fun_GR_baryons, alpha_fun_ACCEPTED, alpha_fun_GE

# Test MACS0416
cluster = 'macs0416'
local_name = 'MACSJ0416'
z_lens = 0.396
z_source = 2.0
team = 'cats'
version = 'v4.1'

print("Testing deflection functions for MACS0416...")
print(f"z_lens={z_lens}, z_source={z_source}")
print()

# Test accepted deflection
print("=== ACCEPTED (from HLSP maps) ===")
try:
    alpha_acc = alpha_fun_ACCEPTED(cluster, team, version, z_lens, z_source)
    if alpha_acc is not None:
        test_angles = [5.0, 10.0, 20.0, 30.0, 50.0]
        for theta in test_angles:
            alpha_val = alpha_acc(theta)
            print(f"  α({theta:5.1f}″) = {alpha_val:8.4f}″")
    else:
        print("  FAILED: returned None")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test GR baryons-only
print("=== GR (baryons only) ===")
try:
    alpha_gr = alpha_fun_GR_baryons(local_name, z_lens, z_source)
    if alpha_gr is not None:
        test_angles = [5.0, 10.0, 20.0, 30.0, 50.0]
        for theta in test_angles:
            alpha_val = alpha_gr(theta)
            print(f"  α({theta:5.1f}″) = {alpha_val:8.4f}″")
    else:
        print("  FAILED: returned None")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test GE custom
print("=== GE (custom with interior boost) ===")
try:
    alpha_ge = alpha_fun_GE(local_name, z_lens, z_source,
                           a=3.0, b=0.2, d=0.1,
                           gamma1=0.2, gamma2=0.1,
                           Rd_kpc=1000.0, R_scale_kpc=100.0,
                           beta_clip=(1.0, 5.0),
                           A_core=0.25, p_core=2.0,
                           Sigma0_hat=0.0, beta_core=0.4,
                           smooth_R_kpc=5.0)
    if alpha_ge is not None:
        test_angles = [5.0, 10.0, 20.0, 30.0, 50.0]
        for theta in test_angles:
            alpha_val = alpha_ge(theta)
            print(f"  α({theta:5.1f}″) = {alpha_val:8.4f}″")
    else:
        print("  FAILED: returned None")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("Done.")

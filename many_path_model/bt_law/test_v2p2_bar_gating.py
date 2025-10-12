#!/usr/bin/env python3
"""
Quick Test of V2.2 Bar Gating
=============================

Tests bar suppression on representative barred/unbarred galaxies
to verify the mechanism works as expected.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bt_laws import morph_to_bt
from bt_laws_v2p1 import eval_all_laws_v2p1
from bt_laws_v2p2 import eval_all_laws_v2p2, load_theta

print("="*80)
print("TESTING V2.2 BAR GATING")
print("="*80)

# Load data
with open('many_path_model/bt_law/sparc_disk_params.json', 'r') as f:
    disk_params = json.load(f)

with open('many_path_model/bt_law/sparc_shear_predictors.json', 'r') as f:
    shear_preds = json.load(f)

with open('many_path_model/bt_law/sparc_bar_classification.json', 'r') as f:
    bar_class = json.load(f)

theta = load_theta('many_path_model/bt_law/bt_law_params_v2p2_initial.json')

# Test on representative galaxies: barred vs unbarred
test_galaxies = [
    # Strongly barred (SB)
    ('ESO079-G014', 'Strongly barred SB'),
    ('NGC0891', 'Strongly barred SB edge-on'),
    ('NGC6946', 'Strongly barred SB'),
    
    # Weakly barred (SAB)
    ('NGC2976', 'Weakly barred SAB'),
    
    # Unbarred (SA/S)
    ('NGC2403', 'Unbarred Sc'),
    ('NGC3198', 'Unbarred Sc'),
    ('DDO154', 'Unbarred dwarf'),
]

print("\nTesting bar suppression on representative galaxies:")
print("-"*80)
print(f"{'Galaxy':15s} {'Bar':10s} {'g_bar':6s} {'λ_v2.1':8s} {'λ_v2.2':8s} {'Δλ':8s} {'ring_v2.1':10s} {'ring_v2.2':10s}")
print("-"*80)

for gal_name, description in test_galaxies:
    # Get galaxy data
    gal_disk = disk_params.get(gal_name, {})
    hubble_type = gal_disk.get('hubble_type', 'Unknown')
    type_group = gal_disk.get('type_group', 'unknown')
    R_d = gal_disk.get('R_d_kpc')
    Sigma0 = gal_disk.get('Sigma0')
    
    gal_shear = shear_preds.get(gal_name, {})
    shear = gal_shear.get('shear_2p2Rd')
    
    gal_bar = bar_class.get(gal_name, {})
    bar_cls = gal_bar.get('bar_class', 'unknown')
    g_bar = gal_bar.get('g_bar', 0.8)
    
    B_T = morph_to_bt(hubble_type, type_group)
    
    # Evaluate with V2.1 (no bar gating)
    params_v2p1 = eval_all_laws_v2p1(B_T, theta, Sigma0=Sigma0, R_d=R_d, shear=shear)
    
    # Evaluate with V2.2 (with bar gating)
    params_v2p2 = eval_all_laws_v2p2(B_T, theta, Sigma0=Sigma0, R_d=R_d, shear=shear, g_bar_in=g_bar)
    
    lambda_v2p1 = params_v2p1['lambda_ring']
    lambda_v2p2 = params_v2p2['lambda_ring']
    delta_lambda = lambda_v2p2 - lambda_v2p1
    
    ring_v2p1 = params_v2p1['ring_amp']
    ring_v2p2 = params_v2p2['ring_amp']
    
    print(f"{gal_name:15s} {bar_cls:10s} {g_bar:6.2f} {lambda_v2p1:8.2f} {lambda_v2p2:8.2f} {delta_lambda:+8.2f} {ring_v2p1:10.3f} {ring_v2p2:10.3f}")

print("-"*80)
print("\nExpected behavior:")
print("  - SB (strongly barred, g_bar~0.45): λ and ring_amp reduced by ~55%")
print("  - SAB (weakly barred, g_bar~0.75): λ and ring_amp reduced by ~25%")
print("  - SA/S (unbarred, g_bar~0.9-1.0): λ and ring_amp nearly unchanged")
print("\n✓ Bar gating working as expected if Δλ negative for barred systems")
print("="*80)

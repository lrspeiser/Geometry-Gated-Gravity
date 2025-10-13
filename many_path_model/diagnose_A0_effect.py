"""
Diagnostic: Show direct effect of A_0 on model predictions

This tests whether A_0 actually scales the boost factor as intended,
and shows the effect on model vs observation comparison WITHOUT fitting g†.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from validation_suite import ValidationSuite

# Physical constants
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19

def test_A0_on_single_galaxy(df: pd.DataFrame, A_0_values: list):
    """Test A_0 effect on a single galaxy's rotation curve"""
    
    # Pick a typical galaxy
    galaxy = df.iloc[50]  # Mid-range galaxy
    
    r_all = galaxy['r_all']
    v_obs = galaxy['v_all']
    
    print(f"Testing galaxy: {galaxy['Galaxy']}")
    print(f"  Radii: {r_all[:5]} ... {r_all[-3:]} kpc")
    print(f"  V_obs: {v_obs[:5]} ... {v_obs[-3:]} km/s")
    
    # Get baryonic components
    v_disk = galaxy.get('v_disk_all', np.zeros_like(v_obs))
    v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_obs))
    v_gas = galaxy.get('v_gas_all', np.zeros_like(v_obs))
    
    # Compute g_bar
    v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
    v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
    r_m = r_all * KPC_TO_M
    g_bar = v_baryonic_m_s**2 / r_m
    
    # Compute g_obs
    v_obs_m_s = v_obs * KM_TO_M
    g_obs = v_obs_m_s**2 / r_m
    
    print(f"\nTesting different A_0 values:")
    print("-" * 80)
    print(f"{'A_0':<8} {'K[0]':<10} {'K[5]':<10} {'K[-1]':<10} {'g_model[5]/g_obs[5]':<20}")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for A_0 in A_0_values:
        hp = PathSpectrumHyperparams(
            L_0=1.82, 
            beta_bulge=1.09, 
            alpha_shear=0.056, 
            gamma_bar=1.06,
            A_0=A_0
        )
        kernel = PathSpectrumKernel(hp, use_cupy=False)
        
        # Compute boost
        BT = galaxy.get('BT', 0.0)
        bar_strength = galaxy.get('bar_strength', 0.0)
        K = kernel.many_path_boost_factor(r=r_all, v_circ=v_obs, BT=BT, bar_strength=bar_strength)
        
        # Model prediction
        g_model = g_bar * (1.0 + K)
        
        ratio = g_model[5] / g_obs[5] if len(g_obs) > 5 else 0
        
        print(f"{A_0:<8.2f} {K[0]:<10.4f} {K[5]:<10.4f} {K[-1]:<10.4f} {ratio:<20.4f}")
        
        # Plot rotation curves
        ax = axes[0]
        v_model = np.sqrt(g_model * r_m) / KM_TO_M  # Back to km/s
        ax.plot(r_all, v_model, label=f'A_0={A_0:.2f}', linewidth=2)
    
    ax = axes[0]
    ax.plot(r_all, v_obs, 'k--', linewidth=3, label='Observed', zorder=10)
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_title(f'{galaxy["Galaxy"]} - Rotation Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot boost factors
    ax = axes[1]
    for A_0 in A_0_values:
        hp = PathSpectrumHyperparams(L_0=1.82, beta_bulge=1.09, alpha_shear=0.056, gamma_bar=1.06, A_0=A_0)
        kernel = PathSpectrumKernel(hp, use_cupy=False)
        K = kernel.many_path_boost_factor(r=r_all, v_circ=v_obs, BT=BT, bar_strength=bar_strength)
        ax.plot(r_all, K, label=f'A_0={A_0:.2f}', linewidth=2)
    
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('Boost Factor K', fontsize=12)
    ax.set_title('Boost Factor K(r) vs A_0', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot g_model/g_obs ratio
    ax = axes[2]
    for A_0 in A_0_values:
        hp = PathSpectrumHyperparams(L_0=1.82, beta_bulge=1.09, alpha_shear=0.056, gamma_bar=1.06, A_0=A_0)
        kernel = PathSpectrumKernel(hp, use_cupy=False)
        K = kernel.many_path_boost_factor(r=r_all, v_circ=v_obs, BT=BT, bar_strength=bar_strength)
        g_model = g_bar * (1.0 + K)
        ratio = g_model / g_obs
        ax.plot(r_all, ratio, label=f'A_0={A_0:.2f}', linewidth=2)
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect match')
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('g_model / g_obs', fontsize=12)
    ax.set_title('Model/Observation Ratio', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 2])
    
    plt.tight_layout()
    output_path = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results/A0_direct_effect.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved diagnostic plot to {output_path}")

def main():
    print("="*80)
    print("DIAGNOSTIC: A_0 DIRECT EFFECT ON MODEL PREDICTIONS")
    print("="*80)
    print("\nThis tests whether A_0 scales the boost factor K as intended.\n")
    
    # Load SPARC data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Test A_0 values
    A_0_values = [0.3, 0.5, 1.0, 2.0]
    
    test_A0_on_single_galaxy(df, A_0_values)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. If K scales linearly with A_0, the parameter works correctly")
    print("2. If g_model/g_obs ratio changes with A_0, we can tune to match observations")
    print("3. RAR fitting is scale-invariant, so we need to compare to FIXED g† = 1.2e-10")

if __name__ == "__main__":
    main()

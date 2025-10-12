#!/usr/bin/env python3
"""
Visualize the fitted B/T laws showing parameter predictions across morphology.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from bt_laws import load_theta, law_value, morph_to_bt, MORPH_TO_BT

def main():
    # Load fitted laws
    theta_file = Path("many_path_model/bt_law/bt_law_params.json")
    theta = load_theta(theta_file)
    
    print(f"Loaded B/T laws from: {theta_file}\n")
    print("Law parameters:")
    for param, law in theta.items():
        print(f"  {param:15s}: lo={law['lo']:7.4f}, hi={law['hi']:7.4f}, γ={law['gamma']:5.2f}")
    
    # Create B/T range
    B_T = np.linspace(0, 0.7, 300)
    
    # Evaluate laws
    params = {}
    for param_name in ['eta', 'ring_amp', 'M_max', 'lambda_ring']:
        law = theta[param_name]
        params[param_name] = law_value(B_T, law['lo'], law['hi'], law['gamma'])
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()
    
    param_names = ['eta', 'ring_amp', 'M_max', 'lambda_ring']
    param_labels = [r'$\eta$ (overall amplitude)', 
                    r'ring\_amp (spiral strength)', 
                    r'$M_{\mathrm{max}}$ (saturation)',
                    r'$\lambda_{\mathrm{ring}}$ (coherence, kpc)']
    
    # Plot each law
    for ax, pname, plabel in zip(axs, param_names, param_labels):
        y = params[pname]
        
        # Main curve
        ax.plot(B_T, y, 'b-', linewidth=2.5, label='B/T law')
        ax.fill_between(B_T, y, alpha=0.2)
        
        # Mark morphological types
        morph_types = ['Sd', 'Scd', 'Sc', 'Sbc', 'Sb', 'Sa']
        morph_colors = plt.cm.viridis(np.linspace(0, 1, len(morph_types)))
        
        for mtype, color in zip(morph_types, morph_colors):
            bt = MORPH_TO_BT.get(mtype, None)
            if bt is not None and bt <= 0.7:
                law = theta[pname]
                val = law_value(bt, law['lo'], law['hi'], law['gamma'])
                ax.axvline(bt, color=color, linestyle='--', alpha=0.3, linewidth=1)
                ax.plot(bt, val, 'o', color=color, markersize=8, 
                       label=f'{mtype} (B/T={bt:.2f})', zorder=5)
        
        # Styling
        ax.set_xlabel('Bulge-to-Total (B/T)', fontsize=11)
        ax.set_ylabel(plabel, fontsize=11)
        ax.grid(alpha=0.3, linestyle=':')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.set_xlim(0, 0.7)
        
        # Add law equation
        law = theta[pname]
        eq_text = f"$y = {law['lo']:.2f} + ({law['hi']:.2f} - {law['lo']:.2f}) \\times (1-B/T)^{{{law['gamma']:.2f}}}$"
        ax.text(0.98, 0.02, eq_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle('B/T Laws for Many-Path Gravity Parameters\n' + 
                 'Higher B/T (more bulge) → Lower enhancement',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    out_file = Path("many_path_model/bt_law/bt_law_visualization.png")
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {out_file}")
    
    # Also create a morphology table
    print("\nPredicted parameters by morphology:")
    print("-" * 90)
    print(f"{'Type':8s} {'B/T':6s} {'η':8s} {'ring_amp':10s} {'M_max':8s} {'λ_ring':10s}")
    print("-" * 90)
    
    for mtype in ['Im', 'Sd', 'Scd', 'Sc', 'Sbc', 'Sb', 'Sa', 'S0']:
        bt = MORPH_TO_BT.get(mtype, None)
        if bt is not None:
            eta_val = law_value(bt, **{k: v for k, v in theta['eta'].items() if k != 'loss'})
            ring_val = law_value(bt, **{k: v for k, v in theta['ring_amp'].items() if k != 'loss'})
            mmax_val = law_value(bt, **{k: v for k, v in theta['M_max'].items() if k != 'loss'})
            lam_val = law_value(bt, **{k: v for k, v in theta['lambda_ring'].items() if k != 'loss'})
            
            print(f"{mtype:8s} {bt:6.2f} {eta_val:8.4f} {ring_val:10.4f} {mmax_val:8.4f} {lam_val:10.2f}")
    print("-" * 90)

if __name__ == '__main__':
    main()

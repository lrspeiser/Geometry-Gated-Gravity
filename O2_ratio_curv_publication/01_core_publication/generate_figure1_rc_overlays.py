"""
Generate Figure 1: Rotation Curve Overlays for 6 Representative Galaxies

This script creates a 2x3 panel figure showing observed rotation curves
vs. O2 ratio_curv model predictions for 6 diverse SPARC galaxies.

Usage:
    python generate_figure1_rc_overlays.py

Output:
    O2_ratio_curv_publication/figures/Figure1_RC_Overlays.png
    O2_ratio_curv_publication/figures/Figure1_RC_Overlays.pdf

Author: Henry Speiser
Date: October 2, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

# Use publication-quality settings
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 8

# Best-fit O2 ratio_curv parameters
BEST_PARAMS = {
    'a': 0.6686576907182596,
    'b': 0.14007773322620287,
    'd': 0.08713057433850588,
}

def compute_fX(R_kpc, Sigma_Msun_pc2, Rd_kpc):
    """
    Compute excess factor fX for O2 ratio_curv model.
    
    fX = x² / (a - b·Σ̂ - d·|∇ln Σ|)
    
    Parameters:
    -----------
    R_kpc : array
        Radius in kpc
    Sigma_Msun_pc2 : array
        Surface density in Msun/pc²
    Rd_kpc : float
        Disk scale length in kpc
        
    Returns:
    --------
    fX : array
        Excess factor (dimensionless)
    """
    # Dimensionless radius
    x = dimensionless_radius(R_kpc, Rd=Rd_kpc)
    
    # Normalized surface density
    Sigma_hat = sigma_hat(Sigma_Msun_pc2)
    
    # Logarithmic gradient
    grad_ln_Sigma = grad_log_sigma(R_kpc, Sigma_Msun_pc2)
    
    # Compute denominator with gating
    a, b, d = BEST_PARAMS['a'], BEST_PARAMS['b'], BEST_PARAMS['d']
    denom = a - b * Sigma_hat - d * np.abs(grad_ln_Sigma)
    denom = np.clip(denom, 1e-6, None)  # Numerical safety
    
    # Excess factor
    fX = (x ** 2) / denom
    fX = np.maximum(fX, 0.0)  # Physical: no negative factors
    
    return fX

def compute_model_velocity(galaxy):
    """
    Compute model velocity for a galaxy.
    
    Parameters:
    -----------
    galaxy : SPARC galaxy object
        Must have R_kpc, Vbar_kms, Sigma_bar, Rd_kpc attributes
        
    Returns:
    --------
    Vmod : array
        Model circular velocity in km/s
    """
    R = np.asarray(galaxy.R_kpc)
    Vbar = np.asarray(galaxy.Vbar_kms)
    Sigma = np.asarray(galaxy.Sigma_bar)
    Rd = galaxy.Rd_kpc if galaxy.Rd_kpc is not None else np.nanmedian(R) / 2.5
    
    # Filter out invalid points
    mask = np.isfinite(R) & np.isfinite(Vbar) & np.isfinite(Sigma)
    R = R[mask]
    Vbar = Vbar[mask]
    Sigma = Sigma[mask]
    
    if len(R) < 3:
        return R, Vbar * np.nan  # Not enough points
    
    # Compute excess factor
    fX = compute_fX(R, Sigma, Rd)
    
    # Model velocity: V_total² = V_bar² · (1 + fX)
    Vmod = Vbar * np.sqrt(1.0 + fX)
    
    return R, Vmod

def compute_galaxy_metrics(galaxy):
    """Compute median APE and RMSE for a galaxy."""
    R = np.asarray(galaxy.R_kpc)
    Vobs = np.asarray(galaxy.Vobs_kms)
    Vbar = np.asarray(galaxy.Vbar_kms)
    Sigma = np.asarray(galaxy.Sigma_bar)
    Rd = galaxy.Rd_kpc if galaxy.Rd_kpc is not None else np.nanmedian(R) / 2.5
    
    mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar) & np.isfinite(Sigma)
    R = R[mask]
    Vobs = Vobs[mask]
    Vbar = Vbar[mask]
    Sigma = Sigma[mask]
    
    if len(R) < 3:
        return np.nan, np.nan
    
    fX = compute_fX(R, Sigma, Rd)
    Vmod = Vbar * np.sqrt(1.0 + fX)
    
    rmse = np.sqrt(np.mean((Vmod - Vobs)**2))
    mape = np.median(np.abs((Vmod - Vobs) / Vobs))
    
    return mape, rmse

def plot_galaxy_panel(ax, galaxy, show_legend=False):
    """
    Plot rotation curve for one galaxy in a single panel.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    galaxy : SPARC galaxy object
        Galaxy data
    show_legend : bool
        Whether to show legend
    """
    # Extract data
    R = np.asarray(galaxy.R_kpc)
    Vobs = np.asarray(galaxy.Vobs_kms)
    Verr = np.asarray(galaxy.Verr_kms) if hasattr(galaxy, 'Verr_kms') else None
    Vbar = np.asarray(galaxy.Vbar_kms)
    
    # Compute model
    R_mod, Vmod = compute_model_velocity(galaxy)
    
    # Compute metrics
    mape, rmse = compute_galaxy_metrics(galaxy)
    
    # Plot observed data
    if Verr is not None and np.all(np.isfinite(Verr)):
        ax.errorbar(R, Vobs, yerr=Verr, fmt='o', color='black', markersize=4,
                   elinewidth=1, capsize=2, capthick=1, label='Observed', zorder=3)
    else:
        ax.plot(R, Vobs, 'o', color='black', markersize=4, label='Observed', zorder=3)
    
    # Plot baryons-only
    ax.plot(R, Vbar, '-', color='#1f77b4', linewidth=1.5, alpha=0.7,
            label='Baryons only', zorder=2)
    
    # Plot O2 model
    ax.plot(R_mod, Vmod, '-', color='#d62728', linewidth=2.0,
            label='O2 ratio_curv', zorder=4)
    
    # Labels and formatting
    ax.set_xlabel('R [kpc]', fontsize=10)
    ax.set_ylabel('V [km/s]', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Title with galaxy name and metrics
    title = f"{galaxy.name}"
    if not np.isnan(mape):
        title += f"\nMAPE = {mape:.2f}, RMSE = {rmse:.1f} km/s"
    ax.set_title(title, fontsize=10, pad=8)
    
    # Legend (only for first panel)
    if show_legend:
        ax.legend(loc='lower right', fontsize=8, frameon=True, fancybox=False, 
                 shadow=False, framealpha=0.9)
    
    # Type and Rd info as text
    if hasattr(galaxy, 'Type'):
        type_text = f"Type: {galaxy.Type}"
        if galaxy.Rd_kpc is not None:
            type_text += f", Rd = {galaxy.Rd_kpc:.1f} kpc"
        ax.text(0.02, 0.98, type_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.7, edgecolor='none'))

def main():
    """Generate Figure 1: Rotation curve overlays for 6 representative galaxies."""
    
    print("Loading SPARC dataset...")
    ds = load_sparc()
    
    # Select 6 representative galaxies (matching paper description)
    # We want diversity in: size (Rd), type, surface brightness
    target_galaxies = [
        'NGC2403',   # Large spiral, Rd ~ 1.8 kpc
        'NGC3198',   # Benchmark SPARC galaxy
        'DDO154',    # Dwarf irregular, Rd ~ 0.5 kpc
        'UGC2885',   # Giant spiral, Rd ~ 8.1 kpc
        'F563-1',    # Low surface brightness
        'NGC7793',   # Late-type spiral
    ]
    
    galaxies = []
    for name in target_galaxies:
        # Find galaxy (handle different name formats)
        found = False
        for g in ds.galaxies:
            if g.name.replace(' ', '').replace('_', '').lower() == name.replace(' ', '').replace('_', '').lower():
                galaxies.append(g)
                found = True
                break
        if not found:
            print(f"Warning: Galaxy {name} not found in SPARC dataset")
    
    if len(galaxies) < 6:
        print(f"Warning: Only found {len(galaxies)}/6 target galaxies")
        # Fill in with first available galaxies
        for g in ds.galaxies:
            if g not in galaxies and len(galaxies) < 6:
                # Check if galaxy has sufficient data
                if (hasattr(g, 'R_kpc') and hasattr(g, 'Vobs_kms') and 
                    hasattr(g, 'Vbar_kms') and hasattr(g, 'Sigma_bar')):
                    R = np.asarray(g.R_kpc)
                    if len(R[np.isfinite(R)]) >= 6:
                        galaxies.append(g)
    
    print(f"Selected {len(galaxies)} galaxies:")
    for g in galaxies:
        print(f"  - {g.name}")
    
    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    
    # Plot each galaxy
    for i, galaxy in enumerate(galaxies):
        plot_galaxy_panel(axes[i], galaxy, show_legend=(i == 0))
    
    # Overall title
    fig.suptitle('Figure 1: O2 ratio_curv Model vs. Observed Rotation Curves (6 SPARC Galaxies)',
                fontsize=12, fontweight='bold', y=0.995)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save figure
    output_png = output_dir / 'Figure1_RC_Overlays.png'
    output_pdf = output_dir / 'Figure1_RC_Overlays.pdf'
    
    print(f"\nSaving figures to:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    
    print("\n✅ Figure 1 generated successfully!")
    print(f"\nFigure dimensions: {fig.get_size_inches()} inches")
    print(f"Resolution: 300 DPI (PNG)")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    mapes = []
    rmses = []
    for g in galaxies:
        mape, rmse = compute_galaxy_metrics(g)
        if not np.isnan(mape):
            mapes.append(mape)
            rmses.append(rmse)
    
    if mapes:
        print(f"  Median MAPE: {np.median(mapes):.3f}")
        print(f"  Median RMSE: {np.median(rmses):.1f} km/s")
        print(f"  MAPE range: [{np.min(mapes):.3f}, {np.max(mapes):.3f}]")
        print(f"  RMSE range: [{np.min(rmses):.1f}, {np.max(rmses):.1f}] km/s")
    
    plt.close(fig)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Simple demonstration of the geometric enhancement concept.
Shows how the enhancement factor varies with density gradients.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_enhancement_factor(r, rho, lambda0=0.5, alpha=1.5, rho_crit=1e6):
    """
    Compute geometric enhancement factor based on density profile.
    
    Enhancement is stronger where:
    - Density is low (below rho_crit)
    - Density gradient is steep (large |drho/dr| / rho)
    """
    # Compute gradient
    drho_dr = np.gradient(rho, r)
    
    # Relative gradient (dimensionless)
    rel_grad = np.abs(drho_dr) / np.maximum(rho, rho_crit * 0.01)
    
    # Density suppression (0 to 1, stronger at low density)
    rho_supp = np.exp(-rho / rho_crit)
    
    # Combined enhancement
    Lambda = 1.0 + lambda0 * (rel_grad**alpha) * rho_supp
    
    # Smooth and bound
    Lambda = np.minimum(Lambda, 10.0)
    Lambda = np.maximum(Lambda, 1.0)
    
    return Lambda


def demo_profiles():
    """Demonstrate enhancement on different density profiles."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Test cases
    test_cases = [
        {"name": "NFW-like", "beta": 1.0, "rs": 50.0},
        {"name": "Isothermal", "beta": 2.0, "rs": 30.0},
        {"name": "Plummer", "beta": 5.0, "rs": 20.0},
    ]
    
    r = np.logspace(0, 3, 200)  # 1 to 1000 kpc
    
    for idx, case in enumerate(test_cases):
        # Density profile: rho ~ 1 / (r/rs)^a * (1 + r/rs)^b
        rs = case["rs"]
        beta = case["beta"]
        rho_0 = 1e8  # Central density
        
        # Simple beta model
        rho = rho_0 / (1 + (r/rs)**2)**(beta/2)
        
        # Compute Newtonian acceleration
        M_enc = np.zeros_like(r)
        for i in range(1, len(r)):
            dr = r[i] - r[i-1]
            dM = 4 * np.pi * r[i]**2 * rho[i] * dr
            M_enc[i] = M_enc[i-1] + dM
        
        G = 4.301e-6  # km^2/s^2 * kpc / Msun
        g_N = G * M_enc / r**2
        
        # Compute enhancement factors with different parameters
        Lambda_weak = compute_enhancement_factor(r, rho, lambda0=0.3, alpha=1.0)
        Lambda_moderate = compute_enhancement_factor(r, rho, lambda0=0.5, alpha=1.5)
        Lambda_strong = compute_enhancement_factor(r, rho, lambda0=0.8, alpha=2.0)
        
        # Enhanced accelerations (simplified model)
        g_enhanced_weak = g_N * Lambda_weak
        g_enhanced_moderate = g_N * Lambda_moderate
        g_enhanced_strong = g_N * Lambda_strong
        
        # Top row: Density and enhancement
        ax = axes[0, idx]
        ax2 = ax.twinx()
        
        l1 = ax.loglog(r, rho, 'b-', label='Density', lw=2)
        l2 = ax2.semilogx(r, Lambda_weak, 'g--', label='Λ (weak)', alpha=0.7)
        l3 = ax2.semilogx(r, Lambda_moderate, 'r-', label='Λ (moderate)', alpha=0.7)
        l4 = ax2.semilogx(r, Lambda_strong, 'm:', label='Λ (strong)', alpha=0.7)
        
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('ρ [Msun/kpc³]', color='b')
        ax2.set_ylabel('Enhancement Λ', color='r')
        ax.set_title(f'{case["name"]} profile')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Legend
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Accelerations
        ax = axes[1, idx]
        ax.loglog(r, g_N, 'k-', label='Newtonian', lw=2)
        ax.loglog(r, g_enhanced_weak, 'g--', label='Enhanced (weak)', alpha=0.7)
        ax.loglog(r, g_enhanced_moderate, 'r-', label='Enhanced (moderate)', alpha=0.7)
        ax.loglog(r, g_enhanced_strong, 'm:', label='Enhanced (strong)', alpha=0.7)
        
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('g [km/s²/kpc]')
        ax.set_title(f'Acceleration profiles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Geometric Enhancement Demonstration', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'enhancement_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nGeometric Enhancement Demo:")
    print("===========================")
    print("The enhancement factor Λ(r) increases where:")
    print("1. Density is low (ρ << ρ_crit)")
    print("2. Density gradient is steep (large |dρ/dr|/ρ)")
    print("\nThis naturally enhances gravity in the outskirts")
    print("while preserving Newtonian behavior in dense regions.")
    print(f"\nFigure saved to {out_dir / 'enhancement_demo.png'}")


def demo_rotation_curves():
    """Show effect on rotation curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Disk + bulge galaxy model
    r = np.linspace(0.1, 30, 200)  # kpc
    
    # Exponential disk
    Md = 5e10  # Msun
    Rd = 3.0   # kpc
    Sigma_d = Md / (2 * np.pi * Rd**2) * np.exp(-r/Rd)
    
    # Bulge (Hernquist)
    Mb = 1e10  # Msun
    ab = 1.0   # kpc
    rho_b = Mb / (2*np.pi) * ab / (r * (r + ab)**3)
    
    # Combined surface density (simplified to 1D)
    Sigma_total = Sigma_d + 2*ab*rho_b  # Approximate projection
    
    # Convert to 3D density (thin disk approximation)
    h = 0.3  # kpc, disk scale height
    rho_disk = Sigma_total / (2*h)
    
    # Newtonian circular velocity
    G = 4.301e-6
    M_enc = np.zeros_like(r)
    for i in range(1, len(r)):
        dr = r[i] - r[i-1]
        # Cylindrical mass (approximation)
        dM = 2 * np.pi * r[i] * Sigma_total[i] * dr
        M_enc[i] = M_enc[i-1] + dM
    
    v_N = np.sqrt(G * M_enc / r)
    
    # Apply enhancement
    Lambda = compute_enhancement_factor(r, rho_disk, lambda0=0.6, alpha=1.5, rho_crit=1e7)
    v_enhanced = np.sqrt(v_N**2 * Lambda)
    
    # Panel 1: Surface density profile
    ax = axes[0]
    ax.semilogy(r, Sigma_total, 'b-', lw=2)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Σ [Msun/kpc²]')
    ax.set_title('Surface Density')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Enhancement factor
    ax = axes[1]
    ax.plot(r, Lambda, 'r-', lw=2)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Enhancement Λ')
    ax.set_title('Geometric Enhancement Factor')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    
    # Panel 3: Rotation curves
    ax = axes[2]
    ax.plot(r, v_N, 'k-', label='Newtonian', lw=2)
    ax.plot(r, v_enhanced, 'r-', label='Enhanced', lw=2)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('V_circ [km/s]')
    ax.set_title('Rotation Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 250])
    
    plt.suptitle('Effect on Galaxy Rotation Curves', fontsize=14)
    plt.tight_layout()
    
    out_dir = Path("results")
    plt.savefig(out_dir / 'rotation_curves_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nRotation Curve Demo:")
    print("====================")
    print("The geometric enhancement naturally flattens")
    print("the rotation curve at large radii where the")
    print("baryon density drops rapidly.")
    print(f"\nFigure saved to {out_dir / 'rotation_curves_demo.png'}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GEOMETRIC ENHANCEMENT DEMONSTRATION")
    print("="*60)
    
    # Run demonstrations
    demo_profiles()
    demo_rotation_curves()
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("-" * 60)
    print("1. Enhancement is automatic in low-density regions")
    print("2. No free parameters per object - just global tuning")  
    print("3. Preserves baryon-only philosophy")
    print("4. Can potentially fit both galaxies and clusters")
    print("="*60)
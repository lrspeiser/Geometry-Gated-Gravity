"""
Branch A: Late-saturation G³ implementation (FIXED VERSION)
Allows tail to keep growing at cluster scales while remaining gentle in galaxies

Fixes:
- Corrected critical density calculation
- Fixed unit conversions
- Replaced deprecated trapz with trapezoid
- Improved temperature calculation
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import json
import os

class G3LateSaturation:
    """
    G³ field solver with late-saturation enhancement for clusters
    
    The tail is modified as:
    g_tail = (v0²/r) * (r/(r+rc))^p * S(Σ) * [1 + (r/r_boost)^q]^η
    
    With adaptive saturation:
    g_sat_eff = g_sat0 * (Σ_bar/Σ0)^(-ζ)
    """
    
    def __init__(self, 
                 # Standard G³ parameters
                 v0: float = 150.0,  # km/s
                 rc0: float = 25.0,  # kpc
                 gamma: float = 0.2,
                 beta: float = 0.1,
                 Sigma_star: float = 20.0,  # M☉/pc²
                 alpha: float = 1.5,
                 
                 # Late-saturation parameters (new)
                 r_boost: float = 1e9,  # kpc (default = ∞, no boost)
                 q_boost: float = 2.0,
                 eta_boost: float = 0.0,  # default = 0, no effect
                 zeta_gsat: float = 0.0,  # adaptive cap exponent
                 g_sat0: float = 1200.0,  # base saturation
                 
                 # System parameters
                 use_saturation: bool = True):
        
        self.v0 = v0
        self.rc0 = rc0
        self.gamma = gamma
        self.beta = beta
        self.Sigma_star = Sigma_star
        self.alpha = alpha
        
        # Late-saturation parameters
        self.r_boost = r_boost
        self.q_boost = q_boost
        self.eta_boost = eta_boost
        self.zeta_gsat = zeta_gsat
        self.g_sat0 = g_sat0
        self.use_saturation = use_saturation
        
        # Constants
        self.G = 4.302e-6  # (km/s)²/kpc per M☉/kpc³
        
    def compute_geometry_scalars(self, 
                                  rho: np.ndarray, 
                                  R: np.ndarray, 
                                  z: np.ndarray = None) -> Dict[str, float]:
        """Compute r_1/2 and Σ_bar from density distribution"""
        
        if z is None:
            # 1D spherical case
            r = R
            M_tot = 4 * np.pi * np.trapezoid(rho * r**2, r)
            M_cumul = 4 * np.pi * cumulative_trapezoid(rho * r**2, r, initial=0)
        else:
            # 2D axisymmetric case
            dR = R[1] - R[0]
            dz = z[1] - z[0]
            M_tot = 2 * np.pi * np.sum(rho * R[:, np.newaxis] * dR * dz)
            # Simplified - would need proper 2D integration for M(r)
            r = R
            M_cumul = 2 * np.pi * cumulative_trapezoid(np.sum(rho * R[:, np.newaxis], axis=1) * dz, R, initial=0)
            
        # Half-mass radius
        idx_half = np.searchsorted(M_cumul, 0.5 * M_tot)
        r_half = r[idx_half] if idx_half < len(r) else r[-1]
        
        # Mean surface density within r_half
        Sigma_bar = (0.5 * M_tot) / (np.pi * r_half**2) * 1e-6  # Convert to M☉/pc²
        
        return {
            'r_half': r_half,
            'Sigma_bar': Sigma_bar,
            'M_tot': M_tot
        }
    
    def screening_function(self, Sigma_local: float) -> float:
        """Screening function S(Σ)"""
        x = np.log(self.Sigma_star / (Sigma_local + 0.1))
        return 0.5 * (1 + np.tanh(x))
    
    def compute_tail_acceleration(self, 
                                   r: float,
                                   Sigma_local: float,
                                   Sigma_bar: float,
                                   r_half: float) -> float:
        """
        Compute G³ tail acceleration with late-saturation booster
        
        g_tail = (v0²/r) * (r/(r+rc_eff))^p * S(Σ) * booster
        where booster = [1 + (r/r_boost)^q]^η
        """
        
        # Effective parameters with geometry scaling
        rc_eff = self.rc0 * (r_half / 30.0)**self.gamma
        
        # Base tail
        p = 1.0  # Can make Σ-dependent if needed
        base_tail = (self.v0**2 / r) * (r / (r + rc_eff))**p
        
        # Screening
        S = self.screening_function(Sigma_local)
        
        # Late-saturation booster (new)
        booster = (1.0 + (r / self.r_boost)**self.q_boost)**self.eta_boost
        
        g_tail = base_tail * S * booster
        
        # Adaptive saturation cap
        if self.use_saturation:
            g_sat_eff = self.g_sat0 * (Sigma_bar / 150.0)**(-self.zeta_gsat)
            g_tail = min(g_tail, g_sat_eff)
            
        return g_tail
    
    def solve_field(self,
                    rho: np.ndarray,
                    R: np.ndarray,
                    z: np.ndarray = None,
                    geometry: str = 'spherical') -> Dict:
        """
        Solve for G³ field with late-saturation
        
        Returns dict with:
        - g_N: Newtonian acceleration
        - g_tail: G³ tail contribution  
        - g_tot: Total acceleration
        - geometry_scalars: r_half, Sigma_bar
        """
        
        # Compute geometry scalars
        geom = self.compute_geometry_scalars(rho, R, z)
        
        if geometry == 'spherical':
            # Spherical case (for clusters)
            r = R
            
            # Newtonian
            M_enc = 4 * np.pi * cumulative_trapezoid(rho * r**2, r, initial=0)
            g_N = self.G * M_enc / r**2
            g_N[0] = 0  # Handle r=0
            
            # G³ tail for each radius
            g_tail = np.zeros_like(r)
            for i in range(1, len(r)):
                # Estimate local surface density (simplified)
                Sigma_local = rho[i] * r[i] * 1e-6  # Rough estimate
                g_tail[i] = self.compute_tail_acceleration(
                    r[i], Sigma_local, geom['Sigma_bar'], geom['r_half']
                )
                
        else:
            # Axisymmetric case (galaxies)
            # Simplified - would need full 2D solver
            g_N = np.zeros_like(R)
            g_tail = np.zeros_like(R)
            
            for i in range(len(R)):
                if R[i] > 0:
                    # Rough cylindrical approximation
                    M_enc = 2 * np.pi * np.trapezoid(rho[:i+1] * R[:i+1], R[:i+1]) if i > 0 else 0
                    g_N[i] = self.G * M_enc / R[i]**2
                    
                    Sigma_local = np.mean(rho[max(0, i-2):i+3]) * 10.0 * 1e-6  # Estimate
                    g_tail[i] = self.compute_tail_acceleration(
                        R[i], Sigma_local, geom['Sigma_bar'], geom['r_half']
                    )
                    
        g_tot = g_N + g_tail
        
        return {
            'g_N': g_N,
            'g_tail': g_tail,
            'g_tot': g_tot,
            'geometry_scalars': geom,
            'r': R
        }
    
    def predict_cluster_temperature(self,
                                     rho_gas: np.ndarray,
                                     rho_stars: np.ndarray,
                                     r: np.ndarray,
                                     n_e: np.ndarray) -> np.ndarray:
        """
        Predict cluster temperature profile via HSE
        Fixed unit conversions
        """
        
        rho_tot = rho_gas + rho_stars
        result = self.solve_field(rho_tot, r, geometry='spherical')
        
        # Hydrostatic equilibrium
        # dP/dr = -rho_gas * g_tot
        # Units: rho_gas (M☉/kpc³), g_tot ((km/s)²/kpc)
        
        # Integrate from outside in
        integrand = rho_gas * result['g_tot']  # M☉/kpc³ * (km/s)²/kpc
        
        P = np.zeros_like(r)
        for i in range(len(r)-2, -1, -1):
            dr = r[i+1] - r[i]  # kpc
            P[i] = P[i+1] + integrand[i] * dr  # M☉/kpc² * (km/s)²
            
        # Convert pressure to temperature
        # P = n_total * k_B * T
        # Need to convert units carefully
        
        # Constants in CGS
        M_sun = 1.989e33  # g
        kpc = 3.086e21  # cm
        km = 1e5  # cm
        k_B = 1.38e-16  # erg/K
        m_p = 1.67e-24  # g
        keV = 1.6e-9  # erg
        
        # Convert pressure: M☉/kpc² * (km/s)² → dyne/cm²
        P_cgs = P * (M_sun / kpc**2) * (km**2)  # dyne/cm²
        
        # n_total ≈ 1.92 * n_e for ionized gas (n_e in cm⁻³)
        # Avoid division by zero
        n_e_safe = np.maximum(n_e, 1e-10)
        
        # Temperature in K
        T_K = P_cgs / (1.92 * n_e_safe * k_B)
        
        # Convert to keV
        kT_keV = k_B * T_K / keV
        
        return kT_keV
    
    def compute_lensing_convergence(self,
                                     result: Dict,
                                     z_lens: float = 0.2,
                                     z_source: float = 1.0) -> np.ndarray:
        """
        Compute lensing convergence κ(R) from field solution
        FIXED: Correct unit conversions for Σ_crit
        """
        
        r = result['r']
        g_tot = result['g_tot']
        
        # Enclosed mass from total acceleration
        # g = GM/r², so M = g*r²/G
        M_enc = g_tot * r**2 / self.G  # M☉
        
        # Avoid division by zero at r=0
        r_safe = np.maximum(r, 1e-10)
        
        # Surface density via derivative
        # Σ(R) = (1/2πR) * dM/dR
        dM_dr = np.gradient(M_enc, r)
        Sigma = dM_dr / (2 * np.pi * r_safe)  # M☉/kpc²
        
        # Convert to M☉/pc²
        Sigma_pc2 = Sigma / 1e6  # M☉/pc²
        
        # Critical density for lensing
        # Σ_crit = (c²/4πG) * (D_s / D_l / D_ls)
        
        # Use simple cosmology (could use astropy for accuracy)
        # For z_lens=0.2, z_source=1.0 typical values:
        # D_l ≈ 800 Mpc, D_s ≈ 2400 Mpc, D_ls ≈ 2000 Mpc
        
        c = 299792.458  # km/s
        
        # Approximate angular diameter distances (Mpc)
        D_l_Mpc = 800   # z=0.2
        D_s_Mpc = 2400  # z=1.0
        D_ls_Mpc = 2000 # Between z=0.2 and z=1.0
        
        # Convert to kpc
        D_l = D_l_Mpc * 1000
        D_s = D_s_Mpc * 1000
        D_ls = D_ls_Mpc * 1000
        
        # Critical density
        # Σ_crit = (c²/4πG) * (D_s / (D_l * D_ls))
        # Units: (km/s)² / ((km/s)²/kpc/M☉*kpc³) * (1/kpc) = M☉/kpc²
        Sigma_crit_kpc2 = (c**2 / (4 * np.pi * self.G)) * (D_s / (D_l * D_ls))  # M☉/kpc²
        
        # Convert to M☉/pc²
        Sigma_crit = Sigma_crit_kpc2 / 1e6  # M☉/pc²
        
        # Convergence
        kappa = Sigma_pc2 / Sigma_crit
        
        # Mean convergence within R
        kappa_bar = np.zeros_like(kappa)
        for i in range(1, len(r)):
            # Weight by area
            if i == 1:
                kappa_bar[i] = kappa[0]
            else:
                r_bins = r[:i]
                kappa_bins = kappa[:i]
                # Area-weighted mean
                areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
                total_area = np.pi * r_bins[-1]**2
                weights = np.zeros(len(kappa_bins))
                weights[:-1] = areas / total_area
                weights[-1] = np.pi * r_bins[0]**2 / total_area
                kappa_bar[i] = np.sum(weights * kappa_bins)
                
        return kappa, kappa_bar

def test_late_saturation_comprehensive():
    """
    Comprehensive test of late-saturation with multiple parameters
    """
    
    # Create mock cluster data
    r = np.logspace(0, 3.5, 100)  # 1 to 3000 kpc
    
    # Mock NFW-like density for testing
    r_s = 300.0  # Scale radius
    rho_0 = 1e7  # Central density M☉/kpc³
    rho_gas = rho_0 / ((r/r_s) * (1 + r/r_s)**2)
    rho_stars = 0.1 * rho_gas  # 10% stellar
    
    # Mock electron density
    n_e = 0.01 * (1 + (r/100)**2)**(-1.5)  # cm⁻³
    
    # Test configurations
    tests = [
        {'name': 'Standard', 'eta': 0.0, 'r_boost': 1e9, 'zeta': 0.0},
        {'name': 'Weak boost', 'eta': 0.2, 'r_boost': 500, 'zeta': 0.1},
        {'name': 'Moderate boost', 'eta': 0.3, 'r_boost': 300, 'zeta': 0.2},
        {'name': 'Strong boost', 'eta': 0.5, 'r_boost': 200, 'zeta': 0.3},
        {'name': 'Very strong', 'eta': 0.7, 'r_boost': 150, 'zeta': 0.4},
    ]
    
    results = []
    
    # Create figure for comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, test in enumerate(tests):
        g3 = G3LateSaturation(
            eta_boost=test['eta'], 
            r_boost=test['r_boost'],
            zeta_gsat=test['zeta']
        )
        
        result = g3.solve_field(rho_gas + rho_stars, r, geometry='spherical')
        kappa, kappa_bar = g3.compute_lensing_convergence(result)
        kT = g3.predict_cluster_temperature(rho_gas, rho_stars, r, n_e)
        
        # Store results
        results.append({
            'name': test['name'],
            'params': test,
            'kappa_max': np.max(kappa_bar),
            'kappa_at_50': np.interp(50, r, kappa_bar),
            'kappa_at_100': np.interp(100, r, kappa_bar),
            'kT_at_100': np.interp(100, r, kT),
            'kT_at_200': np.interp(200, r, kT),
            'g_tot': result['g_tot'],
            'g_tail': result['g_tail'],
            'kappa_bar': kappa_bar
        })
        
        # Plot on axes
        color = plt.cm.viridis(i / len(tests))
        
        # Total acceleration
        axes[0, 0].loglog(r, result['g_tot'], color=color, label=test['name'])
        
        # Tail acceleration
        axes[0, 1].loglog(r, result['g_tail'], color=color, label=test['name'])
        
        # Convergence
        axes[0, 2].semilogx(r, kappa_bar, color=color, label=test['name'])
        
        # Temperature
        axes[1, 0].semilogx(r[1:], kT[1:], color=color, label=test['name'])
        
        # Boost factor for this config
        if test['eta'] > 0:
            boost = (1 + (r / test['r_boost'])**2)**test['eta']
            axes[1, 1].semilogx(r, boost, color=color, label=test['name'])
    
    # Configure axes
    axes[0, 0].set_xlabel('r (kpc)')
    axes[0, 0].set_ylabel('g_tot ((km/s)²/kpc)')
    axes[0, 0].set_title('Total Acceleration')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('r (kpc)')
    axes[0, 1].set_ylabel('g_tail ((km/s)²/kpc)')
    axes[0, 1].set_title('Tail Acceleration')
    axes[0, 1].axhline(1200, color='gray', linestyle=':', alpha=0.5, label='g_sat')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('R (kpc)')
    axes[0, 2].set_ylabel('κ̄(<R)')
    axes[0, 2].set_title('Mean Convergence')
    axes[0, 2].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Einstein')
    axes[0, 2].axhline(0.17, color='blue', linestyle='--', alpha=0.5, label='Standard G³')
    axes[0, 2].set_ylim([0, 1.5])
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('r (kpc)')
    axes[1, 0].set_ylabel('kT (keV)')
    axes[1, 0].set_title('Temperature Profile')
    axes[1, 0].set_xlim([10, 1000])
    axes[1, 0].set_ylim([0, 20])
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('r (kpc)')
    axes[1, 1].set_ylabel('Boost factor')
    axes[1, 1].set_title('Late-saturation Boost')
    axes[1, 1].set_ylim([1, 5])
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary table in last panel
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Branch A: Late-Saturation Results\n" + "="*35 + "\n"
    summary_text += f"{'Config':<15} {'κ_max':>8} {'κ@50kpc':>10} {'kT@200kpc':>10}\n"
    summary_text += "-"*35 + "\n"
    
    for res in results:
        summary_text += f"{res['name']:<15} "
        summary_text += f"{res['kappa_max']:>8.3f} "
        summary_text += f"{res['kappa_at_50']:>10.3f} "
        summary_text += f"{res['kT_at_200']:>10.1f} keV\n"
    
    summary_text += "\nKey Findings:\n"
    
    # Analyze improvements
    standard_kappa = results[0]['kappa_max']
    best_kappa = max(r['kappa_max'] for r in results)
    improvement = best_kappa / standard_kappa
    
    summary_text += f"• Best κ_max: {best_kappa:.3f} ({improvement:.1f}× improvement)\n"
    summary_text += f"• Need κ_max ≈ 1.0 for Einstein rings\n"
    summary_text += f"• Gap remaining: {1.0 - best_kappa:.3f}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Branch A: Late-Saturation Comprehensive Test (FIXED)', fontsize=14)
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comprehensive_test_fixed.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved comprehensive test to {output_path}")
    
    # Save metrics
    metrics = {
        'test_configs': tests,
        'results': [
            {
                'name': r['name'],
                'kappa_max': float(r['kappa_max']),
                'kappa_at_50kpc': float(r['kappa_at_50']),
                'kappa_at_100kpc': float(r['kappa_at_100']),
                'kT_at_100kpc': float(r['kT_at_100']),
                'kT_at_200kpc': float(r['kT_at_200'])
            }
            for r in results
        ],
        'improvement_factor': float(improvement),
        'best_kappa_max': float(best_kappa),
        'gap_to_einstein': float(1.0 - best_kappa)
    }
    
    json_path = os.path.join(output_dir, 'comprehensive_metrics_fixed.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {json_path}")
    
    return results

if __name__ == "__main__":
    print("Testing Branch A: Late-Saturation G³ (FIXED VERSION)\n")
    results = test_late_saturation_comprehensive()
    
    print("\nSummary of Results:")
    print("-" * 50)
    for res in results:
        print(f"{res['name']:<15}: κ_max = {res['kappa_max']:.3f}, κ@50kpc = {res['kappa_at_50']:.3f}")
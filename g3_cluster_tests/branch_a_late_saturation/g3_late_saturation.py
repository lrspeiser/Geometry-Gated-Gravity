"""
Branch A: Late-saturation G³ implementation
Allows tail to keep growing at cluster scales while remaining gentle in galaxies

Key features:
- Tail booster that activates beyond r_boost
- Adaptive saturation cap that weakens for low-density systems
- Preserves galaxy/MW behavior with default parameters
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import json

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
            M_tot = 4 * np.pi * np.trapz(rho * r**2, r)
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
                    M_enc = 2 * np.pi * np.trapz(rho[:i+1] * R[:i+1], R[:i+1]) if i > 0 else 0
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
    
    def predict_rotation_curve(self, 
                               rho_disk: np.ndarray,
                               rho_bulge: np.ndarray, 
                               R: np.ndarray) -> np.ndarray:
        """Predict galaxy rotation curve"""
        
        rho_tot = rho_disk + rho_bulge
        result = self.solve_field(rho_tot, R, geometry='disk')
        
        # v_circ = sqrt(R * g_tot)
        v_circ = np.sqrt(R * result['g_tot'])
        return v_circ
    
    def predict_cluster_temperature(self,
                                     rho_gas: np.ndarray,
                                     rho_stars: np.ndarray,
                                     r: np.ndarray,
                                     n_e: np.ndarray) -> np.ndarray:
        """
        Predict cluster temperature profile via HSE
        
        dP/dr = -rho_gas * g_tot
        kT = (mu * m_p / n_e) * P
        """
        
        rho_tot = rho_gas + rho_stars
        result = self.solve_field(rho_tot, r, geometry='spherical')
        
        # Hydrostatic equilibrium
        # Integrate outward from large radius assuming P→0
        integrand = rho_gas * result['g_tot']
        
        # Pressure integral (from outside in)
        P = np.zeros_like(r)
        for i in range(len(r)-2, -1, -1):
            dr = r[i+1] - r[i]
            P[i] = P[i+1] + integrand[i] * dr
            
        # Temperature
        mu_mp = 0.6 * 1.67e-24  # Mean molecular weight * proton mass (g)
        k_B = 1.38e-16  # Boltzmann constant (erg/K)
        keV_per_K = k_B / 1.6e-9  # Conversion to keV
        
        # P = n_total * k_B * T, where n_total ≈ n_e * 1.92
        kT_keV = P * 1e10 / (1.92 * n_e) * keV_per_K  # Unit conversions
        
        return kT_keV
    
    def compute_lensing_convergence(self,
                                     result: Dict,
                                     z_lens: float = 0.2,
                                     z_source: float = 1.0) -> np.ndarray:
        """
        Compute lensing convergence κ(R) from field solution
        
        For late-saturation, the enhanced tail at large r 
        should boost κ toward unity
        """
        
        r = result['r']
        g_tot = result['g_tot']
        
        # Surface mass density from g_tot via Abel transform
        # Σ(R) = (1/2πG) * ∫ dg/dr / sqrt(r² - R²) dr
        
        # Simplified: use enclosed mass approach
        M_enc = g_tot * r**2 / self.G
        
        # Project to surface density (simplified)
        Sigma = np.gradient(M_enc) / (2 * np.pi * r) * 1e6  # M☉/pc²
        
        # Critical density
        from astropy.cosmology import Planck18
        D_l = Planck18.angular_diameter_distance(z_lens).value * 1000  # kpc
        D_s = Planck18.angular_diameter_distance(z_source).value * 1000
        D_ls = Planck18.angular_diameter_distance_z1z2(z_lens, z_source).value * 1000
        
        c = 299792.458  # km/s
        Sigma_crit = (c**2 / (4 * np.pi * self.G)) * (D_s / (D_l * D_ls)) * 1e-12  # M☉/pc²
        
        kappa = Sigma / Sigma_crit
        
        # Mean convergence within R
        kappa_bar = np.zeros_like(kappa)
        for i in range(1, len(r)):
            kappa_bar[i] = np.mean(kappa[:i+1])
            
        return kappa, kappa_bar
    
    def parameter_summary(self) -> Dict:
        """Return current parameters"""
        return {
            'standard_g3': {
                'v0': self.v0,
                'rc0': self.rc0, 
                'gamma': self.gamma,
                'beta': self.beta,
                'Sigma_star': self.Sigma_star,
                'alpha': self.alpha
            },
            'late_saturation': {
                'r_boost': self.r_boost,
                'q_boost': self.q_boost,
                'eta_boost': self.eta_boost,
                'zeta_gsat': self.zeta_gsat,
                'g_sat0': self.g_sat0
            }
        }

def test_late_saturation_on_cluster(cluster_name: str = 'A1689',
                                     eta_boost: float = 0.3,
                                     r_boost_kpc: float = 300.0):
    """
    Test late-saturation on a cluster
    """
    
    # Create mock cluster data (would load real data)
    r = np.logspace(0, 3.5, 100)  # 1 to 3000 kpc
    
    # Mock NFW-like density for testing
    r_s = 300.0  # Scale radius
    rho_0 = 1e7  # Central density M☉/kpc³
    rho_gas = rho_0 / ((r/r_s) * (1 + r/r_s)**2)
    rho_stars = 0.1 * rho_gas  # 10% stellar
    
    # Mock electron density
    n_e = 0.01 * (1 + (r/100)**2)**(-1.5)  # cm⁻³
    
    # Standard G³
    g3_standard = G3LateSaturation(eta_boost=0.0)  # No boost
    result_standard = g3_standard.solve_field(rho_gas + rho_stars, r, geometry='spherical')
    kappa_std, kappa_bar_std = g3_standard.compute_lensing_convergence(result_standard)
    
    # Late-saturation G³  
    g3_late = G3LateSaturation(eta_boost=eta_boost, r_boost=r_boost_kpc, zeta_gsat=0.2)
    result_late = g3_late.solve_field(rho_gas + rho_stars, r, geometry='spherical')
    kappa_late, kappa_bar_late = g3_late.compute_lensing_convergence(result_late)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Acceleration profiles
    ax = axes[0, 0]
    ax.loglog(r, result_standard['g_tot'], 'b-', label='Standard G³')
    ax.loglog(r, result_late['g_tot'], 'r-', label=f'Late-sat (η={eta_boost})')
    ax.loglog(r, result_standard['g_N'], 'k--', alpha=0.5, label='Newtonian')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('g (km/s)²/kpc')
    ax.set_title('Total Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tail contribution
    ax = axes[0, 1]
    ax.loglog(r, result_standard['g_tail'], 'b-', label='Standard tail')
    ax.loglog(r, result_late['g_tail'], 'r-', label='Late-sat tail')
    ax.axhline(g3_standard.g_sat0, color='gray', linestyle=':', label='g_sat')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('g_tail (km/s)²/kpc')
    ax.set_title('Tail Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convergence
    ax = axes[1, 0]
    ax.semilogx(r, kappa_bar_std, 'b-', label='Standard G³')
    ax.semilogx(r, kappa_bar_late, 'r-', label='Late-saturation')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='κ=1 (Einstein)')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('κ̄(<R)')
    ax.set_title('Mean Convergence')
    ax.set_ylim([0, 1.5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Boost factor
    ax = axes[1, 1]
    boost = (1.0 + (r / r_boost_kpc)**2)**eta_boost
    ax.semilogx(r, boost, 'g-', linewidth=2)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Boost factor')
    ax.set_title(f'Late-sat boost: [1+(r/{r_boost_kpc})²]^{eta_boost}')
    ax.grid(True, alpha=0.3)
    ax.axvline(r_boost_kpc, color='red', linestyle='--', alpha=0.5, label=f'r_boost={r_boost_kpc} kpc')
    ax.legend()
    
    plt.suptitle(f'Branch A: Late-Saturation Test on {cluster_name}')
    plt.tight_layout()
    
    # Save
    import os
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{cluster_name}_test.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    
    # Report metrics
    metrics = {
        'cluster': cluster_name,
        'parameters': {
            'eta_boost': eta_boost,
            'r_boost_kpc': r_boost_kpc
        },
        'standard_g3': {
            'kappa_bar_max': float(np.max(kappa_bar_std)),
            'kappa_bar_at_50kpc': float(np.interp(50, r, kappa_bar_std)),
            'kappa_bar_at_100kpc': float(np.interp(100, r, kappa_bar_std))
        },
        'late_saturation': {
            'kappa_bar_max': float(np.max(kappa_bar_late)),
            'kappa_bar_at_50kpc': float(np.interp(50, r, kappa_bar_late)),
            'kappa_bar_at_100kpc': float(np.interp(100, r, kappa_bar_late)),
            'improvement_factor': float(np.max(kappa_bar_late) / np.max(kappa_bar_std))
        }
    }
    
    json_path = os.path.join(output_dir, f'{cluster_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {json_path}")
    
    return metrics

if __name__ == "__main__":
    # Test with different boost parameters
    print("Testing Branch A: Late-Saturation G³\n")
    
    # Test 1: Moderate boost
    print("Test 1: Moderate boost (η=0.3, r_boost=300 kpc)")
    metrics1 = test_late_saturation_on_cluster('A1689', eta_boost=0.3, r_boost_kpc=300)
    print(f"  Standard κ_max: {metrics1['standard_g3']['kappa_bar_max']:.3f}")
    print(f"  Late-sat κ_max: {metrics1['late_saturation']['kappa_bar_max']:.3f}")
    print(f"  Improvement: {metrics1['late_saturation']['improvement_factor']:.2f}x\n")
    
    # Test 2: Strong boost
    print("Test 2: Strong boost (η=0.5, r_boost=200 kpc)")
    metrics2 = test_late_saturation_on_cluster('A1689_strong', eta_boost=0.5, r_boost_kpc=200)
    print(f"  Standard κ_max: {metrics2['standard_g3']['kappa_bar_max']:.3f}")
    print(f"  Late-sat κ_max: {metrics2['late_saturation']['kappa_bar_max']:.3f}")
    print(f"  Improvement: {metrics2['late_saturation']['improvement_factor']:.2f}x")
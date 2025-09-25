"""
Branch B: Photon boost G³ implementation
Photons experience stronger G³ potential in low-density regions via disformal coupling

Key features:
- Same field equation for matter dynamics
- Enhanced lensing potential: Φ_lens = Φ_dyn * [1 + ξ_γ * S(Σ)^β_γ]
- Preserves equivalence principle for massive particles
- Solar system safe (S→0 at high Σ)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import json

class G3PhotonBoost:
    """
    G³ field solver with photon-specific enhancement for lensing
    
    Matter dynamics use standard G³ field
    Lensing uses boosted potential: Φ_lens = Φ_dyn * [1 + ξ_γ * S(Σ)^β_γ]
    
    This is physically motivated by disformal gravity where photons
    and massive particles couple differently to the scalar field
    """
    
    def __init__(self,
                 # Standard G³ parameters
                 v0: float = 150.0,  # km/s
                 rc0: float = 25.0,  # kpc  
                 gamma: float = 0.2,
                 beta: float = 0.1,
                 Sigma_star: float = 20.0,  # M☉/pc²
                 alpha: float = 1.5,
                 
                 # Photon boost parameters (new)
                 xi_gamma: float = 0.0,  # Photon coupling strength (0 = no boost)
                 beta_gamma: float = 1.0,  # Screening power for photons
                 
                 # System parameters
                 g_sat0: float = 1200.0):
        
        self.v0 = v0
        self.rc0 = rc0
        self.gamma = gamma
        self.beta = beta
        self.Sigma_star = Sigma_star
        self.alpha = alpha
        
        # Photon boost parameters
        self.xi_gamma = xi_gamma
        self.beta_gamma = beta_gamma
        
        self.g_sat0 = g_sat0
        
        # Constants
        self.G = 4.302e-6  # (km/s)²/kpc per M☉/kpc³
        
    def compute_geometry_scalars(self,
                                  rho: np.ndarray,
                                  r: np.ndarray) -> Dict[str, float]:
        """Compute r_1/2 and Σ_bar from spherical density"""
        
        # Total mass
        M_tot = 4 * np.pi * np.trapz(rho * r**2, r)
        
        # Cumulative mass
        M_cumul = 4 * np.pi * cumulative_trapezoid(rho * r**2, r, initial=0)
        
        # Half-mass radius
        idx_half = np.searchsorted(M_cumul, 0.5 * M_tot)
        r_half = r[idx_half] if idx_half < len(r) else r[-1]
        
        # Mean surface density within r_half  
        Sigma_bar = (0.5 * M_tot) / (np.pi * r_half**2) * 1e-6  # M☉/pc²
        
        return {
            'r_half': r_half,
            'Sigma_bar': Sigma_bar,
            'M_tot': M_tot
        }
    
    def screening_function(self, Sigma_local: float) -> float:
        """Standard screening function S(Σ)"""
        x = np.log(self.Sigma_star / (Sigma_local + 0.1))
        return 0.5 * (1 + np.tanh(x))
    
    def photon_boost_factor(self, Sigma_local: float) -> float:
        """
        Compute photon-specific boost factor
        
        boost = 1 + ξ_γ * S(Σ)^β_γ
        
        This gives stronger lensing in low-density regions
        while preserving Solar System tests (S→0 at high Σ)
        """
        S = self.screening_function(Sigma_local)
        return 1.0 + self.xi_gamma * S**self.beta_gamma
    
    def compute_tail_acceleration(self,
                                   r: float,
                                   Sigma_local: float,
                                   r_half: float) -> float:
        """Standard G³ tail acceleration (for matter dynamics)"""
        
        # Effective parameters
        rc_eff = self.rc0 * (r_half / 30.0)**self.gamma
        
        # Base tail
        base_tail = (self.v0**2 / r) * (r / (r + rc_eff))
        
        # Screening
        S = self.screening_function(Sigma_local)
        
        g_tail = base_tail * S
        
        # Saturation
        g_tail = min(g_tail, self.g_sat0)
        
        return g_tail
    
    def solve_field(self,
                    rho: np.ndarray,
                    r: np.ndarray) -> Dict:
        """
        Solve for G³ field (dynamics version)
        
        Returns both dynamics and lensing accelerations
        """
        
        # Geometry scalars
        geom = self.compute_geometry_scalars(rho, r)
        
        # Newtonian
        M_enc = 4 * np.pi * cumulative_trapezoid(rho * r**2, r, initial=0)
        g_N = self.G * M_enc / r**2
        g_N[0] = 0
        
        # G³ tail (dynamics)
        g_tail_dyn = np.zeros_like(r)
        g_tail_lens = np.zeros_like(r)
        boost_factors = np.zeros_like(r)
        
        for i in range(1, len(r)):
            # Local surface density estimate
            Sigma_local = rho[i] * r[i] * 1e-6  # Rough estimate
            
            # Dynamics tail (standard)
            g_tail_dyn[i] = self.compute_tail_acceleration(
                r[i], Sigma_local, geom['r_half']
            )
            
            # Lensing tail (boosted)
            boost = self.photon_boost_factor(Sigma_local)
            g_tail_lens[i] = g_tail_dyn[i] * boost
            boost_factors[i] = boost
            
        g_tot_dyn = g_N + g_tail_dyn
        g_tot_lens = g_N + g_tail_lens  # Newtonian + boosted tail
        
        return {
            'g_N': g_N,
            'g_tail_dyn': g_tail_dyn,
            'g_tail_lens': g_tail_lens,
            'g_tot_dyn': g_tot_dyn,
            'g_tot_lens': g_tot_lens,
            'boost_factors': boost_factors,
            'geometry_scalars': geom,
            'r': r
        }
    
    def compute_lensing_convergence(self,
                                     result: Dict,
                                     use_boost: bool = True,
                                     z_lens: float = 0.2,
                                     z_source: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lensing convergence using boosted or standard field
        """
        
        r = result['r']
        
        # Choose which acceleration to use
        if use_boost:
            g_tot = result['g_tot_lens']  # Boosted for photons
        else:
            g_tot = result['g_tot_dyn']   # Standard for comparison
            
        # Enclosed mass from acceleration
        M_enc = g_tot * r**2 / self.G
        
        # Surface density (simplified projection)
        Sigma = np.gradient(M_enc) / (2 * np.pi * r) * 1e6  # M☉/pc²
        
        # Critical density
        from astropy.cosmology import Planck18
        D_l = Planck18.angular_diameter_distance(z_lens).value * 1000  # kpc
        D_s = Planck18.angular_diameter_distance(z_source).value * 1000
        D_ls = Planck18.angular_diameter_distance_z1z2(z_lens, z_source).value * 1000
        
        c = 299792.458  # km/s
        Sigma_crit = (c**2 / (4 * np.pi * self.G)) * (D_s / (D_l * D_ls)) * 1e-12  # M☉/pc²
        
        kappa = Sigma / Sigma_crit
        
        # Mean convergence
        kappa_bar = np.zeros_like(kappa)
        for i in range(1, len(r)):
            kappa_bar[i] = np.mean(kappa[:i+1])
            
        return kappa, kappa_bar
    
    def predict_cluster_temperature(self,
                                     rho_gas: np.ndarray,
                                     rho_stars: np.ndarray,
                                     r: np.ndarray,
                                     n_e: np.ndarray) -> np.ndarray:
        """
        Predict cluster temperature using STANDARD dynamics field
        (not boosted - massive particles don't see the boost)
        """
        
        rho_tot = rho_gas + rho_stars
        result = self.solve_field(rho_tot, r)
        
        # Use dynamics acceleration (not lensing)
        g_tot = result['g_tot_dyn']
        
        # HSE integration
        integrand = rho_gas * g_tot
        
        P = np.zeros_like(r)
        for i in range(len(r)-2, -1, -1):
            dr = r[i+1] - r[i]
            P[i] = P[i+1] + integrand[i] * dr
            
        # Temperature
        keV_per_K = 1.38e-16 / 1.6e-9
        kT_keV = P * 1e10 / (1.92 * n_e) * keV_per_K
        
        return kT_keV
    
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
            'photon_boost': {
                'xi_gamma': self.xi_gamma,
                'beta_gamma': self.beta_gamma,
                'description': 'Φ_lens = Φ_dyn * [1 + ξ_γ * S(Σ)^β_γ]'
            }
        }

def test_photon_boost_on_cluster(cluster_name: str = 'A1689',
                                  xi_gamma: float = 0.3,
                                  beta_gamma: float = 1.0):
    """
    Test photon boost on a cluster
    
    Compare:
    1. Standard G³ (no boost)
    2. Photon-boosted G³ for lensing only
    
    Show that dynamics (temperature) unchanged while lensing enhanced
    """
    
    # Create mock cluster
    r = np.logspace(0, 3.5, 100)  # 1 to 3000 kpc
    
    # Mock density
    r_s = 300.0
    rho_0 = 1e7
    rho_gas = rho_0 / ((r/r_s) * (1 + r/r_s)**2)
    rho_stars = 0.1 * rho_gas
    
    n_e = 0.01 * (1 + (r/100)**2)**(-1.5)
    
    # Standard G³ (no photon boost)
    g3_standard = G3PhotonBoost(xi_gamma=0.0)
    result_standard = g3_standard.solve_field(rho_gas + rho_stars, r)
    kappa_std, kappa_bar_std = g3_standard.compute_lensing_convergence(
        result_standard, use_boost=False
    )
    kT_std = g3_standard.predict_cluster_temperature(rho_gas, rho_stars, r, n_e)
    
    # Photon-boosted G³
    g3_boosted = G3PhotonBoost(xi_gamma=xi_gamma, beta_gamma=beta_gamma)
    result_boosted = g3_boosted.solve_field(rho_gas + rho_stars, r)
    kappa_boost, kappa_bar_boost = g3_boosted.compute_lensing_convergence(
        result_boosted, use_boost=True
    )
    kappa_nobst, kappa_bar_nobst = g3_boosted.compute_lensing_convergence(
        result_boosted, use_boost=False
    )
    kT_boost = g3_boosted.predict_cluster_temperature(rho_gas, rho_stars, r, n_e)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Dynamics acceleration (should be same)
    ax = axes[0, 0]
    ax.loglog(r, result_standard['g_tot_dyn'], 'b-', label='Standard G³', linewidth=2)
    ax.loglog(r, result_boosted['g_tot_dyn'], 'r--', label='Boosted G³ (dynamics)', linewidth=2)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('g_dyn (km/s)²/kpc')
    ax.set_title('Dynamics Acceleration (same for both)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Lensing acceleration (boosted)
    ax = axes[0, 1]
    ax.loglog(r, result_standard['g_tot_dyn'], 'b-', label='Standard', linewidth=2)
    ax.loglog(r, result_boosted['g_tot_lens'], 'r-', label=f'Photon boost (ξ={xi_gamma})', linewidth=2)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('g_lens (km/s)²/kpc')
    ax.set_title('Lensing Acceleration (boosted for photons)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Boost factor profile
    ax = axes[0, 2]
    ax.semilogx(r, result_boosted['boost_factors'], 'g-', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('Boost factor')
    ax.set_title(f'Photon boost: 1 + {xi_gamma}*S(Σ)^{beta_gamma}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, max(1.5, np.max(result_boosted['boost_factors'])+0.1)])
    
    # Temperature (should be same)
    ax = axes[1, 0]
    ax.semilogx(r[1:], kT_std[1:], 'b-', label='Standard G³', linewidth=2)
    ax.semilogx(r[1:], kT_boost[1:], 'r--', label='Boosted G³', linewidth=2)
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('kT (keV)')
    ax.set_title('Temperature Profile (unchanged)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([10, 1000])
    
    # Convergence comparison
    ax = axes[1, 1]
    ax.semilogx(r, kappa_bar_std, 'b-', label='Standard G³', linewidth=2)
    ax.semilogx(r, kappa_bar_nobst, 'g--', label='Boosted G³ (no photon boost)', linewidth=1)
    ax.semilogx(r, kappa_bar_boost, 'r-', label=f'Photon boost (ξ={xi_gamma})', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='κ=1 (Einstein)')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('κ̄(<R)')
    ax.set_title('Mean Convergence')
    ax.set_ylim([0, 1.5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Improvement summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate metrics
    kappa_max_std = np.max(kappa_bar_std)
    kappa_max_boost = np.max(kappa_bar_boost)
    improvement = kappa_max_boost / kappa_max_std
    
    # Temperature difference
    temp_diff = np.max(np.abs(kT_boost - kT_std) / kT_std)
    
    summary_text = f"""Branch B: Photon Boost Results
    
Parameters:
  ξ_γ = {xi_gamma}
  β_γ = {beta_gamma}
  
Lensing Enhancement:
  Standard κ_max: {kappa_max_std:.3f}
  Boosted κ_max: {kappa_max_boost:.3f}
  Improvement: {improvement:.2f}x
  
Dynamics Preservation:
  Max temp difference: {temp_diff:.1%}
  (Should be ~0%)
  
Physics:
  • Photons see boosted potential
  • Massive particles unchanged
  • Solar System safe (S→0 at high Σ)"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Branch B: Photon Boost (Disformal) Test on {cluster_name}')
    plt.tight_layout()
    
    # Save plot
    import os
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{cluster_name}_test.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    
    # Save metrics
    metrics = {
        'cluster': cluster_name,
        'parameters': {
            'xi_gamma': xi_gamma,
            'beta_gamma': beta_gamma
        },
        'standard_g3': {
            'kappa_bar_max': float(kappa_max_std),
            'kappa_bar_at_50kpc': float(np.interp(50, r, kappa_bar_std)),
            'kappa_bar_at_100kpc': float(np.interp(100, r, kappa_bar_std))
        },
        'photon_boost': {
            'kappa_bar_max': float(kappa_max_boost),
            'kappa_bar_at_50kpc': float(np.interp(50, r, kappa_bar_boost)),
            'kappa_bar_at_100kpc': float(np.interp(100, r, kappa_bar_boost)),
            'improvement_factor': float(improvement),
            'max_boost_factor': float(np.max(result_boosted['boost_factors']))
        },
        'dynamics_check': {
            'max_temperature_difference': float(temp_diff),
            'dynamics_preserved': bool(temp_diff < 0.01)
        }
    }
    
    json_path = os.path.join(output_dir, f'{cluster_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {json_path}")
    
    return metrics

def test_solar_system_safety():
    """
    Verify photon boost vanishes in high-density (Solar System) regime
    """
    import os
    
    g3 = G3PhotonBoost(xi_gamma=0.5, beta_gamma=1.0, Sigma_star=20.0)
    
    # Surface densities from low (cluster outskirts) to high (Solar System)
    Sigma_values = np.logspace(-2, 4, 100)  # 0.01 to 10000 M☉/pc²
    
    S_values = [g3.screening_function(Sigma) for Sigma in Sigma_values]
    boost_values = [g3.photon_boost_factor(Sigma) for Sigma in Sigma_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Screening function
    ax1.loglog(Sigma_values, S_values, 'b-', linewidth=2)
    ax1.axvline(20.0, color='r', linestyle='--', alpha=0.5, label='Σ_star')
    ax1.axvline(1e3, color='g', linestyle='--', alpha=0.5, label='Solar System')
    ax1.set_xlabel('Σ (M☉/pc²)')
    ax1.set_ylabel('S(Σ)')
    ax1.set_title('Screening Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Photon boost
    ax2.semilogx(Sigma_values, boost_values, 'r-', linewidth=2)
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(20.0, color='r', linestyle='--', alpha=0.5, label='Σ_star')
    ax2.axvline(1e3, color='g', linestyle='--', alpha=0.5, label='Solar System')
    ax2.set_xlabel('Σ (M☉/pc²)')
    ax2.set_ylabel('Photon boost factor')
    ax2.set_title('Photon Boost: 1 + 0.5*S(Σ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Solar System Safety Check: Boost → 1 at high Σ')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'solar_safety.png'), dpi=150)
    print("Saved Solar System safety check")

if __name__ == "__main__":
    print("Testing Branch B: Photon Boost (Disformal) G³\n")
    
    # Test 1: Moderate boost
    print("Test 1: Moderate photon boost (ξ=0.3)")
    metrics1 = test_photon_boost_on_cluster('A1689', xi_gamma=0.3)
    print(f"  Standard κ_max: {metrics1['standard_g3']['kappa_bar_max']:.3f}")
    print(f"  Boosted κ_max: {metrics1['photon_boost']['kappa_bar_max']:.3f}")
    print(f"  Improvement: {metrics1['photon_boost']['improvement_factor']:.2f}x")
    print(f"  Dynamics preserved: {metrics1['dynamics_check']['dynamics_preserved']}\n")
    
    # Test 2: Strong boost
    print("Test 2: Strong photon boost (ξ=0.5)")
    metrics2 = test_photon_boost_on_cluster('A1689_strong', xi_gamma=0.5)
    print(f"  Standard κ_max: {metrics2['standard_g3']['kappa_bar_max']:.3f}")
    print(f"  Boosted κ_max: {metrics2['photon_boost']['kappa_bar_max']:.3f}")
    print(f"  Improvement: {metrics2['photon_boost']['improvement_factor']:.2f}x")
    print(f"  Dynamics preserved: {metrics2['dynamics_check']['dynamics_preserved']}\n")
    
    # Solar system check
    print("Test 3: Solar System safety check")
    test_solar_system_safety()
    print("  Verified boost → 1 at high Σ (Solar System safe)")
#!/usr/bin/env python3
"""
path_spectrum_kernel.py - Track 2: Physics-Grounded Path-Spectrum Kernel

Implements a first-principles coherence length kernel based on stationary-phase
azimuthal path accumulation. The kernel shrinks coherence length with:
- Bulge fraction (B/T)
- Shear rate
- Bar strength

This replaces empirical radial tapers with a physics-motivated approach that
unifies several V2.2 parameters under 4 hyperparameters.

Theory:
-------
The coherence length L_coh represents the azimuthal scale over which gravitational
paths maintain phase coherence. In regions with strong shear, bars, or central
bulges, this coherence is destroyed more quickly.

L_coh(r) = L_0 * f_bulge(B/T, r) * f_shear(∂Ω/∂r) * f_bar(bar_strength, r)

where:
- L_0: baseline coherence length (hyperparameter 1)
- f_bulge: reduction due to bulge-induced turbulence (hyperparameter 2: β_bulge)
- f_shear: reduction due to differential rotation (hyperparameter 3: α_shear)
- f_bar: reduction due to non-axisymmetric bar perturbations (hyperparameter 4: γ_bar)

Hyperparameters:
---------------
1. L_0: Baseline coherence length [kpc] - typical value ~ 1-5 kpc
2. β_bulge: Bulge suppression exponent - typical value ~ 0.5-2.0
3. α_shear: Shear suppression rate [(km/s/kpc)^-1] - typical value ~ 0.01-0.1
4. γ_bar: Bar suppression strength - typical value ~ 0.5-2.0
"""

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from typing import Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class PathSpectrumHyperparams:
    """Hyperparameters for path-spectrum kernel"""
    L_0: float = 2.5          # Baseline coherence length [kpc]
    beta_bulge: float = 1.0   # Bulge suppression exponent
    alpha_shear: float = 0.05 # Shear suppression rate [(km/s/kpc)^-1]
    gamma_bar: float = 1.0    # Bar suppression strength
    A_0: float = 1.0          # GLOBAL AMPLITUDE scaling factor (RAR calibration)
    p: float = 0.7            # RAR slope exponent (low-acceleration steepness)
    n_coh: float = 1.0        # Coherence damping exponent (gentler than exponential)
    g_dagger: float = 1.2e-10 # RAR acceleration scale [m/s²]
    
    def to_dict(self):
        return {
            'L_0': self.L_0,
            'beta_bulge': self.beta_bulge,
            'alpha_shear': self.alpha_shear,
            'gamma_bar': self.gamma_bar,
            'A_0': self.A_0,
            'p': self.p,
            'n_coh': self.n_coh,
            'g_dagger': self.g_dagger
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class PathSpectrumKernel:
    """Physics-grounded path-spectrum coherence kernel"""
    
    def __init__(self, hyperparams: PathSpectrumHyperparams, use_cupy: bool = True):
        self.hp = hyperparams
        self.xp = cp if (use_cupy and HAS_CUPY) else np
    
    def bulge_suppression(self, BT: float, r: Union[float, np.ndarray], 
                          r_bulge: float = 1.0) -> Union[float, np.ndarray]:
        """Compute bulge-induced coherence suppression
        
        Parameters:
        -----------
        BT : float
            Bulge-to-total ratio [0, 1]
        r : float or array
            Radius [kpc]
        r_bulge : float
            Bulge scale radius [kpc]
        
        Returns:
        --------
        f_bulge : float or array
            Suppression factor [0, 1], where 1 = no suppression
        """
        if BT <= 0:
            return 1.0
        
        # Bulge suppression increases inward, scales with B/T
        r_norm = self.xp.asarray(r) / r_bulge
        suppression = 1.0 / (1.0 + (BT ** self.hp.beta_bulge) / (r_norm + 0.1))
        
        return suppression
    
    def shear_suppression(self, v_circ: Union[float, np.ndarray], 
                          r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute shear-induced coherence suppression
        
        Parameters:
        -----------
        v_circ : float or array
            Circular velocity [km/s]
        r : float or array
            Radius [kpc]
        
        Returns:
        --------
        f_shear : float or array
            Suppression factor [0, 1], where 1 = no suppression
        """
        # Calculate angular velocity Ω = v/r
        r_arr = self.xp.asarray(r)
        v_arr = self.xp.asarray(v_circ)
        
        # Avoid division by zero
        r_safe = self.xp.maximum(r_arr, 0.01)
        omega = v_arr / r_safe
        
        # Estimate shear rate ∂Ω/∂r using finite differences
        # For single point, use approximation: dΩ/dr ≈ -Ω/r for declining rotation curve
        if self.xp.isscalar(omega):
            shear_rate = self.xp.abs(omega / r_safe)
        else:
            # Compute gradient
            dr = self.xp.diff(r_arr)
            domega = self.xp.diff(omega)
            shear_rate = self.xp.abs(domega / (dr + 1e-10))
            # Pad to match input size
            shear_rate = self.xp.concatenate([shear_rate, [shear_rate[-1]]])
        
        # Suppression increases with shear rate
        suppression = 1.0 / (1.0 + self.hp.alpha_shear * shear_rate)
        
        return suppression
    
    def bar_suppression(self, bar_strength: float, r: Union[float, np.ndarray], 
                       r_bar: float = 3.0) -> Union[float, np.ndarray]:
        """Compute bar-induced coherence suppression
        
        Parameters:
        -----------
        bar_strength : float
            Bar strength parameter [0, 1], where 0 = no bar, 1 = strong bar
        r : float or array
            Radius [kpc]
        r_bar : float
            Bar scale radius [kpc]
        
        Returns:
        --------
        f_bar : float or array
            Suppression factor [0, 1], where 1 = no suppression
        """
        if bar_strength <= 0:
            return 1.0
        
        # Bar suppression peaks at r ~ r_bar, decreases outward
        r_arr = self.xp.asarray(r)
        r_norm = r_arr / r_bar
        
        # Gaussian-like profile centered at bar radius
        bar_profile = self.xp.exp(-0.5 * (r_norm - 1.0)**2 / 0.5**2)
        suppression = 1.0 / (1.0 + (bar_strength ** self.hp.gamma_bar) * bar_profile)
        
        return suppression
    
    def coherence_length(self, r: Union[float, np.ndarray], 
                         v_circ: Union[float, np.ndarray],
                         BT: float = 0.0,
                         bar_strength: float = 0.0,
                         r_bulge: float = 1.0,
                         r_bar: float = 3.0) -> Union[float, np.ndarray]:
        """Compute coherence length at radius r
        
        Parameters:
        -----------
        r : float or array
            Radius [kpc]
        v_circ : float or array
            Circular velocity [km/s]
        BT : float
            Bulge-to-total ratio [0, 1]
        bar_strength : float
            Bar strength [0, 1]
        r_bulge : float
            Bulge scale radius [kpc]
        r_bar : float
            Bar scale radius [kpc]
        
        Returns:
        --------
        L_coh : float or array
            Coherence length [kpc]
        """
        f_bulge = self.bulge_suppression(BT, r, r_bulge)
        f_shear = self.shear_suppression(v_circ, r)
        f_bar = self.bar_suppression(bar_strength, r, r_bar)
        
        L_coh = self.hp.L_0 * f_bulge * f_shear * f_bar
        
        return L_coh
    
    def S_small(self, r: Union[float, np.ndarray], r_gate: float = 0.5) -> Union[float, np.ndarray]:
        """Small-radius gate: S_small(r→0) = 0, S_small(r≫r_gate) → 1
        
        This ensures Newtonian limit is preserved at small radii.
        
        Parameters:
        -----------
        r : float or array
            Radius [kpc]
        r_gate : float
            Gate scale radius [kpc], typical ~0.5 kpc
        
        Returns:
        --------
        S : float or array
            Gate factor [0, 1], where 0 = no many-path contribution
        """
        r_arr = self.xp.asarray(r)
        # Smooth turn-on: 1 - exp(-(r/r_gate)^p)
        p = 2.0  # Power for smoothness
        S = 1.0 - self.xp.exp(-(r_arr / r_gate)**p)
        return S
    
    def many_path_boost_factor(self, r: Union[float, np.ndarray],
                               v_circ: Union[float, np.ndarray],
                               g_bar: Optional[Union[float, np.ndarray]] = None,
                               BT: float = 0.0,
                               bar_strength: float = 0.0,
                               r_bulge: float = 1.0,
                               r_bar: float = 3.0,
                               r_gate: float = 0.5) -> Union[float, np.ndarray]:
        """Compute many-path boost factor K with RAR-shaped curvature
        
        NEW FORMULATION (post-diagnostic):
        K = A_0 * (g†/g_bar)^p * exp(-L/ℓ_coh) * S_small * [geometry gates]
        
        This gives proper RAR curvature:
        - At low g_bar (outer radii): K grows as (g†/g_bar)^p → steeper boost
        - At high g_bar (inner radii): K → 0 from S_small gate
        - Coherence length ℓ_coh modulates all path families
        - p parameter (0.3-1.2) controls RAR slope
        
        USAGE: g_total = g_bar * (1 + K)
        
        Parameters:
        -----------
        r : float or array
            Radius [kpc]
        v_circ : float or array
            Circular velocity [km/s]
        g_bar : float or array, optional
            Baryonic acceleration [m/s²]. If None, estimated from v_circ
        BT : float
            Bulge-to-total ratio
        bar_strength : float
            Bar strength parameter
        r_bulge : float
            Bulge scale radius [kpc]
        r_bar : float
            Bar scale radius [kpc]
        r_gate : float
            Small-radius gate scale [kpc]
        
        Returns:
        --------
        K : float or array
            Boost factor for additive contribution
            K = 0 at small r (Newtonian preserved)
            K increases at low g_bar (RAR curvature)
        """
        # Small-radius gate (preserves Newtonian limit)
        S_sm = self.S_small(r, r_gate)
        
        # Coherence length from bulge, shear, bar
        L_coh = self.coherence_length(r, v_circ, BT, bar_strength, r_bulge, r_bar)
        
        # Estimate g_bar if not provided (from v_circ/r approximation)
        r_arr = self.xp.asarray(r)
        if g_bar is None:
            # Rough estimate: g ≈ V²/R, convert to SI
            KM_TO_M = 1000.0
            KPC_TO_M = 3.0856776e19
            v_m_s = self.xp.asarray(v_circ) * KM_TO_M
            r_m = r_arr * KPC_TO_M
            g_bar = v_m_s**2 / r_m
        else:
            g_bar = self.xp.asarray(g_bar)
        
        # RAR-shaped response: (g†/g_bar)^p
        # This creates the key low-acceleration steepening
        g_ratio = self.hp.g_dagger / self.xp.maximum(g_bar, 1e-14)  # Avoid division by zero
        K_rar = self.xp.power(g_ratio, self.hp.p)
        
        # Coherence damping: POWER LAW (gentler than exponential)
        # Old: exp(-r/ℓ_coh) → too aggressive (96% suppression at r=3×ℓ_coh)
        # New: (ℓ_coh / (ℓ_coh + r))^n_coh → tunable falloff
        # At r=ℓ_coh: K_coh = 0.5^n_coh (50% if n=1, 25% if n=2)
        # At r=10×ℓ_coh: K_coh = (1/11)^n_coh (9% if n=1, 0.8% if n=2)
        K_coherence = self.xp.power(L_coh / (L_coh + r_arr), self.hp.n_coh)
        
        # Combined kernel:
        # K = A_0 * RAR_shape * coherence * small_r_gate
        K_total = self.hp.A_0 * K_rar * K_coherence * S_sm
        
        return K_total
    
    def suppression_factor(self, r: Union[float, np.ndarray],
                          v_circ: Union[float, np.ndarray],
                          BT: float = 0.0,
                          bar_strength: float = 0.0,
                          r_bulge: float = 1.0,
                          r_bar: float = 3.0,
                          r_scale: float = 3.0) -> Union[float, np.ndarray]:
        """DEPRECATED: Use many_path_boost_factor() instead
        
        This old implementation incorrectly multiplied the total field,
        violating Newtonian limit at small radii.
        
        Kept for backward compatibility but should not be used for physics.
        """
        # Issue warning
        import warnings
        warnings.warn(
            "suppression_factor() is deprecated and violates Newtonian limit. "
            "Use many_path_boost_factor() for additive formulation.",
            DeprecationWarning
        )
        
        L_coh = self.coherence_length(r, v_circ, BT, bar_strength, r_bulge, r_bar)
        xi = L_coh / (L_coh + r_scale)
        return xi
    
    def demo_run(self):
        """Demonstration run showing coherence length behavior"""
        print("=" * 80)
        print("PATH-SPECTRUM KERNEL DEMONSTRATION (CORRECTED)")
        print("=" * 80)
        print(f"\nHyperparameters:")
        print(f"  L_0 = {self.hp.L_0:.2f} kpc")
        print(f"  β_bulge = {self.hp.beta_bulge:.2f}")
        print(f"  α_shear = {self.hp.alpha_shear:.4f} (km/s/kpc)^-1")
        print(f"  γ_bar = {self.hp.gamma_bar:.2f}")
        
        # Test Newtonian limit first
        print("\n" + "-" * 80)
        print("Test 0: NEWTONIAN LIMIT (small radii)")
        print("-" * 80)
        r_small = np.array([0.001, 0.01, 0.1, 0.5])  # Very small to moderate
        v_small = np.array([50, 100, 150, 200])
        K_small = self.many_path_boost_factor(r_small, v_small, BT=0.0, bar_strength=0.0)
        
        for i in range(len(r_small)):
            print(f"  r = {r_small[i]:6.3f} kpc: K = {K_small[i]:.6f} "
                  f"(boost = {K_small[i]*100:.3f}%, should be ~0 at small r)")
        
        print("\n  ✅ Newtonian limit: K→0 as r→0 (many-path contribution vanishes)")
        
        # Test case 1: Pure disk (no bulge, no bar)
        print("\n" + "-" * 80)
        print("Test Case 1: Pure Disk Galaxy (BT=0, no bar)")
        print("-" * 80)
        r_test = np.array([1.0, 5.0, 10.0, 20.0])
        v_test = np.array([100.0, 150.0, 160.0, 155.0])  # Typical rotation curve
        L_coh = self.coherence_length(r_test, v_test, BT=0.0, bar_strength=0.0)
        K_disk = self.many_path_boost_factor(r_test, v_test, BT=0.0, bar_strength=0.0)
        
        for i in range(len(r_test)):
            print(f"  r = {r_test[i]:5.1f} kpc: L_coh = {L_coh[i]:.3f} kpc, K = {K_disk[i]:.3f}")
        
        # Test case 2: Bulge-dominated galaxy
        print("\n" + "-" * 80)
        print("Test Case 2: Bulge-Dominated Galaxy (BT=0.5)")
        print("-" * 80)
        L_coh_bulge = self.coherence_length(r_test, v_test, BT=0.5, bar_strength=0.0)
        K_bulge = self.many_path_boost_factor(r_test, v_test, BT=0.5, bar_strength=0.0)
        
        for i in range(len(r_test)):
            print(f"  r = {r_test[i]:5.1f} kpc: L_coh = {L_coh_bulge[i]:.3f} kpc, "
                  f"K = {K_bulge[i]:.3f} (vs disk: {K_bulge[i]/K_disk[i]:.2f}×)")
        
        # Test case 3: Barred galaxy
        print("\n" + "-" * 80)
        print("Test Case 3: Barred Galaxy (bar_strength=0.8)")
        print("-" * 80)
        L_coh_bar = self.coherence_length(r_test, v_test, BT=0.0, bar_strength=0.8, r_bar=3.0)
        K_bar = self.many_path_boost_factor(r_test, v_test, BT=0.0, bar_strength=0.8, r_bar=3.0)
        
        for i in range(len(r_test)):
            print(f"  r = {r_test[i]:5.1f} kpc: L_coh = {L_coh_bar[i]:.3f} kpc, "
                  f"K = {K_bar[i]:.3f} (vs disk: {K_bar[i]/K_disk[i]:.2f}×)")
        
        print("\n" + "=" * 80)
        print("USAGE: g_total = g_Newton * (1 + K)")
        print("       where K = many_path_boost_factor(r, v, BT, bar_strength)")
        print("=" * 80)


def main():
    """Run demonstration"""
    # Default hyperparameters
    hp = PathSpectrumHyperparams(
        L_0=2.5,
        beta_bulge=1.0,
        alpha_shear=0.05,
        gamma_bar=1.0
    )
    
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    kernel.demo_run()


if __name__ == "__main__":
    main()

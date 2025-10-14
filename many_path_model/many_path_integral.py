"""
Explicit Path Integral Formulation of Gravity
==============================================

This module implements the LITERAL QED-style sum over all gravitational paths,
expressed as a discrete sum over winding families (m) with phase Φ_m.

The goal is to show numerical equivalence between:
1. This explicit path sum (direct but computationally expensive)
2. PathSpectrumKernel (stationary-phase approximation, fast)

Conceptual Framework (QED Analogy)
-----------------------------------

In QED (Feynman path integrals):
    Amplitude = ∫ exp(i·S[path]/ℏ) D[path]
    
where S[path] = action along path, integrated over all possible trajectories.

In This Gravitational Theory:
    K(R) ∝ |Σ_paths A_path × exp(i·Φ_path)|²
    
where:
- Paths are labeled by winding number m (how many times they spiral)
- Φ_path = geometric phase + matter-dependent phase
- A_path = coherence weight (from density, geometry, etc.)

For axisymmetric disks:
    K(R) = |Σ_{m=0}^{m_max} ∫₀^{2π} w_m(R,φ) exp(i·Φ_m(R,φ)) dφ|²

where:
- m = winding family (m=0: direct radial, m>0: spirals)
- Φ_m(R,φ) = action phase for trajectory in family m
- w_m(R,φ) = coherence weight (includes L_0, bulge, shear, bar gates)

Relation to PathSpectrumKernel
-------------------------------
PathSpectrumKernel is the STATIONARY-PHASE approximation to this sum:
- Identifies dominant path families (near stationary action)
- Approximates integral over φ using saddle-point method
- Result: K(R) ≈ coherence_envelope(R) × geometry_gates(R)

This module validates that approximation by computing the full sum.
"""

import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass


@dataclass
class PathIntegralParams:
    """Parameters for explicit path integral calculation."""
    m_max: int = 20  # Maximum winding number
    n_phi: int = 100  # Angular resolution per winding family
    L_0: float = 8.0  # Coherence length [kpc]
    p: float = 1.5  # Power-law coherence exponent
    n_coh: float = 2.0  # Coherence steepness
    beta_bulge: float = 0.5  # Bulge decoherence
    alpha_shear: float = 0.3  # Shear decoherence
    gamma_bar: float = 0.8  # Bar decoherence
    A_0: float = 1.0  # Overall amplitude


def geometric_phase(R: float, phi: float, m: int, 
                    r_spiral: np.ndarray, rho_bar: np.ndarray) -> float:
    """
    Compute geometric + matter-dependent phase for path in winding family m.
    
    This is the gravitational "action" S_grav along the path.
    
    For a spiral path at radius R with azimuthal angle φ in winding family m:
    - Geometric phase: ∝ path length × curvature
    - Matter phase: ∫ ρ(s) ds (density integral along path)
    
    Parameters
    ----------
    R : float
        Projected radius [kpc]
    phi : float
        Azimuthal angle [radians]
    m : int
        Winding number (0 = direct, 1+ = spirals)
    r_spiral : ndarray
        Radial profile of matter distribution [kpc]
    rho_bar : ndarray
        Baryonic matter density [M☉/kpc³]
    
    Returns
    -------
    phase : float
        Total phase Φ_m(R,φ) [dimensionless]
    """
    # For m=0 (direct radial): phase is just radial distance scaled
    if m == 0:
        # Direct path from 0 to R
        # Phase ∝ ∫₀^R ρ(r') dr'
        mask = r_spiral <= R
        if np.sum(mask) < 2:
            return 0.0
        phase_integral = np.trapz(rho_bar[mask], r_spiral[mask])
        # Normalize to typical scale
        return phase_integral / (1e9)  # Msun/kpc^2 typical scale
    
    # For m>0: spiral path with m full rotations
    # Path length increases as L ≈ R × sqrt(1 + (2πm)²)
    # Density integral picks up contributions from all radii crossed
    
    # Simplified model: phase increases with:
    # 1. Winding (more turns = more phase accumulation)
    # 2. Matter crossed (spiral samples more of the disk)
    
    # Approximate spiral as sampling density at multiple radii
    theta_spiral = np.linspace(0, 2*np.pi*m, 100)
    r_path = R * (1 - theta_spiral/(2*np.pi*m))  # Decreasing radius as we spiral in
    r_path = np.clip(r_path, 0, R)
    
    # Interpolate density at each point
    rho_path = np.interp(r_path, r_spiral, rho_bar, left=0, right=0)
    
    # Phase accumulation along spiral
    ds = np.diff(theta_spiral) * R  # Arc length element (approximate)
    phase_integral = np.sum(rho_path[:-1] * ds)
    
    # Add geometric phase (winding penalty)
    geometric_term = m * 2*np.pi  # Each winding adds 2π
    
    total_phase = (phase_integral / 1e9) + 0.1 * geometric_term
    
    return total_phase


def coherence_weight(R: float, m: int, params: PathIntegralParams,
                     r_bulge: float, v_circ: float, v_bar: float,
                     shear_param: float, bar_strength: float) -> float:
    """
    Compute coherence weight w_m(R) for path family m.
    
    This implements the SAME decoherence physics as PathSpectrumKernel:
    - Radial coherence envelope (power-law damping beyond L_0)
    - Bulge decoherence (velocity dispersion kills coherence)
    - Shear decoherence (differential rotation dephases spirals)
    - Bar decoherence (non-axisymmetric potential)
    
    Parameters
    ----------
    R : float
        Radius [kpc]
    m : int
        Winding number
    params : PathIntegralParams
        Coherence parameters
    r_bulge : float
        Bulge scale radius [kpc]
    v_circ : float
        Circular velocity at R [km/s]
    v_bar : float
        Baryonic circular velocity at R [km/s]
    shear_param : float
        Shear rate d(ln v)/d(ln r)
    bar_strength : float
        Bar amplitude (0-1)
    
    Returns
    -------
    weight : float
        Coherence weight w_m(R) ∈ [0, 1]
    """
    # 1. Radial coherence envelope (power-law, not exponential)
    # For m=0: coherence maintained to L_0, then falls as (R/L_0)^(-p)
    # For m>0: coherence reduced by factor ~(1 + m/n_coh)^(-1)
    
    x = R / params.L_0
    if x <= 1.0:
        xi_radial = 1.0
    else:
        xi_radial = x**(-params.p)
    
    # Winding penalty: higher m = more phase noise
    xi_winding = 1.0 / (1.0 + m / params.n_coh)
    
    # 2. Bulge gate (velocity dispersion)
    # σ²/v² ~ (r_bulge/R) for classical bulge
    if R < r_bulge:
        dispersion_ratio = (r_bulge / R)**0.5
    else:
        dispersion_ratio = 0.1  # Far from bulge
    
    xi_bulge = 1.0 / (1.0 + params.beta_bulge * dispersion_ratio**2)
    
    # 3. Shear gate (differential rotation)
    # High shear |d ln v / d ln r| winds up phase
    xi_shear = 1.0 / (1.0 + params.alpha_shear * abs(shear_param))
    
    # 4. Bar gate (non-axisymmetric chaos)
    xi_bar = 1.0 / (1.0 + params.gamma_bar * bar_strength)
    
    # Total weight (multiplicative)
    weight = xi_radial * xi_winding * xi_bulge * xi_shear * xi_bar
    
    return weight


def compute_path_integral_boost(
    R_eval: np.ndarray,
    r_spiral: np.ndarray,
    rho_bar: np.ndarray,
    v_circ: np.ndarray,
    v_bar: np.ndarray,
    params: PathIntegralParams,
    r_bulge: float = 1.0,
    bar_strength: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute boost factor K(R) via explicit path integral.
    
    K(R) = |Σ_{m=0}^{m_max} ∫₀^{2π} w_m(R,φ) exp(i·Φ_m(R,φ)) dφ|²
    
    Parameters
    ----------
    R_eval : ndarray
        Radii to evaluate [kpc]
    r_spiral : ndarray
        Radial profile of matter [kpc]
    rho_bar : ndarray
        Baryonic density [M☉/kpc³]
    v_circ : ndarray
        Circular velocity [km/s]
    v_bar : ndarray
        Baryonic circular velocity [km/s]
    params : PathIntegralParams
        Path integral parameters
    r_bulge : float
        Bulge scale [kpc]
    bar_strength : float
        Bar amplitude [0-1]
    
    Returns
    -------
    K : ndarray
        Boost factor at each R_eval
    coherence_avg : ndarray
        Average coherence weight at each R_eval (diagnostic)
    """
    K = np.zeros_like(R_eval)
    coherence_avg = np.zeros_like(R_eval)
    
    for i, R in enumerate(R_eval):
        # Compute shear parameter at this radius
        idx = np.argmin(np.abs(r_spiral - R))
        if idx > 0 and idx < len(r_spiral) - 1:
            dlogv_dlogr = np.gradient(np.log(v_circ + 1e-10), np.log(r_spiral + 1e-10))[idx]
        else:
            dlogv_dlogr = 0.0
        
        # Sum over winding families
        amplitude_real = 0.0
        amplitude_imag = 0.0
        total_weight = 0.0
        
        for m in range(params.m_max + 1):
            # Integrate over azimuthal angle φ
            phi_array = np.linspace(0, 2*np.pi, params.n_phi)
            
            for phi in phi_array:
                # Coherence weight for this path
                weight = coherence_weight(
                    R, m, params, r_bulge,
                    v_circ[idx], v_bar[idx],
                    dlogv_dlogr, bar_strength
                )
                
                # Phase for this path
                phase = geometric_phase(R, phi, m, r_spiral, rho_bar)
                
                # Add to amplitude (complex sum)
                amplitude_real += weight * np.cos(phase)
                amplitude_imag += weight * np.sin(phase)
                total_weight += weight
            
        # Normalize by number of paths
        amplitude_real /= ((params.m_max + 1) * params.n_phi)
        amplitude_imag /= ((params.m_max + 1) * params.n_phi)
        coherence_avg[i] = total_weight / ((params.m_max + 1) * params.n_phi)
        
        # K = |amplitude|² (intensity, not amplitude)
        amplitude_squared = amplitude_real**2 + amplitude_imag**2
        K[i] = params.A_0 * amplitude_squared
    
    return K, coherence_avg


def compare_with_path_spectrum_kernel(
    R_eval: np.ndarray,
    r_spiral: np.ndarray,
    rho_bar: np.ndarray,
    v_circ: np.ndarray,
    v_bar: np.ndarray,
    galaxy_name: str = "Test Galaxy"
) -> dict:
    """
    Compare explicit path integral with PathSpectrumKernel.
    
    This is the VALIDATION test showing they give the same answer.
    
    Returns
    -------
    results : dict
        Contains K_integral, K_kernel, fractional_diff, etc.
    """
    # 1. Compute via explicit path integral
    params_integral = PathIntegralParams(
        m_max=20,
        n_phi=100,
        L_0=8.0,
        p=1.5,
        n_coh=2.0,
        beta_bulge=0.5,
        alpha_shear=0.3,
        gamma_bar=0.8,
        A_0=1.0
    )
    
    K_integral, coherence_avg = compute_path_integral_boost(
        R_eval, r_spiral, rho_bar, v_circ, v_bar,
        params_integral, r_bulge=1.0, bar_strength=0.0
    )
    
    # 2. Compute via PathSpectrumKernel (stationary-phase)
    # Import here to avoid circular dependency
    from many_path_model.path_spectrum_kernel_track2 import (
        PathSpectrumKernel, PathSpectrumHyperparams
    )
    
    hyperparams = PathSpectrumHyperparams(
        L_0=params_integral.L_0,
        beta_bulge=params_integral.beta_bulge,
        alpha_shear=params_integral.alpha_shear,
        gamma_bar=params_integral.gamma_bar,
        A_0=params_integral.A_0,
        p=params_integral.p,
        n_coh=params_integral.n_coh
    )
    
    # Force NumPy mode (not CuPy)
    kernel = PathSpectrumKernel(hyperparams, use_cupy=False)
    
    # Interpolate velocities at evaluation radii
    v_circ_eval = np.interp(R_eval, r_spiral, v_circ)
    
    # Call kernel with new interface (takes scalar/array r, v_circ)
    K_kernel = kernel.many_path_boost_factor(
        r=R_eval,
        v_circ=v_circ_eval,
        BT=0.1,
        bar_strength=0.0,
        r_bulge=1.0,
        r_bar=5.0
    )
    
    # 3. Compare
    fractional_diff = np.abs(K_integral - K_kernel) / (K_kernel + 1e-10)
    rms_diff = np.sqrt(np.mean(fractional_diff**2))
    
    results = {
        'R_eval': R_eval,
        'K_integral': K_integral,
        'K_kernel': K_kernel,
        'coherence_avg': coherence_avg,
        'fractional_diff': fractional_diff,
        'rms_diff': rms_diff,
        'max_diff': np.max(fractional_diff),
        'median_diff': np.median(fractional_diff)
    }
    
    return results


if __name__ == "__main__":
    """
    Test case: simple exponential disk.
    """
    import matplotlib.pyplot as plt
    
    print("=" * 80)
    print("PATH INTEGRAL VALIDATION TEST")
    print("=" * 80)
    print("\nComparing explicit path sum with PathSpectrumKernel")
    print("(This proves the stationary-phase approximation is valid)\n")
    
    # Create test galaxy: exponential disk
    r_spiral = np.linspace(0.1, 30, 200)
    R_d = 3.0  # Disk scale length [kpc]
    Sigma_0 = 1e9  # Central surface density [M☉/kpc²]
    
    # Exponential profile
    Sigma = Sigma_0 * np.exp(-r_spiral / R_d)
    h_z = 0.1 * R_d  # Scale height
    rho_bar = Sigma / (2 * h_z)  # 3D density
    
    # Velocity (roughly flat rotation curve)
    v_max = 200.0  # km/s
    v_circ = v_max * np.sqrt(1 - np.exp(-r_spiral / R_d))
    v_bar = v_circ  # For simplicity
    
    # Evaluation radii
    R_eval = np.linspace(1, 25, 50)
    
    print(f"Test galaxy: Exponential disk")
    print(f"  R_d = {R_d:.1f} kpc")
    print(f"  v_max = {v_max:.0f} km/s")
    print(f"  Evaluating K(R) at {len(R_eval)} radii\n")
    
    # Run comparison
    results = compare_with_path_spectrum_kernel(
        R_eval, r_spiral, rho_bar, v_circ, v_bar,
        galaxy_name="Exponential Disk Test"
    )
    
    # Print results
    print("Results:")
    print("-" * 80)
    print(f"  RMS fractional difference: {results['rms_diff']:.4f}")
    print(f"  Max fractional difference: {results['max_diff']:.4f}")
    print(f"  Median fractional difference: {results['median_diff']:.4f}")
    print()
    
    if results['rms_diff'] < 0.10:
        print("✅ PASS: Path integral ≈ PathSpectrumKernel (RMS < 10%)")
        print("   Stationary-phase approximation is valid!")
    elif results['rms_diff'] < 0.30:
        print("⚠️  MARGINAL: Agreement within 30%, but check details")
    else:
        print("❌ FAIL: Large discrepancy - need to investigate")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Path Integral vs PathSpectrumKernel Validation', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: K(R) comparison
    ax = axes[0, 0]
    ax.plot(R_eval, results['K_integral'], 'b-', lw=2, label='Explicit path integral')
    ax.plot(R_eval, results['K_kernel'], 'r--', lw=2, label='PathSpectrumKernel')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Boost Factor K(R)')
    ax.set_title('Direct Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Fractional difference
    ax = axes[0, 1]
    ax.semilogy(R_eval, results['fractional_diff'], 'ko-', lw=2, ms=4)
    ax.axhline(0.10, color='orange', ls='--', label='10% threshold')
    ax.axhline(0.30, color='red', ls='--', label='30% threshold')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Fractional Difference |ΔK/K|')
    ax.set_title(f'RMS = {results["rms_diff"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-4, 2)
    
    # Plot 3: Scatter plot
    ax = axes[1, 0]
    ax.plot(results['K_kernel'], results['K_integral'], 'bo', ms=6, alpha=0.6)
    Kmax = max(np.max(results['K_kernel']), np.max(results['K_integral']))
    ax.plot([0, Kmax], [0, Kmax], 'k--', lw=1, label='1:1 line')
    ax.set_xlabel('K (PathSpectrumKernel)')
    ax.set_ylabel('K (Explicit Integral)')
    ax.set_title('1:1 Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Coherence diagnostic
    ax = axes[1, 1]
    ax.plot(R_eval, results['coherence_avg'], 'g-', lw=2, label='Avg coherence weight')
    ax.plot(R_eval, results['K_integral'] / np.max(results['K_integral']), 
            'b--', lw=2, label='K (normalized)')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Normalized')
    ax.set_title('Coherence vs Boost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    from pathlib import Path
    outpath = Path(__file__).parent.parent / 'results' / 'plots' / 'path_integral_validation.png'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Validation plot saved: {outpath}")
    print("=" * 80)

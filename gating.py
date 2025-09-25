#!/usr/bin/env python3
"""
C² continuous gating functions for G³ model
Implements smootherstep and cubic spline for perfectly smooth transitions
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

def smootherstep(x):
    """
    C² continuous smootherstep function
    Maps [0,1] → [0,1] with zero derivatives at boundaries
    
    s(x) = 6x^5 - 15x^4 + 10x^3
    s'(x) = 30x^4 - 60x^3 + 30x^2 = 30x^2(x-1)^2
    s''(x) = 120x^3 - 180x^2 + 60x = 60x(2x^2 - 3x + 1)
    
    Both s'(0)=s'(1)=0 and s''(0)=s''(1)=0
    """
    xp = cp if GPU_AVAILABLE and isinstance(x, cp.ndarray) else np
    
    # Clamp to [0, 1]
    x = xp.clip(x, 0.0, 1.0)
    
    # Smootherstep polynomial
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

def cubic_hermite_spline(x, x0, x1, y0, y1, dy0=0.0, dy1=0.0):
    """
    C¹ continuous cubic Hermite spline interpolation
    
    Parameters:
    -----------
    x : array
        Evaluation points
    x0, x1 : float
        Start and end x-coordinates
    y0, y1 : float
        Start and end y-values
    dy0, dy1 : float
        Start and end derivatives
        
    Returns:
    --------
    y : array
        Interpolated values
    """
    xp = cp if GPU_AVAILABLE and isinstance(x, cp.ndarray) else np
    
    # Normalize to [0, 1]
    t = xp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    
    # Hermite basis functions
    h00 = (1.0 + 2.0*t) * (1.0 - t)**2
    h10 = t * (1.0 - t)**2
    h01 = t**2 * (3.0 - 2.0*t)
    h11 = t**2 * (t - 1.0)
    
    # Scale derivatives by interval width
    scale = x1 - x0
    
    # Interpolate
    return h00 * y0 + h10 * scale * dy0 + h01 * y1 + h11 * scale * dy1

def gate_and_exponent(Sigma_loc, params):
    """
    Compute C² continuous gate and exponent from surface density
    
    Parameters:
    -----------
    Sigma_loc : array
        Local surface density (Msun/pc^2)
    params : dict
        Model parameters including:
        - Sigma_star: transition density
        - w_S: gate width (in log space)
        - p_in, p_out: inner/outer exponents
        - Sigma_in, Sigma_out: exponent transition range
        - w_p: exponent width
        
    Returns:
    --------
    S : array
        Gate function (0 → 1)
    p : array
        Power law exponent
    """
    xp = cp if GPU_AVAILABLE and isinstance(Sigma_loc, cp.ndarray) else np
    
    # Extract parameters
    Sigma_star = params.get('Sigma_star', 50.0)
    w_S = params.get('w_S', 2.0)
    p_in = params.get('p_in', 2.0)
    p_out = params.get('p_out', 1.0)
    
    # Transition densities for exponent
    Sigma_in = params.get('Sigma_in', Sigma_star * 2.0)
    Sigma_out = params.get('Sigma_out', Sigma_star * 0.5)
    
    # Compute gate using smootherstep
    log_Sigma = xp.log(xp.maximum(Sigma_loc, 1e-8))
    log_Sigma_star = xp.log(xp.maximum(Sigma_star, 1e-8))
    
    # Normalize to [0, 1] for smootherstep
    x_gate = (log_Sigma - (log_Sigma_star - w_S)) / (2.0 * w_S)
    S = smootherstep(x_gate)
    
    # Compute exponent using cubic spline in log space
    log_Sigma_in = xp.log(Sigma_in)
    log_Sigma_out = xp.log(Sigma_out)
    
    # Smooth derivatives for C¹ continuity of exponent
    dp_in = params.get('dp_in', 0.0)  # derivative at inner edge
    dp_out = params.get('dp_out', 0.0)  # derivative at outer edge
    
    # Below outer density: use p_out
    # Between outer and inner: spline interpolation
    # Above inner density: use p_in
    
    p = xp.where(
        log_Sigma <= log_Sigma_out,
        p_out,
        xp.where(
            log_Sigma >= log_Sigma_in,
            p_in,
            cubic_hermite_spline(log_Sigma, log_Sigma_out, log_Sigma_in, 
                               p_out, p_in, dp_out, dp_in)
        )
    )
    
    # Ensure p is bounded and positive
    p = xp.clip(p, 0.5, 3.0)
    
    return S, p

def curvature_gate(Sigma_loc, R, params):
    """
    Optional curvature-sensitive gate enhancement
    
    Parameters:
    -----------
    Sigma_loc : array
        Local surface density
    R : array
        Radial positions (kpc)
    params : dict
        Including eta_C (curvature weight) and C_0 (curvature threshold)
        
    Returns:
    --------
    C_factor : array
        Curvature enhancement factor (≥1)
    """
    xp = cp if GPU_AVAILABLE and isinstance(Sigma_loc, cp.ndarray) else np
    
    eta_C = params.get('eta_C', 0.0)
    C_0 = params.get('C_0', 1.0)
    
    if eta_C == 0:
        return xp.ones_like(Sigma_loc)
    
    # Compute logarithmic curvature
    log_Sigma = xp.log(xp.maximum(Sigma_loc, 1e-8))
    log_R = xp.log(xp.maximum(R, 1e-6))
    
    # Second derivative using finite differences
    if len(R) > 2:
        # Pad for edge handling
        log_Sigma_pad = xp.pad(log_Sigma, 1, mode='edge')
        log_R_pad = xp.pad(log_R, 1, mode='edge')
        
        # Central differences for first derivative
        dlogS_dlogR = (log_Sigma_pad[2:] - log_Sigma_pad[:-2]) / \
                      (log_R_pad[2:] - log_R_pad[:-2] + 1e-10)
        
        # Second derivative
        d2logS_dlogR2 = xp.zeros_like(Sigma_loc)
        if len(R) > 3:
            d2logS_dlogR2[1:-1] = (dlogS_dlogR[2:] - dlogS_dlogR[:-2]) / \
                                  (log_R[2:] - log_R[:-2] + 1e-10)
    else:
        d2logS_dlogR2 = xp.zeros_like(Sigma_loc)
    
    # Curvature magnitude
    C = xp.abs(d2logS_dlogR2)
    
    # Softplus activation for smooth turn-on
    def softplus(x):
        return xp.log(1.0 + xp.exp(xp.clip(x, -20, 20)))
    
    C_factor = 1.0 + eta_C * softplus(C - C_0)
    
    return C_factor

def gradient_screening(Sigma_loc, R, params):
    """
    Gradient-sensitive screening enhancement
    
    Parameters:
    -----------
    Sigma_loc : array
        Local surface density
    R : array
        Radial positions
    params : dict
        Including grad_weight and grad_scale
        
    Returns:
    --------
    G_factor : array
        Gradient screening factor (0 → 1)
    """
    xp = cp if GPU_AVAILABLE and isinstance(Sigma_loc, cp.ndarray) else np
    
    grad_weight = params.get('grad_weight', 0.0)
    grad_scale = params.get('grad_scale', 1.0)
    
    if grad_weight == 0:
        return xp.ones_like(Sigma_loc)
    
    # Compute gradient magnitude
    if len(R) > 1:
        grad_Sigma = xp.gradient(Sigma_loc, R)
        grad_mag = xp.abs(grad_Sigma)
    else:
        grad_mag = xp.zeros_like(Sigma_loc)
    
    # Normalize gradient
    grad_norm = grad_mag / (xp.maximum(Sigma_loc, 1e-8) + grad_scale)
    
    # Apply smootherstep to gradient magnitude
    x_grad = xp.clip(grad_norm / grad_weight, 0.0, 1.0)
    G_factor = 1.0 - smootherstep(x_grad)  # High gradient → low factor
    
    return G_factor

def test_continuity():
    """
    Test C² continuity of gating functions
    """
    import matplotlib.pyplot as plt
    
    # Test parameters
    params = {
        'Sigma_star': 50.0,
        'w_S': 2.0,
        'p_in': 2.5,
        'p_out': 1.0,
        'Sigma_in': 100.0,
        'Sigma_out': 25.0,
        'dp_in': 0.0,
        'dp_out': 0.0
    }
    
    # Create test surface density profile
    Sigma = np.logspace(0, 3, 1000)  # 1 to 1000 Msun/pc^2
    
    # Compute gate and exponent
    S, p = gate_and_exponent(Sigma, params)
    
    # Compute derivatives numerically
    dS = np.gradient(S, np.log(Sigma))
    d2S = np.gradient(dS, np.log(Sigma))
    
    dp = np.gradient(p, np.log(Sigma))
    d2p = np.gradient(dp, np.log(Sigma))
    
    # Check continuity
    max_jump_S = np.max(np.abs(np.diff(S)))
    max_jump_dS = np.max(np.abs(np.diff(dS)))
    max_jump_d2S = np.max(np.abs(np.diff(d2S)))
    
    max_jump_p = np.max(np.abs(np.diff(p)))
    max_jump_dp = np.max(np.abs(np.diff(dp)))
    
    print("Continuity Test Results:")
    print(f"Gate S:   max jump = {max_jump_S:.2e}")
    print(f"Gate S':  max jump = {max_jump_dS:.2e}")
    print(f"Gate S'': max jump = {max_jump_d2S:.2e}")
    print(f"Exponent p:  max jump = {max_jump_p:.2e}")
    print(f"Exponent p': max jump = {max_jump_dp:.2e}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gate function
    ax = axes[0, 0]
    ax.semilogx(Sigma, S, 'b-', linewidth=2)
    ax.set_xlabel('Σ (M☉/pc²)')
    ax.set_ylabel('Gate S')
    ax.set_title('Gate Function')
    ax.grid(True, alpha=0.3)
    
    # Gate first derivative
    ax = axes[0, 1]
    ax.semilogx(Sigma, dS, 'g-', linewidth=2)
    ax.set_xlabel('Σ (M☉/pc²)')
    ax.set_ylabel("S'")
    ax.set_title('Gate First Derivative')
    ax.grid(True, alpha=0.3)
    
    # Gate second derivative
    ax = axes[0, 2]
    ax.semilogx(Sigma, d2S, 'r-', linewidth=2)
    ax.set_xlabel('Σ (M☉/pc²)')
    ax.set_ylabel("S''")
    ax.set_title('Gate Second Derivative')
    ax.grid(True, alpha=0.3)
    
    # Exponent function
    ax = axes[1, 0]
    ax.semilogx(Sigma, p, 'b-', linewidth=2)
    ax.set_xlabel('Σ (M☉/pc²)')
    ax.set_ylabel('Exponent p')
    ax.set_title('Power Law Exponent')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 3.0])
    
    # Exponent first derivative
    ax = axes[1, 1]
    ax.semilogx(Sigma, dp, 'g-', linewidth=2)
    ax.set_xlabel('Σ (M☉/pc²)')
    ax.set_ylabel("p'")
    ax.set_title('Exponent First Derivative')
    ax.grid(True, alpha=0.3)
    
    # Exponent second derivative
    ax = axes[1, 2]
    ax.semilogx(Sigma, d2p, 'r-', linewidth=2)
    ax.set_xlabel('Σ (M☉/pc²)')
    ax.set_ylabel("p''")
    ax.set_title('Exponent Second Derivative')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/continuity_test.png', dpi=150)
    print(f"\nPlot saved to out/continuity_test.png")
    
    # Return True if C² continuous (very small jumps)
    is_c2 = (max_jump_S < 1e-10 and max_jump_dS < 1e-8 and max_jump_d2S < 1e-4)
    is_c1 = (max_jump_p < 1e-10 and max_jump_dp < 1e-6)
    
    if is_c2:
        print("\n✅ Gate is C² continuous!")
    else:
        print("\n⚠️ Gate has discontinuities")
        
    if is_c1:
        print("✅ Exponent is C¹ continuous!")
    else:
        print("⚠️ Exponent has discontinuities")
    
    return is_c2 and is_c1

if __name__ == '__main__':
    import os
    os.makedirs('out', exist_ok=True)
    test_continuity()
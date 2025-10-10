#!/usr/bin/env python3
"""
Piggybacking Evaluation: Quantify Baryon Magnification
=======================================================

Diagnostic tools to measure and visualize how the cooperative response
mechanism "magnifies" existing baryons through non-local mass export.

Key Metrics:
    1. Baryon magnification: B_κ(R) = κ_eff / κ_bar
    2. Piggyback fraction: f_resp(R) = Σ_resp / Σ_eff
    3. Donor→recipient matrix: C_ij showing mass flow
    4. Hard outcomes: θ_E error, RMS α(θ) error

Author: AI Assistant + User
Date: 2025-01-10
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path

def baryon_magnification(R: np.ndarray,
                        Sigma_eff: np.ndarray,
                        Sigma_bar: np.ndarray,
                        Sigma_crit: float = 1.0) -> np.ndarray:
    """
    Compute baryon magnification factor B_κ(R).
    
    B_κ(R) = κ_eff(R) / κ_bar(R) = Σ_eff(R) / Σ_bar(R)
    
    This shows how much the effective convergence exceeds the baryon-only
    convergence at each radius.
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    Sigma_eff : np.ndarray
        Effective surface density (baryons + response) in Msun/kpc²
    Sigma_bar : np.ndarray
        Baryon-only surface density in Msun/kpc²
    Sigma_crit : float
        Critical surface density (if provided, returns true κ; else proportional)
        
    Returns
    -------
    np.ndarray
        Magnification factor B_κ (dimensionless, ≥ 1)
    """
    kappa_bar = Sigma_bar / Sigma_crit
    kappa_eff = Sigma_eff / Sigma_crit
    
    B_kappa = np.divide(kappa_eff, np.maximum(kappa_bar, 1e-30))
    
    return B_kappa


def piggyback_fraction(Sigma_resp: np.ndarray,
                      Sigma_eff: np.ndarray) -> np.ndarray:
    """
    Compute fraction of effective mass that is "piggybacked" response.
    
    f_resp(R) = Σ_resp(R) / Σ_eff(R)
    
    Parameters
    ----------
    Sigma_resp : np.ndarray
        Response surface density in Msun/kpc²
    Sigma_eff : np.ndarray
        Effective surface density in Msun/kpc²
        
    Returns
    -------
    np.ndarray
        Piggyback fraction (0 to 1, where 0=pure baryons, 1=all response)
    """
    return np.divide(Sigma_resp, np.maximum(Sigma_eff, 1e-30))


def donor_recipient_matrix(R: np.ndarray,
                           Sigma: np.ndarray,
                           A_resp: float,
                           lam: float,
                           nu: float,
                           w_don: np.ndarray,
                           w_rec: np.ndarray) -> np.ndarray:
    """
    Build donor→recipient contribution matrix C_ij.
    
    C_ij represents the contribution from donor ring j to recipient ring i,
    such that:
        Σ_resp(R_i) = Σ_j C_ij · Σ(R_j) · ΔA_j
    
    This lets you audit where outskirts mass comes from.
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    Sigma : np.ndarray
        Baryon surface density in Msun/kpc²
    A_resp : float
        Response amplitude
    lam : float
        Export scale in kpc
    nu : float
        Kernel tail index
    w_don : np.ndarray
        Donor weights
    w_rec : np.ndarray
        Recipient weights
        
    Returns
    -------
    np.ndarray
        Matrix C_ij (shape: len(R) × len(R))
        Units: dimensionless (per unit area)
    """
    from scripts.cooperative_response import power_kernel
    
    dR = np.gradient(R)
    dA = 2.0 * np.pi * R * dR  # Area of rings
    
    # Kernel matrix
    dR_matrix = np.abs(R[:, None] - R[None, :])
    K = power_kernel(dR_matrix, lam=lam, nu=nu)
    
    # C_ij: contribution from j to i (per unit Σ_j)
    # Recipient gate at i, donor gate at j, kernel between, area factor at j
    C = A_resp * (w_rec[:, None] * K * w_don[None, :] * dA[None, :]) / np.maximum(dA[:, None], 1e-30)
    
    return C


def plot_baryon_magnification(R: np.ndarray,
                              B_kappa: np.ndarray,
                              cluster_name: str,
                              output_dir: Path,
                              R_edge: float = None):
    """
    Plot baryon magnification profile B_κ(R).
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    B_kappa : np.ndarray
        Magnification factor
    cluster_name : str
        Cluster identifier
    output_dir : Path
        Output directory for plot
    R_edge : float, optional
        Edge radius to mark on plot
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.semilogx(R, B_kappa, lw=2, label='B_κ(R)')
    ax.axhline(1.0, color='k', ls=':', alpha=0.5, label='GR baseline')
    
    if R_edge is not None:
        ax.axvline(R_edge, color='r', ls='--', alpha=0.5, label=f'R_edge={R_edge:.0f} kpc')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Baryon Magnification  B_κ(R)', fontsize=12)
    ax.set_title(f'{cluster_name}: How Response Magnifies Baryons', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(bottom=0.9)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{cluster_name}_baryon_magnification.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"  Saved: {out_path}")


def plot_piggyback_fraction(R: np.ndarray,
                            f_resp: np.ndarray,
                            cluster_name: str,
                            output_dir: Path,
                            R_edge: float = None):
    """
    Plot piggyback fraction profile f_resp(R).
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    f_resp : np.ndarray
        Piggyback fraction (0-1)
    cluster_name : str
        Cluster identifier
    output_dir : Path
        Output directory
    R_edge : float, optional
        Edge radius to mark
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.semilogx(R, f_resp, lw=2, color='C1', label='f_resp(R)')
    ax.axhline(0.5, color='k', ls=':', alpha=0.5, label='50% response')
    
    if R_edge is not None:
        ax.axvline(R_edge, color='r', ls='--', alpha=0.5, label=f'R_edge={R_edge:.0f} kpc')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Piggyback Fraction  f_resp(R)', fontsize=12)
    ax.set_title(f'{cluster_name}: Response vs Baryon Mass', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([-0.05, 1.05])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{cluster_name}_piggyback_fraction.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"  Saved: {out_path}")


def plot_donor_recipient_heatmap(R: np.ndarray,
                                 C_matrix: np.ndarray,
                                 cluster_name: str,
                                 output_dir: Path,
                                 R_edge: float = None):
    """
    Plot donor→recipient contribution matrix as heatmap.
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    C_matrix : np.ndarray
        Contribution matrix C_ij
    cluster_name : str
        Cluster identifier
    output_dir : Path
        Output directory
    R_edge : float, optional
        Edge radius to mark
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Log scale for visualization
    C_log = np.log10(np.maximum(C_matrix, 1e-12))
    
    im = ax.imshow(C_log, origin='lower', aspect='auto', cmap='viridis',
                   extent=[R.min(), R.max(), R.min(), R.max()])
    
    cbar = fig.colorbar(im, ax=ax, label='log₁₀(C_ij) [contribution]')
    
    if R_edge is not None:
        ax.axhline(R_edge, color='r', ls='--', alpha=0.7, lw=1)
        ax.axvline(R_edge, color='r', ls='--', alpha=0.7, lw=1, label=f'R_edge={R_edge:.0f} kpc')
    
    ax.set_xlabel('Donor Radius R\' [kpc]', fontsize=12)
    ax.set_ylabel('Recipient Radius R [kpc]', fontsize=12)
    ax.set_title(f'{cluster_name}: Donor→Recipient Matrix', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{cluster_name}_donor_recipient_matrix.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"  Saved: {out_path}")


def summarize_piggybacking(R: np.ndarray,
                          Sigma_bar: np.ndarray,
                          Sigma_resp: np.ndarray,
                          Sigma_eff: np.ndarray,
                          w_don: np.ndarray,
                          w_rec: np.ndarray,
                          R_checkpoints: list = [50, 100, 150, 200]) -> Dict:
    """
    Compute summary statistics for piggybacking diagnostics.
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    Sigma_bar : np.ndarray
        Baryon surface density in Msun/kpc²
    Sigma_resp : np.ndarray
        Response surface density in Msun/kpc²
    Sigma_eff : np.ndarray
        Effective surface density in Msun/kpc²
    w_don : np.ndarray
        Donor weights
    w_rec : np.ndarray
        Recipient weights
    R_checkpoints : list
        Radii to sample for summary
        
    Returns
    -------
    dict
        Summary metrics
    """
    B_kappa = baryon_magnification(R, Sigma_eff, Sigma_bar)
    f_resp = piggyback_fraction(Sigma_resp, Sigma_eff)
    
    # Total masses
    M_bar_total = 2.0 * np.pi * np.trapezoid(Sigma_bar * R, R)
    M_resp_total = 2.0 * np.pi * np.trapezoid(Sigma_resp * R, R)
    M_eff_total = 2.0 * np.pi * np.trapezoid(Sigma_eff * R, R)
    
    # Checkpoint values
    B_kappa_at = {f"B_kappa_{r}kpc": float(np.interp(r, R, B_kappa)) for r in R_checkpoints}
    f_resp_at = {f"f_resp_{r}kpc": float(np.interp(r, R, f_resp)) for r in R_checkpoints}
    
    # Donor/recipient core fractions
    donor_mask = w_don > 0.5
    recipient_mask = w_rec > 0.5
    
    M_donors = 2.0 * np.pi * np.trapezoid(Sigma_bar[donor_mask] * R[donor_mask], R[donor_mask]) if np.any(donor_mask) else 0.0
    M_recipients = 2.0 * np.pi * np.trapezoid(Sigma_resp[recipient_mask] * R[recipient_mask], R[recipient_mask]) if np.any(recipient_mask) else 0.0
    
    summary = {
        'M_bar_total_Msun': float(M_bar_total),
        'M_resp_total_Msun': float(M_resp_total),
        'M_eff_total_Msun': float(M_eff_total),
        'M_resp_over_M_bar': float(M_resp_total / M_bar_total) if M_bar_total > 0 else 0.0,
        'M_donors_Msun': float(M_donors),
        'M_recipients_piggybacked_Msun': float(M_recipients),
        'fraction_from_donors_to_recipients': float(M_recipients / M_donors) if M_donors > 0 else 0.0,
        **B_kappa_at,
        **f_resp_at
    }
    
    return summary


# ==================== TESTING / DEMO ====================

if __name__ == "__main__":
    print("Testing piggybacking evaluation module...\n")
    
    # Mock data (same as cooperative_response test)
    R = np.logspace(np.log10(0.1), np.log10(1000.0), 500)
    R_c = 50.0
    Sigma_bar = 1e9 / (1.0 + (R / R_c)**2)**1.5
    
    # Add mock response
    Sigma_resp = 0.5 * Sigma_bar * (R / 100.0)**0.5 * np.exp(-(R / 200.0))
    Sigma_eff = Sigma_bar + Sigma_resp
    
    # Mock gates
    from scripts.cooperative_response import donor_recipient_gates
    w_don, w_rec = donor_recipient_gates(R, Sigma_bar)
    
    # Compute diagnostics
    B_kappa = baryon_magnification(R, Sigma_eff, Sigma_bar)
    f_resp = piggyback_fraction(Sigma_resp, Sigma_eff)
    
    print(f"Baryon magnification:")
    print(f"  B_κ(50 kpc) = {np.interp(50, R, B_kappa):.3f}")
    print(f"  B_κ(100 kpc) = {np.interp(100, R, B_kappa):.3f}")
    print(f"  B_κ(200 kpc) = {np.interp(200, R, B_kappa):.3f}")
    
    print(f"\nPiggyback fraction:")
    print(f"  f_resp(50 kpc) = {np.interp(50, R, f_resp):.3f}")
    print(f"  f_resp(100 kpc) = {np.interp(100, R, f_resp):.3f}")
    print(f"  f_resp(200 kpc) = {np.interp(200, R, f_resp):.3f}")
    
    # Summary
    summary = summarize_piggybacking(R, Sigma_bar, Sigma_resp, Sigma_eff, w_don, w_rec)
    print(f"\nSummary metrics:")
    print(f"  M_resp / M_bar = {summary['M_resp_over_M_bar']:.3f}")
    print(f"  Donors → Recipients = {summary['fraction_from_donors_to_recipients']:.3f}")
    
    print("\n✓ Piggybacking evaluation module test complete")

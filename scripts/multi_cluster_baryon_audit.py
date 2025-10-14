#!/usr/bin/env python3
"""
Multi-Cluster Baryon Audit
===========================

Diagnostic tool to check baryon data quality across multiple clusters.
Tests:
1. Gas mass normalization vs published values
2. Surface density slope at Einstein radius
3. Current shortfall factor for lensing
4. Identifies anomalous vs representative clusters

Usage:
    python scripts/multi_cluster_baryon_audit.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List
from many_path_model.cluster_data_loader import ClusterDataLoader
from many_path_model.lensing_utilities import LensingCosmology

# Known Einstein radii and masses from literature
CLUSTER_PARAMETERS = {
    'MACSJ0416': {
        'theta_E_obs': 35.0,  # arcsec, Frontier Fields
        'z_lens': 0.396,
        'M_500_pub': 1.15e15,  # Msun, Umetsu+ 2016
        'R_500': 1200.0,  # kpc, approximate
        'f_gas_pub': 0.11,  # Published gas fraction
    },
    'MACSJ0717': {
        'theta_E_obs': 55.0,  # arcsec
        'z_lens': 0.548,
        'M_500_pub': 2.0e15,
        'R_500': 1400.0,
        'f_gas_pub': 0.12,
    },
    'ABELL_1689': {
        'theta_E_obs': 45.0,  # arcsec
        'z_lens': 0.183,
        'M_500_pub': 1.3e15,
        'R_500': 1300.0,
        'f_gas_pub': 0.11,
    },
    'ABELL_0370': {
        'theta_E_obs': 40.0,  # arcsec, Frontier Fields
        'z_lens': 0.375,
        'M_500_pub': 1.0e15,
        'R_500': 1150.0,
        'f_gas_pub': 0.10,
    },
    'ABELL_2744': {
        'theta_E_obs': 30.0,  # arcsec, Frontier Fields
        'z_lens': 0.308,
        'M_500_pub': 1.2e15,
        'R_500': 1250.0,
        'f_gas_pub': 0.11,
    },
}

@dataclass
class ClusterAudit:
    """Results from cluster baryon audit."""
    cluster_name: str
    z_lens: float
    
    # Observed values
    theta_E_obs: float
    R_E_kpc: float
    M_500_pub: float
    f_gas_pub: float
    
    # Loaded data
    M_gas_loaded: float
    M_star_loaded: float
    M_baryon_loaded: float
    f_gas_loaded: float
    
    # Surface density at Einstein radius
    R_test: float
    Sigma_gas: float
    Sigma_star: float
    Sigma_total: float
    Sigma_crit: float
    
    # Slope diagnostic
    dlnSigma_dlnR: float
    
    # Shortfall analysis
    kappa_baryon: float
    kappa_needed: float
    shortfall_factor: float
    boost_needed: float


def compute_slope(R_array, Sigma_array, R_target):
    """Compute d ln Σ / d ln R at target radius."""
    # Find indices around target
    idx = np.argmin(np.abs(R_array - R_target))
    
    # Use centered difference over ±20% range
    R_low = R_target * 0.8
    R_high = R_target * 1.2
    
    mask = (R_array >= R_low) & (R_array <= R_high)
    if np.sum(mask) < 3:
        return np.nan
    
    R_fit = R_array[mask]
    Sigma_fit = Sigma_array[mask]
    
    # Fit in log-log space
    log_R = np.log(R_fit)
    log_Sigma = np.log(np.maximum(Sigma_fit, 1e-10))
    
    coeffs = np.polyfit(log_R, log_Sigma, 1)
    slope = coeffs[0]
    
    return slope


def audit_cluster(cluster_name: str, loader: ClusterDataLoader, 
                  cosmo: LensingCosmology, z_src: float = 2.0) -> ClusterAudit:
    """Run full audit on a single cluster."""
    
    params = CLUSTER_PARAMETERS[cluster_name]
    
    # Load cluster data
    data = loader.load_cluster(cluster_name, validate=False)
    
    # Compute masses
    r = data.r_kpc
    M_gas = np.trapezoid(4*np.pi*r**2*data.rho_gas, r)
    M_star = np.trapezoid(4*np.pi*r**2*data.rho_stars, r)
    M_baryon = M_gas + M_star
    
    # Gas fraction
    R_500 = params['R_500']
    idx_500 = np.argmin(np.abs(r - R_500))
    M_gas_500 = np.trapezoid(4*np.pi*r[:idx_500+1]**2*data.rho_gas[:idx_500+1], r[:idx_500+1])
    M_500 = params['M_500_pub']
    f_gas_loaded = M_gas_500 / M_500 if M_500 > 0 else 0
    
    # Einstein radius in kpc
    theta_E = params['theta_E_obs']
    R_E = theta_E / cosmo.physical_to_angular(1.0, data.z_lens)  # convert arcsec to kpc
    
    # Project to get surface densities
    from many_path_model.lensing_utilities import AbelProjection
    projector = AbelProjection()
    
    R_array = np.geomspace(10, 1500, 100)
    Sigma_gas = projector.project_density_to_surface(r, data.rho_gas, R_array)
    Sigma_star = projector.project_density_to_surface(r, data.rho_stars, R_array)
    Sigma_total = projector.project_density_to_surface(r, data.rho_total, R_array)
    
    # Values at Einstein radius
    idx_E = np.argmin(np.abs(R_array - R_E))
    R_test = R_array[idx_E]
    
    # Slope at Einstein radius
    slope = compute_slope(R_array, Sigma_total, R_E)
    
    # Critical surface density
    Sigma_crit = cosmo.critical_surface_density(data.z_lens, z_src)
    
    # Shortfall analysis
    kappa_baryon = Sigma_total[idx_E] / Sigma_crit
    kappa_needed = 1.0
    shortfall_factor = kappa_needed / kappa_baryon if kappa_baryon > 0 else np.inf
    boost_needed = shortfall_factor - 1.0  # K_Σ needed beyond baryons
    
    return ClusterAudit(
        cluster_name=cluster_name,
        z_lens=data.z_lens,
        theta_E_obs=theta_E,
        R_E_kpc=R_E,
        M_500_pub=M_500,
        f_gas_pub=params['f_gas_pub'],
        M_gas_loaded=M_gas,
        M_star_loaded=M_star,
        M_baryon_loaded=M_baryon,
        f_gas_loaded=f_gas_loaded,
        R_test=R_test,
        Sigma_gas=Sigma_gas[idx_E],
        Sigma_star=Sigma_star[idx_E],
        Sigma_total=Sigma_total[idx_E],
        Sigma_crit=Sigma_crit,
        dlnSigma_dlnR=slope,
        kappa_baryon=kappa_baryon,
        kappa_needed=kappa_needed,
        shortfall_factor=shortfall_factor,
        boost_needed=boost_needed,
    )


def print_audit_report(audits: List[ClusterAudit]):
    """Print comprehensive audit report."""
    
    print("\n" + "="*80)
    print("MULTI-CLUSTER BARYON AUDIT REPORT")
    print("="*80)
    
    for audit in audits:
        print(f"\n{'─'*80}")
        print(f"CLUSTER: {audit.cluster_name}")
        print(f"{'─'*80}")
        
        print(f"\n1. OBSERVABLES:")
        print(f"   z_lens = {audit.z_lens:.3f}")
        print(f"   θ_E (observed) = {audit.theta_E_obs:.1f} arcsec → R_E = {audit.R_E_kpc:.1f} kpc")
        print(f"   M_500 (published) = {audit.M_500_pub:.2e} M☉")
        
        print(f"\n2. LOADED BARYON MASSES:")
        print(f"   M_gas = {audit.M_gas_loaded:.2e} M☉")
        print(f"   M_star = {audit.M_star_loaded:.2e} M☉")
        print(f"   M_baryon (total) = {audit.M_baryon_loaded:.2e} M☉")
        
        print(f"\n3. GAS FRACTION CHECK:")
        print(f"   f_gas (published) = {audit.f_gas_pub:.3f}")
        print(f"   f_gas (loaded) = {audit.f_gas_loaded:.3f}")
        gas_ratio = audit.f_gas_loaded / audit.f_gas_pub if audit.f_gas_pub > 0 else 0
        status = "✅" if 0.7 < gas_ratio < 1.3 else "⚠️"
        print(f"   Ratio (loaded/pub) = {gas_ratio:.2f} {status}")
        
        print(f"\n4. SURFACE DENSITY AT R_E = {audit.R_test:.1f} kpc:")
        print(f"   Σ_gas = {audit.Sigma_gas:.2e} M☉/kpc²")
        print(f"   Σ_star = {audit.Sigma_star:.2e} M☉/kpc²")
        print(f"   Σ_total = {audit.Sigma_total:.2e} M☉/kpc²")
        print(f"   Σ_crit = {audit.Sigma_crit:.2e} M☉/kpc²")
        
        print(f"\n5. PROFILE SLOPE:")
        print(f"   d ln Σ / d ln R = {audit.dlnSigma_dlnR:.2f}")
        slope_status = "✅ OK" if -2.0 < audit.dlnSigma_dlnR < -1.0 else "⚠️ TOO STEEP"
        print(f"   Expected: -1.0 to -1.5  →  {slope_status}")
        
        print(f"\n6. LENSING SHORTFALL:")
        print(f"   κ (baryons only) = {audit.kappa_baryon:.3f}")
        print(f"   κ (needed) = {audit.kappa_needed:.3f}")
        print(f"   Shortfall factor = {audit.shortfall_factor:.2f}×")
        print(f"   K_Σ boost needed = {audit.boost_needed:.1f}")
        
        if audit.shortfall_factor > 5:
            print(f"   STATUS: ❌ SEVERE SHORTFALL (>{5}×)")
        elif audit.shortfall_factor > 2:
            print(f"   STATUS: ⚠️ MODERATE SHORTFALL (2-5×)")
        else:
            print(f"   STATUS: ✅ WITHIN RANGE (<2×)")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    gas_ratios = [a.f_gas_loaded / a.f_gas_pub for a in audits if a.f_gas_pub > 0]
    shortfalls = [a.shortfall_factor for a in audits if np.isfinite(a.shortfall_factor)]
    slopes = [a.dlnSigma_dlnR for a in audits if np.isfinite(a.dlnSigma_dlnR)]
    
    print(f"\nGas Fraction (loaded/published):")
    print(f"   Mean: {np.mean(gas_ratios):.2f} ± {np.std(gas_ratios):.2f}")
    print(f"   Range: {np.min(gas_ratios):.2f} - {np.max(gas_ratios):.2f}")
    
    print(f"\nShortfall Factor:")
    print(f"   Mean: {np.mean(shortfalls):.2f}×")
    print(f"   Median: {np.median(shortfalls):.2f}×")
    print(f"   Range: {np.min(shortfalls):.2f}× - {np.max(shortfalls):.2f}×")
    
    print(f"\nProfile Slope at R_E:")
    print(f"   Mean: {np.mean(slopes):.2f}")
    print(f"   Expected: -1.0 to -1.5")
    
    # Systematic issues?
    if np.mean(gas_ratios) < 0.7:
        print(f"\n⚠️  SYSTEMATIC ISSUE: Gas masses consistently LOW")
    if np.mean(shortfalls) > 5:
        print(f"\n⚠️  SYSTEMATIC ISSUE: Universal severe shortfall")
    if np.mean(slopes) < -2.0:
        print(f"\n⚠️  SYSTEMATIC ISSUE: Profiles too steep")
    
    print(f"\n{'='*80}\n")


def main():
    """Run multi-cluster audit."""
    
    loader = ClusterDataLoader()
    cosmo = LensingCosmology()
    
    audits = []
    failed = []
    
    for cluster_name in CLUSTER_PARAMETERS.keys():
        try:
            print(f"Auditing {cluster_name}...", end=" ")
            audit = audit_cluster(cluster_name, loader, cosmo)
            audits.append(audit)
            print("✅")
        except Exception as e:
            print(f"❌ {e}")
            failed.append((cluster_name, str(e)))
    
    if audits:
        print_audit_report(audits)
        
        # Save to JSON
        output = {
            'n_clusters': len(audits),
            'clusters': [
                {
                    'name': a.cluster_name,
                    'f_gas_ratio': a.f_gas_loaded / a.f_gas_pub,
                    'shortfall_factor': float(a.shortfall_factor),
                    'slope': float(a.dlnSigma_dlnR),
                    'boost_needed': float(a.boost_needed),
                }
                for a in audits
            ],
            'failed': failed
        }
        
        output_path = Path('results/multi_cluster_baryon_audit.json')
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {output_path}")
    else:
        print("\n❌ No clusters successfully audited")
        for name, error in failed:
            print(f"  {name}: {error}")


if __name__ == '__main__':
    main()

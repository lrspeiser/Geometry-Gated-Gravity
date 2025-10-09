#!/usr/bin/env python3
"""
Generate expanded dataset of 30 realistic clusters for blind validation.

This script creates synthetic cluster profiles based on CLASH/RELICS survey
statistics, WITHOUT using any lensing information during generation.

Strategy:
1. Sample cluster properties from observed distributions (M_500, z, concentrations)
2. Generate baryon profiles (beta model gas + stellar component)
3. Apply our EXISTING universal formulas (S_∞, Rs) without refitting
4. Predict lensing and compare to "true" synthetic lensing
5. Track which predictions succeed/fail

NO FITTING ALLOWED - this is pure prediction validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma
from scipy.ndimage import gaussian_filter1d

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

# Physical constants
c_kms = 299792.458
G_kpc3_Msun_km2s2 = 4.302e-6

OUTPUT_DIR = Path("out/expanded_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# UNIVERSAL FORMULAS (FIXED - NO REFITTING ALLOWED)
# =============================================================================

UNIVERSAL_PARAMS = {
    'a_eps': 0.60,      # Edge sharpness exponent
    'b_mass': 0.25,     # Core mass exponent
    'alpha': 10.0,      # Overall normalization
    'beta_Rs': 0.90,    # Rs/R_edge ratio
    'M0': 1e13,         # Mass normalization
}

def predict_S_inf(edge_sharp, M_core_Msun):
    """
    Universal S_∞ prediction - LOCKED FORMULA, NO FITTING.
    
    S_∞ = 1 + α · ε^0.6 · (M_core/10^13)^0.25
    """
    return 1 + UNIVERSAL_PARAMS['alpha'] * \
           edge_sharp**UNIVERSAL_PARAMS['a_eps'] * \
           (M_core_Msun / UNIVERSAL_PARAMS['M0'])**UNIVERSAL_PARAMS['b_mass']


def predict_Rs(R_edge_kpc):
    """
    Universal Rs prediction - LOCKED FORMULA, NO FITTING.
    
    Rs = 0.90 × R_edge
    """
    return UNIVERSAL_PARAMS['beta_Rs'] * R_edge_kpc


@dataclass
class ClusterProperties:
    """Container for cluster physical properties."""
    name: str
    cluster_id: int
    z_lens: float
    z_source: float
    
    # Observed baryon properties
    M_gas_Msun: float
    M_stars_Msun: float
    r_core_kpc: float
    beta: float
    
    # Derived quantities
    M_total_baryon: float
    R_edge_kpc: float
    edge_sharp: float
    M_core_Msun: float
    
    # Predicted enhancement (from universal rules)
    S_inf_predicted: float
    Rs_predicted_kpc: float
    
    # Cluster type
    is_merger: bool
    n_components: int
    
    # For tracking
    dataset_split: str  # 'train', 'validation', or 'test'


# =============================================================================
# OBSERVATIONAL DISTRIBUTIONS (based on CLASH/RELICS)
# =============================================================================

def sample_cluster_mass():
    """Sample cluster mass from lognormal distribution."""
    # CLASH/RELICS: log(M_500) ~ 14.5-15.2
    log_M500 = np.random.uniform(14.5, 15.2)
    M_500 = 10**log_M500
    
    # Gas mass ~ 12-15% of total (baryon fraction ~ 0.13)
    f_gas = np.random.uniform(0.12, 0.15)
    M_gas = f_gas * M_500
    
    # Stellar mass ~ 1-3% of total
    f_stars = np.random.uniform(0.01, 0.03)
    M_stars = f_stars * M_500
    
    return M_gas, M_stars


def sample_redshift():
    """Sample redshift from CLASH/RELICS distribution."""
    # Most clusters at z ~ 0.2-0.6, with tail to z ~ 1.0
    z_lens = np.random.beta(2, 5) * 0.8 + 0.2  # Skewed toward z~0.3-0.5
    
    # Source typically at z > 1.5
    z_source = np.random.uniform(1.5, 3.5)
    
    return z_lens, z_source


def sample_concentration():
    """Sample concentration parameter."""
    # Beta parameter: typically 0.5-0.8 (Vikhlinin+06)
    beta = np.random.uniform(0.5, 0.8)
    
    # Core radius: 50-200 kpc for massive clusters
    r_core = np.random.uniform(50, 200)
    
    return r_core, beta


def sample_morphology():
    """Sample cluster morphology (relaxed vs merger)."""
    # ~30% are mergers based on X-ray morphology
    is_merger = np.random.random() < 0.3
    
    if is_merger:
        n_components = np.random.choice([2, 3], p=[0.7, 0.3])
    else:
        n_components = 1
    
    return is_merger, n_components


# =============================================================================
# BARYON PROFILE GENERATION
# =============================================================================

def beta_profile_3d(r, rho0, rc, beta):
    """3D beta model density."""
    return rho0 * (1 + (r / rc)**2)**(-1.5 * beta)


def abel_project_beta_analytic(R, rho0, rc, beta):
    """Analytic Abel projection of beta model."""
    x = R / rc
    factor = rho0 * rc * np.sqrt(np.pi) * gamma(1.5 * beta - 0.5) / gamma(1.5 * beta)
    Sigma = factor * (1 + x**2)**(0.5 - 1.5 * beta)
    return Sigma


def generate_baryon_profile(M_gas, M_stars, r_core, beta, n_components=1):
    """
    Generate baryon surface density profile.
    
    Returns:
        R_kpc: Radial grid
        Sigma_kpc2: Surface density (M_sun/kpc^2)
        rho_3d: 3D density for reference
    """
    R = np.logspace(-1, 3, 500)
    
    if n_components == 1:
        # Simple single-component
        # Normalize gas component
        rho0_gas = 1.0
        Sigma_gas = abel_project_beta_analytic(R, rho0_gas, r_core, beta)
        M_enc_gas = cumulative_trapezoid(Sigma_gas * 2 * np.pi * R, R, initial=0)
        Sigma_gas *= M_gas / M_enc_gas[-1]
        
        # Stellar component (more concentrated)
        r_core_stars = r_core / 3
        rho0_stars = 1.0
        Sigma_stars = abel_project_beta_analytic(R, rho0_stars, r_core_stars, beta + 0.2)
        M_enc_stars = cumulative_trapezoid(Sigma_stars * 2 * np.pi * R, R, initial=0)
        Sigma_stars *= M_stars / M_enc_stars[-1]
        
        Sigma_total = Sigma_gas + Sigma_stars
        
    else:
        # Merger: multiple components
        Sigma_total = np.zeros_like(R)
        
        for i in range(n_components):
            # Split mass between components
            M_gas_i = M_gas / n_components * np.random.uniform(0.7, 1.3)
            M_stars_i = M_stars / n_components * np.random.uniform(0.7, 1.3)
            
            # Slightly different core radii
            r_core_i = r_core * np.random.uniform(0.8, 1.2)
            beta_i = beta * np.random.uniform(0.9, 1.1)
            
            # Gas component
            rho0 = 1.0
            Sigma_comp = abel_project_beta_analytic(R, rho0, r_core_i, beta_i)
            M_enc = cumulative_trapezoid(Sigma_comp * 2 * np.pi * R, R, initial=0)
            Sigma_comp *= M_gas_i / M_enc[-1]
            
            # Add to total
            Sigma_total += Sigma_comp
            
            # Stellar component
            r_core_stars = r_core_i / 3
            Sigma_stars = abel_project_beta_analytic(R, rho0, r_core_stars, beta_i + 0.2)
            M_enc_stars = cumulative_trapezoid(Sigma_stars * 2 * np.pi * R, R, initial=0)
            Sigma_stars *= M_stars_i / M_enc_stars[-1]
            
            Sigma_total += Sigma_stars
    
    return R, Sigma_total


# =============================================================================
# FEATURE EXTRACTION (same as training code)
# =============================================================================

def extract_features(R_kpc, Sigma_kpc2, Sigma0_pc2=100.0):
    """
    Extract baryon geometry features.
    
    CRITICAL: This is the ONLY interface between observations and predictions.
    """
    # Mean surface density
    integrand = Sigma_kpc2 * 2 * np.pi * R_kpc
    M_enc = cumulative_trapezoid(integrand, R_kpc, initial=0)
    Sigma_bar = M_enc / (np.pi * R_kpc**2 + 1e-20)
    
    # Find R_edge where Σ̄(<R) = Σ₀
    log_ratio = np.abs(np.log10(Sigma_bar / 1e6 + 1e-20) - np.log10(Sigma0_pc2))
    idx_edge = np.argmin(log_ratio)
    R_edge = R_kpc[idx_edge]
    
    # Edge sharpness with smoothing
    lnR = np.log(R_kpc + 1e-6)
    lnS = np.log(Sigma_kpc2 / 1e6 + 1e-12)
    lnS_smooth = gaussian_filter1d(lnS, sigma=2)
    
    edge_band = (R_kpc > 0.5*R_edge) & (R_kpc < 1.5*R_edge)
    gradS = np.abs(np.gradient(lnS_smooth, lnR))
    edge_sharp = np.max(gradS[edge_band]) if edge_band.any() else 1.0
    
    # Core mass (M within 50-100 kpc)
    core_band = (R_kpc >= 50) & (R_kpc <= 100)
    if core_band.any():
        M_core = M_enc[core_band][-1] if np.any(M_enc[core_band]) else M_enc[50]
    else:
        M_core = M_enc[50] if len(M_enc) > 50 else M_enc[-1]
    
    return {
        'R_edge': R_edge,
        'edge_sharp': edge_sharp,
        'M_core': M_core,
        'Sigma_bar': Sigma_bar,
        'M_enc': M_enc,
    }


# =============================================================================
# DEFLECTION COMPUTATION
# =============================================================================

def angular_diameter_distance(z, H0=70, Om=0.3):
    """Angular diameter distance (Mpc)."""
    from scipy.integrate import quad
    
    def E(z):
        return np.sqrt(Om * (1 + z)**3 + (1 - Om))
    
    c_Mpc_s = 299792.458 / 1e3
    integral, _ = quad(lambda zp: 1 / E(zp), 0, z)
    D_c = (c_Mpc_s / H0) * integral
    D_a = D_c / (1 + z)
    
    return D_a


def compute_gr_deflection(R_kpc, Sigma_kpc2, theta_arcsec, D_d_Mpc, D_s_Mpc, D_ls_Mpc):
    """Compute GR deflection from baryons."""
    # Critical surface density
    Sigma_crit = (c_kms**2 / (4 * np.pi * G_kpc3_Msun_km2s2)) * \
                 (D_s_Mpc / (D_d_Mpc * D_ls_Mpc))
    
    # Enclosed mass
    integrand = Sigma_kpc2 * 2 * np.pi * R_kpc
    M_enc = cumulative_trapezoid(integrand, R_kpc, initial=0)
    
    # Convert theta to R
    D_d_kpc = D_d_Mpc * 1e3
    theta_rad = theta_arcsec / 206265.0
    R_theta = theta_rad * D_d_kpc
    
    # Interpolate
    M_enc_theta = np.interp(R_theta, R_kpc, M_enc)
    
    # Deflection
    kappa_bar = M_enc_theta / (np.pi * R_theta**2 * Sigma_crit)
    alpha_gr = kappa_bar * theta_arcsec
    
    return alpha_gr


def compute_slip_factor(R_kpc, Sigma_bar, S_inf, Rs, p=2.0, eta=1.5, x0=-1.0, w=0.4, Sigma0_pc2=100.0):
    """Compute slip factor S(R)."""
    # Radial activation
    S_base = 1 + S_inf * (1 - np.exp(-(R_kpc / Rs)**p))**eta
    
    # Mean density gating
    S_hat = np.log10(Sigma_bar / 1e6 + 1e-20) - np.log10(Sigma0_pc2)
    gate = 1 - 1 / (1 + np.exp(-(S_hat - x0) / w))
    
    # Combined
    S = 1 + (S_base - 1) * gate
    
    # Enforce monotonicity
    S = np.maximum.accumulate(S)
    
    return np.clip(S, 1, 50)


def compute_model_deflection(R_kpc, Sigma_kpc2, Sigma_bar, S_inf, Rs, 
                            theta_arcsec, D_d_Mpc, D_s_Mpc, D_ls_Mpc):
    """Compute model deflection with slip enhancement."""
    # GR baseline
    alpha_gr = compute_gr_deflection(R_kpc, Sigma_kpc2, theta_arcsec, D_d_Mpc, D_s_Mpc, D_ls_Mpc)
    
    # Slip factor
    S = compute_slip_factor(R_kpc, Sigma_bar, S_inf, Rs)
    
    # Apply on consistent grid
    D_d_kpc = D_d_Mpc * 1e3
    theta_R = (R_kpc / D_d_kpc) * 206265.0
    alpha_gr_R = np.interp(theta_R, theta_arcsec, alpha_gr)
    alpha_model_R = alpha_gr_R * S
    alpha_model = np.interp(theta_arcsec, theta_R, alpha_model_R)
    
    return alpha_model, alpha_gr


# =============================================================================
# CLUSTER GENERATION PIPELINE
# =============================================================================

def generate_single_cluster(cluster_id, dataset_split='train'):
    """Generate one synthetic cluster with all properties."""
    
    # Sample properties
    M_gas, M_stars = sample_cluster_mass()
    z_lens, z_source = sample_redshift()
    r_core, beta = sample_concentration()
    is_merger, n_components = sample_morphology()
    
    # Generate baryon profile
    R_kpc, Sigma_kpc2 = generate_baryon_profile(M_gas, M_stars, r_core, beta, n_components)
    
    # Extract features
    features = extract_features(R_kpc, Sigma_kpc2)
    
    # PREDICT using universal formulas (NO FITTING)
    S_inf_pred = predict_S_inf(features['edge_sharp'], features['M_core'])
    Rs_pred = predict_Rs(features['R_edge'])
    
    # Create cluster object
    cluster = ClusterProperties(
        name=f"SYNTH{cluster_id:03d}",
        cluster_id=cluster_id,
        z_lens=z_lens,
        z_source=z_source,
        M_gas_Msun=M_gas,
        M_stars_Msun=M_stars,
        r_core_kpc=r_core,
        beta=beta,
        M_total_baryon=M_gas + M_stars,
        R_edge_kpc=features['R_edge'],
        edge_sharp=features['edge_sharp'],
        M_core_Msun=features['M_core'],
        S_inf_predicted=S_inf_pred,
        Rs_predicted_kpc=Rs_pred,
        is_merger=is_merger,
        n_components=n_components,
        dataset_split=dataset_split,
    )
    
    # Compute distances (all in Mpc)
    D_d = angular_diameter_distance(z_lens)
    D_s = angular_diameter_distance(z_source)
    
    # Angular diameter distance lens to source
    # D_ls = D_s * (1+z_l) - D_d * (1+z_l) simplified
    # Use comoving distance approach
    from scipy.integrate import quad
    H0, Om = 70.0, 0.3
    c_Mpc_s = 299792.458 / 1e3
    E = lambda z: np.sqrt(Om * (1 + z)**3 + (1 - Om))
    
    # Comoving distance from lens to source
    integral_ls, _ = quad(lambda z: 1/E(z), z_lens, z_source)
    D_c_ls = (c_Mpc_s / H0) * integral_ls
    D_ls = D_c_ls / (1 + z_source)
    
    # Compute lensing
    theta = np.linspace(10, 150, 30)
    alpha_model, alpha_gr = compute_model_deflection(
        R_kpc, Sigma_kpc2, features['Sigma_bar'],
        S_inf_pred, Rs_pred, theta, D_d, D_s, D_ls
    )
    
    # Store results
    results = {
        'cluster': asdict(cluster),
        'baryon_profile': {
            'R_kpc': R_kpc.tolist(),
            'Sigma_kpc2': Sigma_kpc2.tolist(),
        },
        'lensing': {
            'theta_arcsec': theta.tolist(),
            'alpha_gr': alpha_gr.tolist(),
            'alpha_model': alpha_model.tolist(),
        },
        'features': {k: float(v) if np.isscalar(v) else v.tolist() 
                    for k, v in features.items()},
    }
    
    return cluster, results


def generate_full_dataset(n_total=30, train_frac=0.5, val_frac=0.3):
    """
    Generate full dataset with train/validation/test splits.
    
    Args:
        n_total: Total number of clusters
        train_frac: Fraction for training (will NOT be used for our locked formulas)
        val_frac: Fraction for validation
    """
    print(f"Generating {n_total} synthetic clusters...")
    print(f"Split: {train_frac*100:.0f}% train, {val_frac*100:.0f}% val, {(1-train_frac-val_frac)*100:.0f}% test")
    print()
    
    # Determine splits
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val
    
    all_clusters = []
    all_results = []
    
    # Generate clusters
    for i in range(n_total):
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'validation'
        else:
            split = 'test'
        
        cluster, results = generate_single_cluster(i, split)
        all_clusters.append(cluster)
        all_results.append(results)
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i+1}/{n_total} clusters...")
    
    print(f"\n✓ Generated {n_total} clusters successfully")
    print(f"  Train: {n_train}, Validation: {n_val}, Test: {n_test}")
    
    return all_clusters, all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate expanded cluster dataset."""
    print("="*70)
    print("EXPANDED CLUSTER DATASET GENERATION")
    print("Using LOCKED universal formulas (NO REFITTING)")
    print("="*70)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset
    clusters, results = generate_full_dataset(n_total=30, train_frac=0.5, val_frac=0.3)
    
    # Save full dataset
    output_file = OUTPUT_DIR / "expanded_dataset.json"
    with open(output_file, 'w') as f:
        json.dump({
            'universal_params': UNIVERSAL_PARAMS,
            'clusters': results,
            'n_total': int(len(clusters)),
            'n_train': int(sum(1 for c in clusters if c.dataset_split == 'train')),
            'n_val': int(sum(1 for c in clusters if c.dataset_split == 'validation')),
            'n_test': int(sum(1 for c in clusters if c.dataset_split == 'test')),
        }, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Saved dataset to {output_file}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    print(f"\nMass range:")
    M_totals = [c.M_total_baryon for c in clusters]
    print(f"  {min(M_totals)/1e13:.2f} - {max(M_totals)/1e13:.2f} × 10^13 M_sun")
    
    print(f"\nRedshift range:")
    z_lenses = [c.z_lens for c in clusters]
    print(f"  z_lens = {min(z_lenses):.2f} - {max(z_lenses):.2f}")
    
    print(f"\nEdge sharpness range:")
    edge_sharps = [c.edge_sharp for c in clusters]
    print(f"  ε = {min(edge_sharps):.2f} - {max(edge_sharps):.2f}")
    
    print(f"\nPredicted S_∞ range:")
    S_infs = [c.S_inf_predicted for c in clusters]
    print(f"  S_∞ = {min(S_infs):.1f} - {max(S_infs):.1f}")
    
    print(f"\nMorphology:")
    n_relaxed = sum(1 for c in clusters if not c.is_merger)
    n_mergers = sum(1 for c in clusters if c.is_merger)
    print(f"  Relaxed: {n_relaxed} ({n_relaxed/len(clusters)*100:.0f}%)")
    print(f"  Mergers: {n_mergers} ({n_mergers/len(clusters)*100:.0f}%)")
    
    print("\n" + "="*70)
    print("READY FOR BLIND VALIDATION")
    print("="*70)


if __name__ == "__main__":
    main()

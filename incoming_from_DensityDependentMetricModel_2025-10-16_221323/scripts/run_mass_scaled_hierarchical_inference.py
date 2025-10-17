#!/usr/bin/env python3
"""
run_mass_scaled_hierarchical_inference.py

Hierarchical Bayesian inference for cluster strong lensing with mass-scaled 
coherence length: ℓ₀(M) = ℓ₀,⋆ (R₅₀₀/1 Mpc)^γ

Tests the central scientific question:
    Does the coherence length of Sigma-Gravity scale with halo mass?

Model hierarchy:
    Population level:
        μ_A ~ N(16.5, 1.5²)           # Amplitude mean
        σ_A ~ HalfNormal(1.0)          # Amplitude scatter  
        ℓ₀,⋆ ~ LogNormal(ln 200 kpc, 0.5²)  # Pivot coherence at 1 Mpc
        γ ~ Uniform(0, 1)              # Mass-scaling exponent
        
    Per-cluster:
        A_c,i ~ N(μ_A, σ_A²)
        q_LOS,i ~ N(1, 0.15²) truncated [0.7, 1.4]
        q_plane,i ~ N(1, 0.15²) truncated [0.7, 1.4]
        κ_ext,i ~ N(0, 0.03²)
        
    Observation model:
        θ_E,obs ~ N(θ_E,model, σ_obs²)

Pre-registered decision rules:
    - ΔWAIC ≥ 4 favoring mass-scaled → evidence for scaling
    - γ posterior piled near 0 with tight CI → scale-invariant
    
Failure modes guarded:
    - Training bias: hold out specified clusters
    - Geometry leakage: hierarchical q with shrinkage  
    - Clumping: gas density divided by √C(r)
    - Selection: document tier cuts

Usage:
    # Mass-scaled model
    python scripts/run_mass_scaled_hierarchical_inference.py \\
        --tiers 1,2 --exclude MACS0717 --use-triaxial 1 \\
        --draws 4000 --tune 2000 --chains 4 --target_accept 0.9 \\
        --out output/mass_scaled/
    
    # Fixed-scale comparison (γ=0)
    python scripts/run_mass_scaled_hierarchical_inference.py \\
        --tiers 1,2 --exclude MACS0717 --use-triaxial 1 \\
        --fix-gamma 0 --draws 4000 --tune 2000 --chains 4 \\
        --out output/fixed_scale/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    print("Warning: PyMC not available. Install with: pip install pymc arviz", file=sys.stderr)

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Local utilities
from scripts.cluster_overrides import load_cluster_override
from scripts import lensing_utils
from scripts.baryon_loader import load_baryon_profile, interpolate_baryon


def load_cluster_catalog(
    tiers: List[int],
    exclude: Optional[List[str]] = None,
    data_path: Optional[Path] = None,
    include: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cluster catalog with tier filtering and exclusions.
    
    Catalog schema:
        cluster_name, z_lens, z_source, theta_E_obs, sigma_theta_E,
        M500_1e14Msun, R500_Mpc, tier, has_weak_lensing
    
    Tier definitions:
        1: Gold (high-quality strong lensing, minimal contamination)
        2: Silver (good lensing, some systematics)
        3: Complex mergers (MACS0717, etc.)
    """
    if data_path is None:
        data_path = REPO_ROOT / "data" / "cluster_lensing_catalog.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cluster catalog not found: {data_path}\n"
            "Create it with scripts/build_cluster_catalog.py"
        )
    
    df = pd.read_csv(data_path)
    
    # Filter by tier
    df = df[df['tier'].isin(tiers)]
    
    # Exclude specified clusters
    if exclude:
        exclude_upper = [c.upper().replace(' ', '').replace('-', '') for c in exclude]
        df = df[~df['cluster_name'].str.upper().str.replace(' ', '').str.replace('-', '').isin(exclude_upper)]
    
    # Include filter (if provided)
    if include:
        include_upper = [c.upper().replace(' ', '').replace('-', '') for c in include]
        df = df[df['cluster_name'].str.upper().str.replace(' ', '').str.replace('-', '').isin(include_upper)]
    
    # Require finite Einstein radius
    df = df[np.isfinite(df['theta_E_obs']) & (df['theta_E_obs'] > 0)]
    
    return df.reset_index(drop=True)


def sigma_gravity_kernel(
    R_kpc: np.ndarray,
    A_c: float,
    ell_0_kpc: float,
    Sigma_bar_Msun_kpc2: np.ndarray
) -> np.ndarray:
    """
    Sigma-Gravity surface density kernel with exponential coherence.
    
    Σ_eff(R) = Σ_bar(R) + A_c × Σ_bar(R) × exp(-R / ℓ₀)
    
    Returns:
        Σ_eff in M_sun/kpc²
    """
    enhancement = A_c * np.exp(-R_kpc / ell_0_kpc)
    return Sigma_bar_Msun_kpc2 * (1.0 + enhancement)


def compute_theta_E_triaxial(
    Sigma_bar_profile: np.ndarray,
    R_kpc: np.ndarray,
    A_c: float,
    ell_0_kpc: float,
    q_LOS: float,
    q_plane: float,
    kappa_ext: float,
    D_lens_Mpc: float,
    D_source_Mpc: float,
    D_LS_Mpc: float
) -> float:
    """
    Compute Einstein radius with triaxial geometry projection.
    
    Projection:
        Σ_projected = q_LOS × q_plane × Σ_spherical
    
    Args:
        Sigma_bar_profile: Baryon surface density [M_sun/kpc²]
        R_kpc: Radial grid [kpc]
        A_c: Coherence amplitude
        ell_0_kpc: Coherence length [kpc]
        q_LOS: Line-of-sight axis ratio
        q_plane: In-plane axis ratio
        kappa_ext: External convergence sheet
        D_lens_Mpc: Angular diameter distance to lens
        D_source_Mpc: Angular diameter distance to source
        D_LS_Mpc: Lens-source angular diameter distance
        
    Returns:
        theta_E in arcsec
    """
    # Apply Sigma-Gravity kernel
    Sigma_eff = sigma_gravity_kernel(R_kpc, A_c, ell_0_kpc, Sigma_bar_profile)
    
    # Apply triaxial geometry projection
    Sigma_projected = q_LOS * q_plane * Sigma_eff
    
    # Critical surface density (convert Mpc→kpc for distance inputs)
    # Σ_cr = c²/(4πG) × D_s/(D_l × D_ls)
    c_km_s = 299792.458
    G_kpc3_Msun_s2 = 4.30091e-6  # G in (kpc/M_sun) (km/s)²
    
    D_lens_kpc = D_lens_Mpc * 1000.0
    D_source_kpc = D_source_Mpc * 1000.0
    D_LS_kpc = D_LS_Mpc * 1000.0
    
    Sigma_crit = (c_km_s**2 / (4 * np.pi * G_kpc3_Msun_s2)) * (
        D_source_kpc / (D_lens_kpc * D_LS_kpc)
    )  # M_sun/kpc²
    
    # Convergence κ = Σ/Σ_crit
    kappa = Sigma_projected / Sigma_crit
    
    # Add external sheet
    kappa_total = kappa + kappa_ext
    
    # Find Einstein radius (κ crossing from ≥1 to <1)
    if np.any(kappa_total >= 1.0):
        idx = np.where(kappa_total >= 1.0)[0]
        j = idx[-1]  # last index where κ >= 1
        if j < len(R_kpc) - 1:
            # Linear interpolation between j and j+1
            k0, k1 = kappa_total[j], kappa_total[j+1]
            r0, r1 = R_kpc[j], R_kpc[j+1]
            if k0 == k1:
                R_E_kpc = r0
            else:
                R_E_kpc = r0 + (1.0 - k0) * (r1 - r0) / (k1 - k0)
        else:
            R_E_kpc = R_kpc[j]
    else:
        # No Einstein ring (κ < 1 everywhere)
        return 0.0
    
    # Convert to arcsec
    theta_E_arcsec = (R_E_kpc / (D_lens_Mpc * 1000)) * (180 * 3600 / np.pi)
    
    return float(theta_E_arcsec)


def build_hierarchical_model(
    catalog: pd.DataFrame,
    fix_gamma: Optional[float] = None,
    use_triaxial: bool = True
) -> pm.Model:
    """
    Build PyMC hierarchical model with mass-scaled coherence.
    
    Args:
        catalog: Cluster catalog DataFrame
        fix_gamma: If not None, fix γ to this value (for comparison)
        use_triaxial: Include triaxial geometry parameters
        
    Returns:
        PyMC model
    """
    if not HAS_PYMC:
        raise RuntimeError("PyMC not available")
    
    N_clusters = len(catalog)
    
    # Cluster-specific κ_ext prior widths via overrides (default 0.03)
    sigma_kappa_vec = np.full(N_clusters, 0.03, dtype=float)
    for i in range(N_clusters):
        override = load_cluster_override(catalog.iloc[i]['cluster_name'])
        if override and isinstance(override.get('kappa_ext_sigma'), (int, float)):
            sigma_kappa_vec[i] = float(override['kappa_ext_sigma'])

    # Precompute cluster-specific numerics outside the PyMC graph
    R_kpc_grid = np.linspace(1, 2000, 200)
    pre_R500 = catalog['R500_Mpc'].values.astype(float)
    pre_Dlens = np.zeros(N_clusters, dtype=float)
    pre_Dsource = np.zeros(N_clusters, dtype=float)
    pre_Dls = np.zeros(N_clusters, dtype=float)
    pre_Sigma_bar = []

    for i in range(N_clusters):
        row = catalog.iloc[i]
        override = load_cluster_override(row['cluster_name'])
        pre_Sigma_bar.append(compute_baryon_surface_density(row, R_kpc_grid, override=override))
        D_lens, D_source, D_LS = lensing_utils.effective_distances(
            z_lens=row['z_lens'], z_source=row.get('z_source', None), override=override
        )
        pre_Dlens[i] = D_lens
        pre_Dsource[i] = D_source
        pre_Dls[i] = D_LS

    pre_Sigma_bar = np.array(pre_Sigma_bar, dtype=float)  # shape (N, len(R))

    # Black-box op to compute theta_E for all clusters given parameters
    @as_op(itypes=[pt.dvector, pt.dscalar, pt.fscalar, pt.dvector, pt.dvector, pt.dvector], otypes=[pt.dvector])
    def theta_E_model_op(A_c_vec, ell0_star_kpc, gamma_val, q_los_vec, q_plane_vec, kappa_ext_vec):
        A_c_np = np.asarray(A_c_vec, dtype=float)
        qlos_np = np.asarray(q_los_vec, dtype=float)
        qpl_np = np.asarray(q_plane_vec, dtype=float)
        kappa_np = np.asarray(kappa_ext_vec, dtype=float)
        out = np.zeros(A_c_np.shape[0], dtype=float)
        for j in range(A_c_np.shape[0]):
            ell0 = float(ell0_star_kpc) * (pre_R500[j] / 1.0)**float(gamma_val)
            out[j] = compute_theta_E_triaxial(
                pre_Sigma_bar[j], R_kpc_grid, float(A_c_np[j]), float(ell0),
                float(qlos_np[j]), float(qpl_np[j]), float(kappa_np[j]),
                float(pre_Dlens[j]), float(pre_Dsource[j]), float(pre_Dls[j])
            )
        return out

    with pm.Model() as model:
        # Population-level priors
        mu_A = pm.Normal('mu_A', mu=16.5, sigma=1.5)
        sigma_A = pm.HalfNormal('sigma_A', sigma=1.0)
        
        # Coherence length at pivot (1 Mpc)
        ell_0_star_kpc = pm.Lognormal(
            'ell_0_star_kpc',
            mu=np.log(200),
            sigma=0.5
        )
        
        # Mass-scaling exponent
        if fix_gamma is not None:
            gamma = pm.Deterministic('gamma', pm.math.constant(fix_gamma))
        else:
            gamma = pm.Uniform('gamma', lower=0.0, upper=1.0)
        
        # Geometry population parameters (if used)
        if use_triaxial:
            mu_q = pm.Normal('mu_q', mu=1.0, sigma=0.1)
            sigma_q = pm.HalfNormal('sigma_q', sigma=0.1)
        
        # Per-cluster parameters
        A_c = pm.Normal('A_c', mu=mu_A, sigma=sigma_A, shape=N_clusters)
        
        if use_triaxial:
            q_LOS = pm.TruncatedNormal(
                'q_LOS',
                mu=mu_q,
                sigma=sigma_q,
                lower=0.7,
                upper=1.4,
                shape=N_clusters
            )
            q_plane = pm.TruncatedNormal(
                'q_plane',
                mu=mu_q,
                sigma=sigma_q,
                lower=0.7,
                upper=1.4,
                shape=N_clusters
            )
        else:
            q_LOS = pm.Deterministic('q_LOS', pm.math.ones(N_clusters))
            q_plane = pm.Deterministic('q_plane', pm.math.ones(N_clusters))
        
        # External convergence with per-cluster sigma if specified
        kappa_ext = pm.Normal('kappa_ext', mu=0.0, sigma=sigma_kappa_vec, shape=N_clusters)
        
        # Black-box forward model
        theta_E_model = theta_E_model_op(A_c, ell_0_star_kpc, gamma, q_LOS, q_plane, kappa_ext)
        pm.Deterministic('theta_E_model', theta_E_model)

        # Intrinsic scatter (arcsec) to capture unmodeled systematics
        sigma_int = pm.HalfNormal('sigma_int', sigma=5.0)

        # Likelihood with intrinsic scatter
        sigma_obs = catalog['sigma_theta_E'].values
        sigma_tot = pm.math.sqrt(sigma_int**2 + sigma_obs**2)
        pm.Normal(
            'theta_E_obs',
            mu=theta_E_model,
            sigma=sigma_tot,
            observed=catalog['theta_E_obs'].values
        )
        
    return model


def compute_baryon_surface_density(cluster_row: pd.Series, R_kpc: np.ndarray, override: Optional[Dict] = None) -> np.ndarray:
    """
    Compute baryon surface density from gas + BCG/ICL, with optional overrides.
    
    TODO: Load real profiles from data/
    For now, use gNFW gas + Hernquist BCG placeholders, plus optional extra components.
    """
    # If a per-cluster projected baryon profile exists, use it directly (gas); add BCG if override provides
    df_prof = load_baryon_profile(cluster_row['cluster_name'])
    if df_prof is not None:
        Sigma_prof = interpolate_baryon(df_prof, R_kpc)
        # Mass-normalize to target baryon fraction within R500
        try:
            R500_kpc = float(cluster_row.get('R500_Mpc', 1.0)) * 1000.0
            M500_Msun = float(cluster_row.get('M500_1e14Msun', 5.0)) * 1.0e14
            f_b = float(override.get('target_fb', 0.15)) if override else 0.15
            R_max = min(R500_kpc, float(R_kpc[-1]))
            mask = R_kpc <= R_max
            M_proj = 2.0 * np.pi * np.trapz(Sigma_prof[mask] * R_kpc[mask], R_kpc[mask])
            target = f_b * M500_Msun
            if M_proj > 0 and target > 0:
                Sigma_prof = Sigma_prof * (target / M_proj)
        except Exception:
            pass
        # Optional BCG on top of gas profile
        if override and 'bcg' in override:
            M_BCG_csv = float(override['bcg'].get('M_Msun', 0.0))
            a_BCG_csv = float(override['bcg'].get('a_kpc', 15.0))
            Sigma_BCG_csv = (M_BCG_csv / (2 * np.pi * a_BCG_csv**2)) / (1 + R_kpc/a_BCG_csv)**3
            return Sigma_prof + Sigma_BCG_csv
        return Sigma_prof

    # Cluster mass scaling fallback
    M500_e14 = float(cluster_row.get('M500_1e14Msun', 5.0))
    
    # Gas (gNFW) placeholder scaled by mass
    r_s = 300  # kpc, scale radius
    rho_0 = 5e7 * M500_e14  # M_sun/kpc³, mass-scaled normalization
    alpha, beta, gamma_gnfw = 1.0, 3.0, 1.0
    
    rho_gas_3d = rho_0 / ((R_kpc/r_s)**gamma_gnfw * (1 + (R_kpc/r_s)**alpha)**((beta-gamma_gnfw)/alpha))
    
    # Approximate projection to surface density
    Sigma_gas = 2 * rho_gas_3d * r_s  # Rough approximation

    # Optional second gNFW component from overrides (cluster-specific substructure)
    if override and 'second_gnfw' in override:
        sg = override['second_gnfw']
        r_s2 = float(sg.get('r_s_kpc', 150.0))
        rho02 = float(sg.get('rho0_Msun_kpc3', 1e7))
        a2 = float(sg.get('alpha', 1.0)); b2 = float(sg.get('beta', 3.0)); g2 = float(sg.get('gamma', 1.0))
        rho2 = rho02 / ((R_kpc/r_s2)**g2 * (1 + (R_kpc/r_s2)**a2)**((b2-g2)/a2))
        Sigma_gas = Sigma_gas + 2 * rho2 * r_s2
    
    # BCG (Hernquist) with optional override; mass-scaled default
    M_BCG = 2.0e12 * (M500_e14 / 5.0)  # M_sun
    a_BCG = 15.0                       # kpc
    if override and 'bcg' in override:
        M_BCG = float(override['bcg'].get('M_Msun', M_BCG))
        a_BCG = float(override['bcg'].get('a_kpc', a_BCG))
    Sigma_BCG = (M_BCG / (2 * np.pi * a_BCG**2)) / (1 + R_kpc/a_BCG)**3
    
    Sigma_total = Sigma_gas + Sigma_BCG
    
    # Extra baryon components from overrides (e.g., substructures)
    if override and 'extra_baryon_components' in override:
        for comp in override['extra_baryon_components']:
            comp_type = comp.get('type', '').lower()
            if comp_type == 'hernquist':
                M = float(comp.get('M_Msun', 0.0))
                a = float(comp.get('a_kpc', 1.0))
                Sigma_comp = (M / (2 * np.pi * a**2)) / (1 + R_kpc/a)**3
                Sigma_total = Sigma_total + Sigma_comp
            # Future: support gNFW, NFW, etc.
    
    # Normalize total projected mass within R500 to cosmic baryon fraction of M500
    try:
        R500_kpc = float(cluster_row.get('R500_Mpc', 1.0)) * 1000.0
        # Limit integration to grid max
        R_max = min(R500_kpc, float(R_kpc[-1]))
        mask = R_kpc <= R_max
        M_proj = 2.0 * np.pi * np.trapz(Sigma_total[mask] * R_kpc[mask], R_kpc[mask])
        M500_Msun = M500_e14 * 1.0e14
        f_b = 0.15
        target_baryon_mass = f_b * M500_Msun
        if M_proj > 0 and target_baryon_mass > 0:
            scale = target_baryon_mass / M_proj
            Sigma_total = Sigma_total * scale
    except Exception:
        pass
    
    return Sigma_total


def run_inference(
    catalog: pd.DataFrame,
    fix_gamma: Optional[float] = None,
    use_triaxial: bool = True,
    draws: int = 4000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    output_dir: Path = Path("output/mass_scaled")
) -> Tuple[az.InferenceData, Dict]:
    """
    Run MCMC inference and save outputs.
    
    Returns:
        (trace, summary_dict)
    """
    print("="*60)
    print("MASS-SCALED HIERARCHICAL INFERENCE")
    print("="*60)
    print(f"Clusters: {len(catalog)}")
    print(f"Gamma: {'FIXED at ' + str(fix_gamma) if fix_gamma is not None else 'FREE'}")
    print(f"Geometry: {'TRIAXIAL' if use_triaxial else 'SPHERICAL'}")
    print(f"Draws: {draws}, Tune: {tune}, Chains: {chains}")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model
    model = build_hierarchical_model(catalog, fix_gamma, use_triaxial)
    
    # Sample (use Metropolis for black-box likelihood)
    with model:
        step = pm.DEMetropolisZ()
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            step=step,
            return_inferencedata=True,
            cores=min(chains, 2),
            idata_kwargs={"log_likelihood": True}
        )
    
    # Save trace
    trace.to_netcdf(output_dir / "trace.netcdf")
    
    # Compute diagnostics
    summary = az.summary(trace, hdi_prob=0.68)
    summary.to_csv(output_dir / "summary.csv")
    
    # Compute WAIC/LOO (may be unavailable with black-box ops)
    waic_val = None
    loo_val = None
    waic_se = None
    loo_se = None
    try:
        waic = az.waic(trace)
        waic_val = float(waic.elpd_waic)
        waic_se = float(waic.se)
    except Exception as e:
        print(f"Warning: WAIC unavailable ({e})")
    try:
        loo = az.loo(trace)
        loo_val = float(loo.elpd_loo)
        loo_se = float(loo.se)
    except Exception as e:
        print(f"Warning: LOO unavailable ({e})")
    
    # Save metrics
    metrics = {
        'n_clusters': len(catalog),
        'gamma_fixed': fix_gamma is not None,
        'gamma_value': float(fix_gamma) if fix_gamma is not None else None,
        'use_triaxial': use_triaxial,
        'waic': waic_val,
        'waic_se': waic_se,
        'loo': loo_val,
        'loo_se': loo_se,
        'n_divergences': int(trace.sample_stats['diverging'].sum()) if 'diverging' in trace.sample_stats else None,
        'mean_tree_depth': float(trace.sample_stats['tree_depth'].mean()) if 'tree_depth' in trace.sample_stats else None
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nPosterior summary (68% HDI):")
    print(summary[['mean', 'hdi_16%', 'hdi_84%', 'r_hat']])
    
    if metrics['waic'] is not None:
        print(f"\nWAIC: {metrics['waic']:.1f} ± {metrics['waic_se']:.1f}")
    else:
        print("\nWAIC: unavailable (black-box likelihood)")
    if metrics['loo'] is not None:
        print(f"LOO: {metrics['loo']:.1f} ± {metrics['loo_se']:.1f}")
    else:
        print("LOO: unavailable (black-box likelihood)")
    print(f"Divergences: {metrics['n_divergences']}")
    
    print(f"\nOutputs saved to: {output_dir}")
    
    return trace, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical inference for mass-scaled cluster coherence length"
    )
    parser.add_argument('--tiers', type=str, default='1,2',
                        help='Comma-separated tier list (default: 1,2)')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated cluster names to exclude')
    parser.add_argument('--fix-gamma', type=float, default=None,
                        help='Fix gamma to this value (for model comparison)')
    parser.add_argument('--use-triaxial', type=int, default=1,
                        help='Include triaxial geometry (1=yes, 0=no)')
    parser.add_argument('--draws', type=int, default=4000)
    parser.add_argument('--tune', type=int, default=2000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target_accept', type=float, default=0.9)
    parser.add_argument('--out', type=str, default='output/mass_scaled',
                        help='Output directory')
    parser.add_argument('--catalog', type=str, default=None,
                        help='Path to cluster catalog CSV')
    parser.add_argument('--include', type=str, default=None,
                        help='Comma-separated cluster names to include (filters after tiers/exclude)')
    
    args = parser.parse_args()
    
    if not HAS_PYMC:
        print("ERROR: PyMC not available. Install with:", file=sys.stderr)
        print("  pip install pymc arviz", file=sys.stderr)
        sys.exit(1)
    
    # Parse arguments
    tiers = [int(t) for t in args.tiers.split(',')]
    exclude = args.exclude.split(',') if args.exclude else None
    include = args.include.split(',') if args.include else None
    catalog_path = Path(args.catalog) if args.catalog else None
    output_dir = Path(args.out)
    
    # Load catalog
    print("Loading cluster catalog...")
    catalog = load_cluster_catalog(tiers, exclude, catalog_path, include)
    print(f"Loaded {len(catalog)} clusters")
    
    # Run inference
    trace, metrics = run_inference(
        catalog,
        fix_gamma=args.fix_gamma,
        use_triaxial=bool(args.use_triaxial),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Check convergence: az.plot_trace(trace)")
    print(f"2. Posterior predictive: scripts/run_holdout_validation.py")
    print(f"3. Model comparison: Compare WAIC between fixed/free gamma")


if __name__ == '__main__':
    main()

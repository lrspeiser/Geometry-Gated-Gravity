# Phase A Complete — Immediate Fixes Roadmap

## Executive Summary

**Phase A (N=6 training) achieved:**
- ✅ Mass-scaling detection: γ = 0.389 [0.195, 0.654]
- ✅ Training fit quality: χ²/d.o.f. = 3.3, median residual ~8%
- ✅ NFW M_200c → M_500 conversions implemented
- ✅ Per-cluster geometry sampling (q_plane, q_LOS, κ_ext)
- ❌ **Holdout validation FAILED:** A1689 (CI collapse), MACS1149 (+3.8σ)

**Root causes identified:**
1. **Software bug:** A1689 posterior predictive CI collapsed to zero width
2. **Missing physics:** BCG/ICL stellar mass (~10-15% boost), P(z_s) lensing efficiency (~5-10%)
3. **Sample size:** N=6 insufficient for robust hierarchical calibration

---

## Immediate Fixes (Priority Order)

### Fix 1: Posterior Predictive Sampling Bug ✅ DIAGNOSED

**Problem:**  
A1689 68% CI = [46.6, 46.6] arcsec (zero width) indicates deterministic forward model reusing cached Σ maps without propagating geometry/hyperparameter uncertainty.

**Solution:**
```python
# In validate_holdout_mass_scaled.py
for i in range(N_posterior_draws):
    # Sample from posterior
    ell0_star_i = ell0_star_post[i]
    gamma_i = gamma_post[i]
    A_c_i = np.random.normal(mu_A_post[i], sigma_A_post[i])
    
    # Sample geometry per draw (don't use fixed defaults)
    q_plane_i = np.random.normal(1.0, 0.2)
    q_LOS_i = np.random.normal(1.0, 0.2)
    kappa_ext_i = np.random.normal(0.0, 0.03)
    
    # Compute ell0 with mass-scaling
    ell0_cluster = ell0_star_i * (R_500 / 1000.0)**gamma_i
    
    # Build Sigma_bar with sampled geometry (recompute, don't cache)
    Sigma_bar_i = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_LOS_i, q_plane_i)
    
    # Predict theta_E
    theta_E_pred_i = predict_theta_E_full(Sigma_bar_i, ell0_cluster, A_c_i, kappa_ext_i, ...)
    theta_E_samples.append(theta_E_pred_i)
```

**Expected impact:** σ(θ_E) ~ 2-4 arcsec for both A1689 and MACS1149

---

### Fix 2: Add BCG/ICL Stellar Mass ✅ IMPLEMENTED

**Component added:** `core/bcg_profiles.py`

**Physics:**
- Hernquist profile: ρ(r) ∝ 1/(r(r+a)³)
- Empirical scaling: log₁₀(M_BCG/M☉) = 0.4 × log₁₀(M_500/10¹⁴ M☉) + 11.9 (Huang+2018)
- Typical: M_BCG ~ 2×10¹² M☉, R_eff ~ 15 kpc

**Impact estimate:**
- MACS0416 (M_500=1.15×10¹⁵): M_BCG = 2.1×10¹² M☉
- Central Σ boost: **+10-15%** at R < 50 kpc
- θ_E boost: **+8-12%** (roughly scales with √M)

**Integration:**
```python
# In build_cluster_baryons.py or equivalent
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

M_BCG, r_eff = estimate_bcg_mass(M_500, z_lens)
Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff)
Sigma_total = Sigma_baryons + Sigma_BCG  # Add to gas + stellar
```

---

### Fix 3: Source Redshift Distribution P(z_s)

**Current:** Single effective z_s = 2.0 (or 1.67 for CL0024)

**Improvement:** Effective lensing efficiency
```
⟨D_LS/D_S⟩ = ∫ P(z_s) × [D_LS(z_l, z_s) / D_S(z_s)] dz_s
```

**Implementation options:**

**A. Generic CLASH/HFF P(z_s):**
```python
# Typical strong-lensing source distribution (Williams+2017)
z_s_grid = np.linspace(0.5, 6.0, 100)
P_z_s = (z_s_grid - z_lens)**2 * np.exp(-(z_s_grid - z_lens) / 1.5)
P_z_s /= np.trapz(P_z_s, z_s_grid)  # Normalize

# Compute effective ratio
D_LS_arr = [cosmo.angular_diameter_distance_LS(z_lens, zs) for zs in z_s_grid]
D_S_arr = [cosmo.angular_diameter_distance(zs) for zs in z_s_grid]
ratio_eff = np.trapz(P_z_s * (D_LS_arr / D_S_arr), z_s_grid)

Sigma_crit_eff = (c²/(4πG)) / ratio_eff  # Effective critical density
```

**B. Cluster-specific arc catalogs (if available):**
Use measured photo-z or spec-z distributions from HFF/CLASH catalogs.

**Expected impact:** +5-10% on θ_E for high-z clusters (z_lens > 0.4)

---

### Fix 4: Expand Training to N=10

**Add Tier 3 clusters:**
- A383 (z=0.188, M_500=0.87×10¹⁵)
- RXJ2129 (z=0.235, M_500=0.69×10¹⁵) — **has NFW data!**
- MACS0329 (z=0.450, M_500=1.48×10¹⁵) — **has NFW data!**
- A611 (z=0.288, M_500=0.94×10¹⁵) — **has NFW data!**

**Benefits:**
1. **Hierarchical shrinkage reduction:** σ_A and γ posteriors will tighten by ~√(10/6) ≈ 1.3×
2. **BIC/AIC power:** ΔBIC threshold of -6 becomes meaningful with N=10
3. **Mass/redshift coverage:** extends to lower masses (0.7-0.9 ×10¹⁵ M☉)

**Expected γ precision:** 0.39 ± 0.20 (vs current ± 0.23)

---

## Phase B Priorities (After Immediate Fixes)

### B.1: Rerun Full Pipeline with All Fixes

```bash
# 1. Update baryon model builder to include BCG
python scripts/update_baryon_models_with_bcg.py

# 2. Rerun mass-scaled inference (N=10, BCG, P(z_s))
python scripts/run_mass_scaled_emcee.py \
    --catalog data/clusters/master_catalog_nfw.csv \
    --tiers 1,2,3 --exclude NONE --holdout A1689,MACS1149 \
    --use_bcg 1 --use_source_dist 1 \
    --outdir output/mass_scaled_n10_full --seed 42

# 3. Rerun baseline for comparison
python scripts/run_hierarchical_tier12_mcmc.py \
    --catalog data/clusters/master_catalog_nfw.csv \
    --tiers 1,2,3 --exclude NONE --holdout A1689,MACS1149 \
    --use_bcg 1 --fixed_ell0 200 \
    --outdir output/hierarchical_n10_baseline --seed 42

# 4. Model comparison
python scripts/compare_model_predictions.py \
    --baseline output/hierarchical_n10_baseline \
    --mass_scaled output/mass_scaled_n10_full \
    --outdir output/model_comparison_n10

# 5. Blind holdout validation (fixed sampling)
python scripts/validate_holdout_mass_scaled_fixed.py \
    --posterior output/mass_scaled_n10_full/flat_samples.npy \
    --catalog data/clusters/master_catalog_nfw.csv \
    --outdir output/holdout_validation_n10
```

**Acceptance criteria:**
- ✅ A1689: |Z-score| ≤ 1.0
- ✅ MACS1149: |Z-score| ≤ 2.0 (ideally ≤1.5 after BCG + P(z_s))
- ✅ Model comparison: ΔBIC ≤ -6 (strong evidence) OR blind validation passes

---

### B.2: Posterior Predictive Checks (PPCs)

**Mass trend:**
```python
residual_frac = (theta_E_obs - theta_E_pred) / theta_E_obs
plt.scatter(M_500, residual_frac, c=z_lens, cmap='viridis')
plt.xlabel('M_500 (Msun)')
plt.ylabel('Fractional residual')
plt.colorbar(label='z_lens')
```

**Redshift trend:**
```python
plt.scatter(z_lens, residual_frac, s=M_500/1e14, alpha=0.7)
plt.xlabel('z_lens')
plt.ylabel('Fractional residual')
```

**Pass if:** No significant slope (|β| < 0.1 per unit log₁₀M or per Δz)

---

### B.3: Optional Extensions (Only If Needed)

**If MACS1149 still >2σ after BCG + P(z_s):**

1. **Redshift evolution of ℓ₀:**
   ```
   ℓ₀(M, z) = ℓ₀,⋆ × (R_500 / 1 Mpc)^γ × (1+z)^η
   ```
   Prior: η ~ N(0, 0.2²)
   
   Keep only if ΔBIC ≤ -6

2. **Multi-component fit for MACS1149:**
   - Check HFF lensing model for substructure
   - If present, model as 2-component Σ with separate centers

---

## Code Changes Required

### 1. Update `build_cluster_baryons.py`

Add BCG component:
```python
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

def build_cluster_baryon_model_with_bcg(r_3d, params, R_grid_2d=None, apply_bcg=True):
    """Extended version with optional BCG/ICL component."""
    # Build gas + stellar as before
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False)
    
    if apply_bcg and R_grid_2d is not None:
        M_BCG, r_eff = estimate_bcg_mass(params.M_500, params.z)
        Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff)
        
        # Add to total (already projected)
        components.Sigma_total = components.Sigma_total + Sigma_BCG
        components.M_BCG = M_BCG
        components.r_eff_BCG = r_eff
    
    return components
```

### 2. Update `run_mass_scaled_emcee.py`

Add CLI flags:
```python
parser.add_argument('--use_bcg', type=int, default=1, help='Include BCG/ICL (0=off, 1=on)')
parser.add_argument('--use_source_dist', type=int, default=1, help='Use P(z_s) (0=single z_s, 1=distribution)')
```

Wire into `build_cache()`:
```python
if args.use_bcg:
    # Build Sigma with BCG component
    M_BCG, r_eff = estimate_bcg_mass(cluster['M_500_Msun'], cluster['z_lens'])
    Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff)
    Sigma_bar = Sigma_baryons + Sigma_BCG
```

### 3. Fix `validate_holdout_mass_scaled.py`

Replace deterministic geometry with sampling:
```python
# OLD (causes CI collapse):
q_plane_default, q_LOS_default = 0.9, 1.0
theta_E_pred = predict_theta_E(cluster_name, ell0_cluster, A_c, q_plane_default, q_LOS_default)

# NEW (proper sampling):
q_plane_i = np.random.normal(1.0, 0.2)
q_LOS_i = np.random.normal(1.0, 0.2)
kappa_ext_i = np.random.normal(0.0, 0.03)

# Recompute Sigma_bar per draw (don't cache by name only)
Sigma_bar_i = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_LOS_i, q_plane_i)
theta_E_pred_i = predict_theta_E_with_Sigma(Sigma_bar_i, ell0_cluster, A_c, kappa_ext_i, ...)
```

---

## Timeline Estimate

| Task | Time | Dependencies |
|------|------|--------------|
| Fix posterior sampling bug | 1 hr | None |
| Integrate BCG into training | 2 hrs | bcg_profiles.py (done) |
| Add P(z_s) effective lensing | 2 hrs | cosmo utils |
| Expand to N=10 (catalog prep) | 1 hr | NFW data |
| Rerun full pipeline (N=10) | 1 hr compute | All fixes |
| Blind validation v2 | 30 min | Fixed sampling |
| **Total** | **~7-8 hours** | |

---

## Success Metrics

**Minimum acceptable (Phase B gate):**
- ✅ Holdout: ≥1/2 within 1σ, both within 2σ
- ✅ Training: χ²/d.o.f. ≤ 3.5, no mass/z trend in residuals
- ✅ γ detection: 68% CI excludes zero

**Publication-ready (Phase C gate):**
- ✅ Holdout: ≥2/3 within 1σ (with N=10+ training)
- ✅ Model selection: ΔBIC ≤ -6 for mass-scaling
- ✅ Cross-scale consistency: RAR scatter ≤0.11 dex with cluster-calibrated ℓ₀(M)

---

## Notes

- **Don't add more free parameters** until BCG + P(z_s) are in — avoid confounding γ
- **Document every systematic** (BCG mass prior, P(z_s) assumed shape) for referee response
- **Keep universal kernel fixed** (A_c, p=2, n_coh=2) while correcting optics/baryons

End of Phase A roadmap. Ready for immediate fixes implementation.

# Hierarchical Cluster Calibration - Complete Roadmap
**Baryons-Only, Geometry-Aware, Many-Paths Framework**

Date: 2025-01-14  
Status: ✅ Framework created, unified physics confirmed, ready for implementation

---

## Executive Summary

**Completed:**
1. ✅ Fixed critical clumping bug (now divides by sqrt(C) correctly)
2. ✅ Unified physics everywhere (C0=1.3, C_max=2.5 clumping)
3. ✅ Verified predictions with unified physics:
   - Interior-only: 16.74" (-44% vs 30" observed)
   - Both families: 95.13" (+217% overprediction)
4. ✅ Created `hierarchical_cluster_calibration.py` framework

**Key Insight:** Interior-only systematically underpredicts. We need increased A_c OR small w_exterior to reach 30".

**Next Step:** Wire up triaxial lensing computation and launch first hierarchical fit.

---

## Implementation Phases

### Phase 0: Foundation ✅ COMPLETE

**What Was Done:**
- Fixed `gas_profiles.py::apply_clumping_correction` to divide by sqrt(C)
- Unified `test_macs0416_full_physics.py` to use `build_cluster_baryons` model
- Confirmed MACS0416 now predicts 16.74" (interior-only), matching blind suite
- Created hierarchical calibration framework with:
  - Global kernel parameters (A_c, ell0, p_density, n_coh, w_exterior)
  - Per-cluster geometry (q_plane, q_LOS, clumping, BCG/ICL scatter)
  - Train/hold-out split
  - Sparsity prior on w_exterior
  - Joint loss (theta_E + gamma_t + priors)

**Files:**
- ✅ `core/gas_profiles.py` - Fixed clumping
- ✅ `scripts/test_macs0416_full_physics.py` - Unified physics
- ✅ `core/hierarchical_cluster_calibration.py` - Framework
- ✅ `CLUMPING_BUG_CRITICAL.md` - Bug documentation
- ✅ `CLUMPING_FIX_ANALYSIS.md` - Analysis

---

### Phase 1: Triaxial Lensing ⬜ IN PROGRESS

**Goal:** Enable per-cluster geometry (q_plane, q_LOS) to vary

**Tasks:**

#### 1.1 Create triaxial density transformer
File: `core/triaxial_lensing.py`

```python
def spherical_to_triaxial_density(
    rho_spherical: Callable,  # rho(r)
    q_plane: float,  # b/a (in-plane)
    q_LOS: float,    # c/a (line-of-sight)
    phi: float = 0,  # Euler angle (rotation in sky plane)
    theta: float = 0  # Inclination angle
) -> Callable:  # rho(x, y, z)
    """
    Transform spherical density rho(r) to triaxial rho(x,y,z).
    
    Ellipsoidal radius: m² = x² + (y/q_plane)² + (z/q_LOS)²
    Triaxial density: rho(x,y,z) = rho_spherical(m) / (q_plane × q_LOS)
    
    Volume element correction ensures mass conservation.
    """
    def rho_triaxial(x, y, z):
        # Apply rotation (phi, theta) if needed
        x_rot, y_rot, z_rot = rotate_coords(x, y, z, phi, theta)
        
        # Compute ellipsoidal radius
        m = np.sqrt(x_rot**2 + (y_rot/q_plane)**2 + (z_rot/q_LOS)**2)
        
        # Evaluate spherical density at m, apply volume correction
        return rho_spherical(m) / (q_plane * q_LOS)
    
    return rho_triaxial
```

#### 1.2 Extend lensing kernel for triaxial input
Modify `lensing_profiles_3d_shell` to accept 3D density grid instead of 1D rho(r).

Currently: `lensing_profiles_3d_shell(R, z_lens, z_src, r_grid, rho_3d, params, cosmo)`

Change to: Accept `rho_3d_func(x, y, z)` or pre-computed 3D grid.

#### 1.3 Implement projection integrals
```python
def project_triaxial_to_surface_density(
    rho_triaxial: Callable,  # rho(x, y, z)
    R_proj: np.ndarray,  # Projected radii [kpc]
    z_max: float = 5000.0  # Integration limit along LOS
) -> np.ndarray:  # Sigma(R)
    """
    Project triaxial 3D density along line-of-sight.
    
    Sigma(R) = ∫ rho(sqrt(R² - b²), 0, z) dz
    
    For triaxial case, need 2D integration over (y, z) then azimuthal average.
    """
    Sigma = np.zeros_like(R_proj)
    
    for i, R in enumerate(R_proj):
        # Integrate along z-axis through (R, 0, z)
        integrand = lambda z: rho_triaxial(R, 0, z)
        Sigma[i], _ = quad(integrand, -z_max, z_max)
    
    return Sigma
```

#### 1.4 Test triaxial effects on MACS0416
Script: `scripts/test_triaxial_macs0416.py`

```python
# Sweep q_LOS from 0.6 (oblate) to 1.6 (prolate)
q_LOS_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
theta_E_values = []

for q_LOS in q_LOS_values:
    # Build triaxial baryon model
    params = ClusterBaryonParams(M_500=1.15e15, R_500=1200, z=0.396)
    rho_spherical = build_cluster_baryon_model(r_grid, params)
    rho_triaxial = spherical_to_triaxial_density(
        rho_spherical, q_plane=0.9, q_LOS=q_LOS
    )
    
    # Compute lensing
    profiles = lensing_profiles_triaxial(...)
    theta_E = profiles['theta_E_arcsec']
    theta_E_values.append(theta_E)
    
    print(f"q_LOS={q_LOS:.1f}: theta_E={theta_E:.2f}\"")

# Plot theta_E vs q_LOS
# Expected: monotonic increase with q_LOS (elongated clusters focus more)
```

**Expected Outcomes:**
- q_LOS = 0.7 (oblate): theta_E ~ 14-15" (flatter → less focusing)
- q_LOS = 1.0 (sphere): theta_E ~ 17" (baseline, matches current)
- q_LOS = 1.4 (prolate): theta_E ~ 20-22" (elongated → more focusing)

**Deliverables:**
- ✅ `core/triaxial_lensing.py`
- ✅ Updated `core/cluster_kernel_3d_shell.py` (if needed)
- ✅ `scripts/test_triaxial_macs0416.py`
- ✅ Figure: `theta_E_vs_qLOS.png`

---

### Phase 2: Hierarchical Calibration ⬜ NEXT

**Goal:** Fit global kernel + per-cluster geometries on train set

**Prerequisites:**
- Triaxial lensing working (Phase 1)
- `_predict_lensing()` implemented in hierarchical_cluster_calibration.py

#### 2.1 Wire up lensing in framework
Implement `HierarchicalClusterCalibration._predict_lensing()`:

```python
def _predict_lensing(self, cluster_row, global_kernel, geometry):
    # 1. Build triaxial baryon model
    params = ClusterBaryonParams(
        M_500=cluster_row['M_500_Msun'],
        R_500=cluster_row['R_500_kpc'],
        z=cluster_row['z_lens'],
        fgas_target=cluster_row['fgas_R500'],
        T_keV=cluster_row['TX_central_keV'],
        # Use fitted clumping
        C0=geometry.C0,
        C_max=geometry.C_max,
        eta=geometry.eta_clump,
        # Use fitted BCG/ICL scatter
        M_BCG=None,  # Will be scaled by f_BCG_scatter
        M_ICL=None   # Will be scaled by f_ICL_scatter
    )
    
    r_grid = np.logspace(-1, 3.5, 2000)
    components = build_cluster_baryon_model(r_grid, params)
    
    # Apply BCG/ICL scatter
    components.rho_bcg *= geometry.f_BCG_scatter
    components.rho_icl *= geometry.f_ICL_scatter
    components.rho_total = components.rho_gas + components.rho_bcg + components.rho_icl
    
    # 2. Transform to triaxial
    rho_triaxial = spherical_to_triaxial_density(
        lambda r: np.interp(r, r_grid, components.rho_total),
        q_plane=geometry.q_plane,
        q_LOS=geometry.q_LOS
    )
    
    # 3. Compute lensing
    R_proj = np.geomspace(10, 1500, 200)
    
    kernel_params = Shell3DKernelParams(
        A_c=global_kernel.A_c,
        ell0=global_kernel.ell0,
        p_density=global_kernel.p_density,
        n_coh=global_kernel.n_coh,
        w_interior=global_kernel.w_interior,
        w_exterior=global_kernel.w_exterior,
        # ... other params
    )
    
    profiles = lensing_profiles_triaxial(
        R_proj, cluster_row['z_lens'], cluster_row['z_source'],
        rho_triaxial, kernel_params, cosmo
    )
    
    theta_E = profiles['theta_E_arcsec']
    gamma_t = profiles['gamma_t']  # At observed radii if available
    
    return theta_E, gamma_t
```

#### 2.2 Prepare observational data
File: `data/clusters/observational_data.json`

```json
{
  "MACS0416": {
    "theta_E_obs": 30.0,
    "theta_E_err": 1.5,
    "gamma_t_R": null,
    "gamma_t_obs": null,
    "gamma_t_err": null,
    "has_xray_sz": true
  },
  "A1689": {
    "theta_E_obs": 47.0,
    "theta_E_err": 3.0,
    "gamma_t_R": [50, 100, 200, 400, 800],
    "gamma_t_obs": [0.45, 0.28, 0.15, 0.08, 0.04],
    "gamma_t_err": [0.05, 0.03, 0.02, 0.01, 0.01],
    "has_xray_sz": true
  },
  ...
}
```

#### 2.3 Launch calibration
Script: `scripts/run_hierarchical_calibration.py`

```python
# Load data
catalog = pd.read_csv('data/clusters/master_catalog.csv')
with open('data/clusters/observational_data.json') as f:
    obs_dict = json.load(f)

obs_data = {}
for name, data in obs_dict.items():
    obs_data[name] = ObservationalData(
        theta_E_obs=data['theta_E_obs'],
        theta_E_err=data['theta_E_err'],
        gamma_t_R=np.array(data['gamma_t_R']) if data['gamma_t_R'] else None,
        gamma_t_obs=np.array(data['gamma_t_obs']) if data['gamma_t_obs'] else None,
        gamma_t_err=np.array(data['gamma_t_err']) if data['gamma_t_err'] else None,
        has_xray_sz=data['has_xray_sz']
    )

# Initialize calibration
calib = HierarchicalClusterCalibration(catalog, obs_data, verbose=True)
calib.set_train_holdout_split(holdout_fraction=0.25, random_seed=42)

# Fit (3-5 iterations)
calib.fit_hierarchical(n_iterations=5, global_method='differential_evolution')

# Validate
calib.validate_holdout()

# Save
calib.save_results('results/hierarchical_calib_v1')
```

**Expected Results:**
- Global kernel:
  - A_c: 12-18 (increased from 10 to compensate for strong clumping)
  - ell0: 150-220 kpc
  - w_exterior: 0.05-0.15 (small due to sparsity prior)
  - p_density, n_coh: near priors

- Per-cluster geometry:
  - q_LOS: varies 0.7-1.3 (mergers lower, relaxed near 1)
  - q_plane: 0.8-0.95 (mildly oblate on average)
  - Clumping: C0 ~ 1.2-1.4, C_max ~ 2.3-2.7

- Train set performance:
  - Median residual: 10-15%
  - Within ±20%: 7-9 out of 9 clusters

**Deliverables:**
- ✅ `scripts/run_hierarchical_calibration.py`
- ✅ `results/hierarchical_calib_v1/global_kernel.json`
- ✅ `results/hierarchical_calib_v1/cluster_geometries.json`
- ✅ `results/hierarchical_calib_v1/train_residuals.csv`

---

### Phase 3: Hold-Out Validation ⬜

**Goal:** Test universal kernel on unseen clusters

```python
# In run_hierarchical_calibration.py, after training:
calib.validate_holdout()
```

**Analysis:**
1. Compare train vs hold-out residual distributions
   - Similar distributions → good generalization
   - Hold-out much worse → overfitting or non-universal physics

2. Identify outliers
   - Which clusters have >30% error?
   - Do they share properties (merger state, mass, redshift)?

3. Check systematic trends
   - Residual vs M_500 (over/underprediction for massive clusters?)
   - Residual vs z (redshift evolution not captured?)
   - Residual vs dynamical state (mergers harder to fit?)

**Acceptance Criteria:**
- Hold-out median residual < 20%
- No catastrophic outliers (>50%)
- Similar performance to train set

**Deliverables:**
- ✅ `results/hierarchical_calib_v1/holdout_residuals.csv`
- ✅ Figure: Train vs hold-out residual distributions
- ✅ Figure: Residual vs cluster properties (M_500, z, state)

---

### Phase 4: Weak Lensing Integration ⬜

**Goal:** Break shape-amplitude degeneracy with gamma_t(R) data

**Tasks:**

#### 4.1 Add weak lensing data
Sources:
- Umetsu+ 2020 (HSC CAMIRA clusters)
- CLASH survey (Merten+ 2015)
- HLS survey (Schrabback+ 2021)

For each cluster with WL data, add to observational_data.json:
- `gamma_t_R`: Radii [kpc or arcsec]
- `gamma_t_obs`: Measured shear
- `gamma_t_err`: Errors (including shape noise + systematics)

#### 4.2 Implement gamma_t computation
In `lensing_profiles_3d_shell`, add:
```python
# Tangential shear: gamma_t = mean_kappa - kappa
gamma_t = mean_kappa - kappa
```

Already computed internally, just need to return it.

#### 4.3 Re-run calibration with joint loss
```python
# joint_loss() already implements:
chi2_total = chi2_theta_E + w_wl * chi2_gamma_t + w_prior * prior_penalty

# Where w_wl = 1.0 if WL data available, 0.0 otherwise
```

#### 4.4 Analyze posterior correlations
```python
# Extract fitted parameters
A_c_values = [global_kernel_history[i].A_c for i in iterations]
q_LOS_values = [cluster_geom[name][i].q_LOS for name in clusters for i in iterations]

# Plot pairwise correlations
import seaborn as sns
sns.pairplot(df[['A_c', 'ell0', 'q_LOS', 'q_plane', 'w_exterior']])
```

**Expected:**
- Without WL: A_c and q_LOS strongly correlated (degeneracy)
- With WL: Correlation breaks, both tightly constrained

**Deliverables:**
- ✅ Updated `data/clusters/observational_data.json` with WL data
- ✅ `results/hierarchical_calib_wl_v1/` (new run with WL)
- ✅ Figure: Posterior correlations with/without WL
- ✅ Figure: Best-fit gamma_t(R) vs observations for clusters with WL

---

### Phase 5: Ablation Tests & Diagnostics ⬜

**Goal:** Validate physics choices and identify failure modes

#### Test 1: Interior-only vs full model
```python
# Run 1: Freeze w_exterior = 0, fit only A_c
calib_interior_only = ...
calib_interior_only.global_kernel_best.w_exterior = 0.0
calib_interior_only.fit_hierarchical(...)

# Run 2: Full model with w_exterior free
calib_full = ...
calib_full.fit_hierarchical(...)

# Compare:
# - A_c_interior vs A_c_full (expect A_c_interior higher)
# - Train residuals (expect full slightly better if data support it)
```

#### Test 2: No weak lensing
```python
# Remove gamma_t data, re-fit
for name in obs_data:
    obs_data[name].gamma_t_R = None  # Disable WL

calib_no_wl = ...
calib_no_wl.fit_hierarchical(...)

# Check if q_LOS posterior becomes broader (degeneracy)
```

#### Test 3: Freeze clumping
```python
# Fix C0=1.3, C_max=2.5 for all clusters
for name in train_clusters:
    calib.cluster_geometry_best[name].C0 = 1.3
    calib.cluster_geometry_best[name].C_max = 2.5
    # Only fit q_plane, q_LOS, BCG/ICL scatter

# Check if fit quality degrades significantly
```

#### Diagnostic Plots:
1. **Residual waterfall**: theta_E residual vs M_500, z, dynamical state
2. **Geometry distributions**: Histograms of q_LOS, q_plane (color by merger/relaxed)
3. **Kernel evolution**: A_c, ell0, w_exterior vs iteration
4. **Per-cluster profiles**: Sigma(R), K_Sigma(R), gamma_t(R) for best/worst fits

**Deliverables:**
- ✅ `results/ablation_tests/` directory with sub-runs
- ✅ Figures: Residual diagnostics, geometry distributions, kernel evolution
- ✅ Document: `ABLATION_RESULTS.md` summarizing findings

---

### Phase 6: Physical Interpretation ⬜

**Questions:**

#### Q1: Is interior dominance confirmed?
- Check w_exterior posterior: mean ~ 0.05-0.10?
- Compute interior contribution fraction: theta_E_interior / theta_E_full ~ 80-90%?

#### Q2: Do mergers prefer elongated/flattened geometry?
- Compare q_LOS distributions:
  - Relaxed clusters (cool cores): q_LOS ~ 0.9-1.1 (near spherical)
  - Mergers: q_LOS ~ 0.7-0.8 (oblate) or 1.2-1.4 (prolate, if bimodal merger)

#### Q3: Is clumping consistent with X-ray?
- Plot fitted C0 vs X-ray surface brightness concentration
- Expect: relaxed (high concentration) → lower C0
- Mergers (disturbed) → higher C0

#### Q4: Does kernel scale with cluster properties?
- Plot A_c vs <M_500>, <z>: Any trends?
- Plot ell0 vs R_500: Expect ell0 ~ 0.1-0.2 R_500

**Deliverables:**
- ✅ Document: `PHYSICAL_INTERPRETATION.md`
- ✅ Figures: q_LOS vs dynamical state, C0 vs X-ray properties, ell0 vs R_500

---

## File Structure (Final)

```
GravityCalculator/
├── core/
│   ├── build_cluster_baryons.py                ✅ Unified clumping
│   ├── hierarchical_cluster_calibration.py     ✅ Framework
│   ├── triaxial_lensing.py                     ⬜ Phase 1
│   ├── cluster_kernel_3d_shell.py              ✅ 3D kernel
│   └── gas_profiles.py                         ✅ Fixed clumping
├── scripts/
│   ├── test_macs0416_full_physics.py           ✅ Unified physics
│   ├── run_cluster_suite.py                    ✅ Blind suite
│   ├── test_triaxial_macs0416.py               ⬜ Phase 1
│   ├── run_hierarchical_calibration.py         ⬜ Phase 2
│   └── analyze_calibration_results.py          ⬜ Phase 5
├── data/
│   └── clusters/
│       ├── master_catalog.csv                  ✅ 12 clusters
│       └── observational_data.json             ⬜ Phase 2 (add WL)
├── results/
│   ├── cluster_suite_blind_v1/                 ✅ Interior-only results
│   ├── hierarchical_calib_v1/                  ⬜ Phase 2
│   ├── hierarchical_calib_wl_v1/               ⬜ Phase 4
│   └── ablation_tests/                         ⬜ Phase 5
└── docs/
    ├── CLUMPING_BUG_CRITICAL.md                ✅ Bug analysis
    ├── CLUMPING_FIX_ANALYSIS.md                ✅ Post-fix
    ├── HIERARCHICAL_CALIBRATION_COMPLETE_ROADMAP.md ✅ This file
    ├── ABLATION_RESULTS.md                     ⬜ Phase 5
    └── PHYSICAL_INTERPRETATION.md              ⬜ Phase 6
```

---

## Timeline Estimate

- **Phase 1** (Triaxial): 2-3 days
- **Phase 2** (Calibration): 1-2 days
- **Phase 3** (Validation): 1 day
- **Phase 4** (Weak lensing): 2-3 days (data collection + re-fit)
- **Phase 5** (Ablations): 1-2 days
- **Phase 6** (Interpretation): 1 day

**Total: ~2 weeks** for complete hierarchical calibration pipeline

---

## Success Metrics

### Minimum Viable Product (MVP)
- ✅ Unified physics across all tests
- ⬜ Triaxial lensing working
- ⬜ Hierarchical fit: median train residual < 20%
- ⬜ Hold-out: similar to train (no overfitting)

### Gold Standard
- ⬜ Weak lensing data for 6+ clusters
- ⬜ Joint fit: theta_E median < 10%, gamma_t chi^2/dof ~ 1
- ⬜ w_exterior posterior: mean ~ 0.05-0.15, tight
- ⬜ Physical trends confirmed (mergers → q_LOS ≠ 1, etc.)
- ⬜ Population paper-ready diagnostics

---

## Key Physics Principles (Unchanging)

1. **NO dark matter** anywhere in pipeline
2. **gNFW gas** normalized to f_gas(R_500) = 0.11
3. **Clumping**: C0=1.3, C_max=2.5, correct sign (divide by sqrt(C))
4. **Interior chords dominate**: w_interior = 1, w_exterior ~ 0 unless data demand
5. **Kernel universal**: Same A_c, ell0, p_density, n_coh for all clusters
6. **Geometry varies**: q_plane, q_LOS fitted per-cluster with priors

---

**Bottom Line:** Framework is ready. Next immediate action: implement triaxial lensing (Phase 1), then launch hierarchical calibration (Phase 2).


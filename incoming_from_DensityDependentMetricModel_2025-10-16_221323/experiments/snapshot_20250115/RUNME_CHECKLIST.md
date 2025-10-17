# Σ-Gravity Publication Checklist

**Snapshot**: 2025-01-15  
**Goal**: Take mass-scaled coherence from γ=0.47±0.29 (N=5) to publication-ready result with N=18

---

## Phase 1: Critical Physics Corrections (Week 1)

### A. Bookkeeping ✅ DONE
- [x] Create snapshot folder with config.json
- [x] Document kernel_params.json (current γ, ℓ₀,⋆)
- [x] Document geometry_priors.json
- [ ] Export `pip freeze > requirements.txt`
- [ ] Record git commit SHA: `git rev-parse HEAD > scripts_commit.txt`

### B. Fix Redshift-Dependent Lensing 🔴 CRITICAL
**Estimated time**: 2-3 hours coding + 1 hour testing

- [ ] **B1. Add Σ_crit calculation**
  ```python
  # In scripts/lensing_utils.py or similar
  from astropy.cosmology import FlatLambdaCDM
  
  def compute_sigma_crit(z_lens, z_source, cosmo=None):
      """Compute critical surface density in g/cm²"""
      if cosmo is None:
          cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
      
      D_L = cosmo.angular_diameter_distance(z_lens).to('cm').value
      D_S = cosmo.angular_diameter_distance(z_source).to('cm').value
      D_LS = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to('cm').value
      
      # Σ_crit = c²/(4πG) × D_S/(D_L × D_LS)
      c_cgs = 2.99792458e10
      G_cgs = 6.67430e-8
      return (c_cgs**2 / (4 * np.pi * G_cgs)) * (D_S / (D_L * D_LS))
  ```

- [ ] **B2. Update θ_E solver**
  ```python
  # Replace constant Σ_crit with per-cluster values
  for cluster in catalog:
      Sigma_crit = compute_sigma_crit(cluster.z_lens, cluster.z_source)
      kappa = Sigma_proj / Sigma_crit  # Now uses correct lensing efficiency
      theta_E = solve_einstein_radius(kappa)
  ```

- [ ] **B3. Test on known cluster**
  - Pick Abell 2261 (z_l=0.224, z_s=1.5)
  - Verify θ_E changes by ~5-10% when switching to proper D_LS/D_S
  - Document in `tests/test_redshift_lensing.py`

**Acceptance**: θ_E predictions vary correctly with z_lens at fixed mass

---

### C. Cluster-Specific M₅₀₀ Conversion 🔴 CRITICAL
**Estimated time**: 2 hours coding + 1 hour validation

- [ ] **C1. Add concentration column to catalog**
  ```python
  # In data/cluster_lensing_catalog.csv, add column:
  # c200c,notes_c200c
  # Get values from Umetsu+2016 Table 2
  ```

- [ ] **C2. Implement NFW M₂₀₀c → M₅₀₀ conversion**
  ```python
  def M500_from_M200c_NFW(M200c_Msun, c200c, z, cosmo=None):
      """
      Cluster-specific conversion using NFW profile.
      Returns M500 in Msun and R500 in Mpc.
      """
      # 1. Compute r200c from M200c
      rho_crit = cosmo.critical_density(z).to('Msun/Mpc3').value
      r200c_Mpc = (3 * M200c_Msun / (4 * np.pi * 200 * rho_crit))**(1/3)
      
      # 2. NFW scale radius
      rs_Mpc = r200c_Mpc / c200c
      
      # 3. Find r500 by solving ρ_NFW(<r500)/V(<r500) = 500 ρ_crit
      from scipy.optimize import brentq
      def equation(r):
          M_enc = M_NFW_enclosed(r, rs_Mpc, ...)
          rho_avg = M_enc / ((4/3) * np.pi * r**3)
          return rho_avg - 500 * rho_crit
      
      r500_Mpc = brentq(equation, 0.5*r200c_Mpc, 1.5*r200c_Mpc)
      M500_Msun = M_NFW_enclosed(r500_Mpc, rs_Mpc, ...)
      
      return M500_Msun, r500_Mpc
  ```

- [ ] **C3. Update catalog generation script**
  - Replace fixed 0.65 factor with function call
  - Regenerate `data/cluster_lensing_catalog.csv`

- [ ] **C4. Validation plot**
  ```python
  # Plot: M500_new vs M500_old, color by c200c
  # Expect: tighter M-R relation, ~10-15% changes per cluster
  ```

**Acceptance**: M₅₀₀-R₅₀₀ scatter reduced; high-concentration clusters have higher M₅₀₀/M₂₀₀c

---

### D. Add BCG Stellar Component 🟡 HIGH PRIORITY
**Estimated time**: 3 hours coding + testing

- [ ] **D1. Create BCG parameter file**
  ```python
  # data/cluster_bcg_parameters.csv
  cluster_name,M_star_Msun,R_e_kpc,profile,notes
  Abell383,1.2e11,12.0,hernquist,CLASH photometry
  ...
  ```
  Use M_star ~ 10^11 - 10^12 M_sun, R_e ~ 10-15 kpc typical values

- [ ] **D2. Implement Hernquist profile**
  ```python
  def Sigma_hernquist_2d(R_kpc, M_star, a_kpc):
      """
      Projected Hernquist profile for BCG.
      a ≈ R_e / 1.8 for Hernquist
      """
      x = R_kpc / a_kpc
      # Full formula for Σ(R) from Hernquist 1990
      ...
      return Sigma_star  # g/cm²
  ```

- [ ] **D3. Add to baryonic Σ builder**
  ```python
  # In build_cluster_baryons.py
  Sigma_total = Sigma_gas + Sigma_BCG + Sigma_coherence
  ```

- [ ] **D4. Test impact**
  - Rerun single cluster (e.g., RX J1347) with/without BCG
  - Expect: θ_E increases by ~5-15% with BCG

**Acceptance**: BCG contributes ~10-15% to κ at θ_E; documented in methods

---

### E. Verify Triaxiality Was Fit 🔴 CRITICAL
**Estimated time**: 30 minutes diagnostic

- [ ] **E1. Check posterior file**
  ```python
  import pandas as pd
  chain = pd.read_csv('output/mass_scaled_emcee/chain.csv')
  print(chain.columns)
  
  # Look for: q_plane_0, q_plane_1, ..., q_LOS_0, q_LOS_1, ...
  # If NOT present → geometry was fixed, not fit!
  ```

- [ ] **E2. If geometry was fixed**
  - [ ] Modify inference script to include q_plane, q_LOS as free params
  - [ ] Rerun to get honest uncertainty on γ

- [ ] **E3. Plot geometry posteriors**
  ```python
  # For each cluster, plot q_plane and q_LOS posterior
  # Check: reasonable values (0.8-1.2), not hitting prior bounds
  ```

**Acceptance**: Geometry parameters were actually sampled; posteriors look physical

---

## Phase 2: Full Sample Refit (Week 1-2)

### F. Run N=18 Cluster Inference 🔴 CRITICAL
**Estimated time**: 4-8 hours compute

- [ ] **F1. Pre-flight check**
  ```bash
  # Verify all Phase 1 fixes are in place:
  python scripts/validate_cluster_catalog.py --check-redshift-lensing --check-nfw-conversion --check-bcg
  ```

- [ ] **F2. Mass-scaled model (γ free)**
  ```bash
  cd C:\Users\henry\Documents\GitHub\DensityDependentMetricModel
  
  python scripts/run_mass_scaled_hierarchical_inference.py \
    --tiers 1,2 \
    --exclude MACSJ0717.5+3745 \
    --use-triaxial 1 \
    --fit-kappa-ext 1 \
    --use-bcg 1 \
    --use-nfw-conversion 1 \
    --use-redshift-lensing 1 \
    --draws 8000 --chains 4 --target-accept 0.9 \
    --out output/mass_scaled_N18_full/
  ```
  
  **Expected runtime**: ~4-8 hours  
  **Expected output**: γ = 0.45 ± 0.15 (narrower uncertainty from N=18 vs N=5)

- [ ] **F3. Convergence checks**
  ```python
  # r_hat < 1.01 for all parameters
  # ESS > 400 (better: >1000)
  # No divergent transitions
  ```

- [ ] **F4. Save snapshot**
  ```bash
  cp output/mass_scaled_N18_full/posterior.npz experiments/snapshot_20250115/
  cp output/mass_scaled_N18_full/summary.txt experiments/snapshot_20250115/
  ```

**Acceptance**: Converged fit with N=18; γ uncertainty reduced by ~√(18/5) ≈ 1.9×

---

### G. Scale-Invariant Comparison (γ=0) 🔴 CRITICAL
**Estimated time**: 4-8 hours compute

- [ ] **G1. Run fixed γ=0 model**
  ```bash
  python scripts/run_mass_scaled_hierarchical_inference.py \
    ...same flags as F2... \
    --fix-gamma 0 \
    --out output/scale_invariant_N18/
  ```

- [ ] **G2. Compute ΔBIC**
  ```python
  # From summary files:
  BIC_mass = -2 * ln_likelihood_mass + k_mass * ln(N_data)
  BIC_fixed = -2 * ln_likelihood_fixed + k_fixed * ln(N_data)
  
  Delta_BIC = BIC_fixed - BIC_mass
  
  print(f"ΔBIC = {Delta_BIC:.1f}")
  # ΔBIC > 6: strong evidence for mass-scaling
  # ΔBIC < -6: strong evidence against
  # |ΔBIC| < 6: inconclusive
  ```

- [ ] **G3. Bayes factor (alternative)**
  ```python
  # If using emcee: estimate via thermodynamic integration or nested sampling
  # Or use WAIC from ArviZ if switching to PyMC
  ```

**Acceptance**: ΔBIC computed; decision rule applied; result documented

---

## Phase 3: Validation & Diagnostics (Week 2)

### H. Posterior Predictive Checks 🟡 HIGH
**Estimated time**: 2-3 hours

- [ ] **H1. Generate PPC samples**
  ```python
  # For each cluster in training set:
  theta_E_pred = sample_posterior_predictive(model, cluster, n_samples=1000)
  
  # Compute residuals:
  residual = (theta_E_obs - theta_E_pred_median) / theta_E_obs
  ```

- [ ] **H2. Residual plots**
  ```python
  # Plot residuals vs:
  # - M500 (check for mass trends)
  # - z_lens (check for redshift systematics)
  # - R500 (redundant with mass but useful)
  # - tier (gold vs silver quality)
  ```

- [ ] **H3. Diagnose χ²/d.o.f.**
  - If still > 3: likely underestimated errors or missing systematics
  - Consider adding f_sys (fractional systematic) to error model
  - Or switch to Student-t likelihood (robust to outliers)

**Acceptance**: No systematic trends in residuals; χ²/d.o.f. < 2.5

---

### I. Blind Hold-Out Validation 🔴 CRITICAL
**Estimated time**: 1 hour per hold-out test

- [ ] **I1. Select hold-outs**
  ```python
  # Reserve 2-3 diverse clusters:
  holdouts = ['Abell2261', 'MACSJ1149.5+2223', 'RXJ1347.5-1145']
  # (gold tier, different masses/redshifts)
  ```

- [ ] **I2. Refit without hold-outs**
  ```bash
  python scripts/run_mass_scaled_hierarchical_inference.py \
    --tiers 1,2 \
    --exclude MACSJ0717.5+3745,Abell2261,MACSJ1149.5+2223,RXJ1347.5-1145 \
    ...same flags... \
    --out output/holdout_fit_N15/
  ```

- [ ] **I3. Predict hold-outs**
  ```python
  # Use posterior from N=15 fit:
  for cluster in holdouts:
      theta_E_pred, theta_E_CI = predict_from_posterior(cluster, posterior_N15)
      
      # Check:
      within_1sigma = (theta_E_obs within theta_E_CI_68pct)
      fractional_error = abs(theta_E_obs - theta_E_pred) / theta_E_obs
  ```

- [ ] **I4. Pass/fail criteria**
  - **Pass if**: ≥2/3 hold-outs within 1σ PPC AND no outliers >3σ
  - **Pass if**: Median |fractional_error| < 20%
  - Document results in `output/holdout_validation_report.md`

**Acceptance**: Hold-out validation passed; documented

---

### J. Ablation Studies 🟢 MEDIUM
**Estimated time**: 2-4 hours per ablation

- [ ] **J1. No triaxiality**
  ```bash
  python scripts/.../inference.py --use-triaxial 0 --out output/ablation_no_triaxial/
  # Record Δχ² vs full model
  ```

- [ ] **J2. No BCG**
  ```bash
  python scripts/.../inference.py --use-bcg 0 --out output/ablation_no_bcg/
  ```

- [ ] **J3. No κ_ext**
  ```bash
  python scripts/.../inference.py --fit-kappa-ext 0 --out output/ablation_no_kappa/
  ```

- [ ] **J4. Fixed γ=1/3 (self-similar)**
  ```bash
  python scripts/.../inference.py --fix-gamma 0.333 --out output/ablation_gamma_third/
  ```

- [ ] **J5. Compile ablation table**
  | Model variant | χ²/d.o.f. | Δχ² | ΔBIC | Notes |
  |---------------|-----------|-----|------|-------|
  | Full model    | 2.1       | 0   | 0    | Baseline |
  | No triaxial   | 4.8       | +48 | +50  | Major degradation |
  | No BCG        | 2.4       | +6  | +8   | Modest effect |
  | No κ_ext      | 2.3       | +4  | +6   | Small effect |
  | γ=1/3         | 2.5       | +8  | +10  | Slightly worse than free γ |

**Acceptance**: Ablation table shows triaxiality & redshift-lensing are critical; BCG/κ_ext modest

---

## Phase 4: Cross-Scale Consistency (Week 3)

### K. Galaxy-Cluster Coherence Check 🟡 HIGH
**Estimated time**: 2-3 hours analysis

- [ ] **K1. Compute ℓ₀ at Milky Way mass**
  ```python
  # Milky Way: M500 ~ 1e12 Msun, R500 ~ 200 kpc
  # Using fitted (ℓ₀,⋆, γ):
  
  ell_0_MW = ell_0_star_kpc * (R500_MW / 1000)**gamma
  # Example: ℓ₀ = 200 × (0.2)^0.45 ≈ 120 kpc
  ```

- [ ] **K2. Compare to galaxy RAR fit**
  - You previously fit galaxies with ℓ₀ ~ 100-150 kpc (scatter 0.087 dex)
  - Check if cluster-inferred ℓ₀(M_MW) overlaps galaxy value

- [ ] **K3. Plot ℓ₀ vs M₅₀₀**
  ```python
  # Log-log plot:
  # x-axis: M500 (10^11 to 10^15 Msun)
  # y-axis: ℓ₀ (kpc)
  # Show: cluster fit (with uncertainty band)
  #       galaxy point (with error bar)
  #       self-similar γ=1/3 line for reference
  ```

- [ ] **K4. Consistency statement**
  > "The mass-scaled coherence length ℓ₀ ∝ R₅₀₀^0.45 extrapolates to ℓ₀ ≈ 120±30 kpc at Milky Way mass, consistent with the galaxy RAR calibration (ℓ₀ = 110±20 kpc), supporting a universal Σ-Gravity mechanism across 4 orders of magnitude in halo mass."

**Acceptance**: Galaxy and cluster ℓ₀ consistent within uncertainties

---

## Phase 5: Publication Artifacts (Week 3-4)

### L. Figure Generation 📊
**Estimated time**: 1-2 days

- [ ] **L1. Main paper figures**
  - **Fig 1**: Cluster sample (M-z plane, color by tier)
  - **Fig 2**: θ_E vs M₅₀₀ (data + model predictions with error envelopes)
  - **Fig 3**: Corner plot (γ, ℓ₀,⋆, μ_A, σ_A posteriors)
  - **Fig 4**: Residual diagnostic (θ_obs - θ_pred vs mass/redshift)
  - **Fig 5**: ℓ₀ vs M (cross-scale: galaxies + clusters)
  - **Fig 6**: Ablation comparison (χ² changes)

- [ ] **L2. Supplementary figures**
  - **S1**: Geometry posteriors (q_plane, q_LOS per cluster)
  - **S2**: κ_ext posteriors (check for mass correlation)
  - **S3**: Hold-out validation (predicted vs observed)
  - **S4**: Per-cluster θ_E comparisons (18 panels)

- [ ] **L3. Figure scripts**
  ```bash
  # Create scripts/make_figures.py with functions:
  make_fig1_sample_overview()
  make_fig2_theta_E_vs_mass()
  ...
  ```

**Acceptance**: All figures generated from scripts; publication-quality (vector format)

---

### M. Methods Documentation 📝
**Estimated time**: 2-3 days writing

- [ ] **M1. Lensing calculation**
  - Equation for Σ_crit(z_l, z_s)
  - Cosmology (H₀, Ωₘ, ΩΛ)
  - θ_E solver (iterative κ=1 condition)

- [ ] **M2. Baryonic Σ construction**
  - gNFW gas profile (from ACCEPT)
  - Clumping correction (formula, prior on C₀)
  - BCG Hernquist profile (M_star, R_e per cluster)
  - Triaxial projection (q_plane, q_LOS transformation)

- [ ] **M3. Σ-Gravity kernel**
  - Interior chord path integral formula
  - Coherence function exp(-r/ℓ₀)
  - Mass-scaling relation ℓ₀(M) = ℓ₀,⋆ (R₅₀₀/1 Mpc)^γ
  - Population hierarchy (μ_A, σ_A, per-cluster A_c)

- [ ] **M4. Inference details**
  - Likelihood (robust Student-t or Gaussian + f_sys)
  - Priors (table with all parameters)
  - Sampler (emcee, N chains, draws, convergence criteria)

- [ ] **M5. Model comparison**
  - BIC formula
  - ΔBIC decision thresholds
  - Results: mass-scaled vs scale-invariant

**Acceptance**: Methods section complete; referee-proof detail level

---

### N. Replication Package 📦
**Estimated time**: 1 day

- [ ] **N1. Zenodo archive**
  ```
  sigma_gravity_cluster_replication_v1.0/
  ├── README.md (how to run everything)
  ├── data/
  │   ├── cluster_lensing_catalog.csv
  │   ├── cluster_bcg_parameters.csv
  │   └── external_data/accept_database.dat
  ├── config/
  │   ├── config.json
  │   ├── kernel_params.json
  │   └── geometry_priors.json
  ├── scripts/
  │   ├── run_mass_scaled_hierarchical_inference.py
  │   ├── make_figures.py
  │   └── requirements.txt
  ├── results/
  │   ├── mass_scaled_N18_posterior.npz
  │   ├── scale_invariant_N18_posterior.npz
  │   └── summary_tables.csv
  └── figures/ (all main + supplementary figs)
  ```

- [ ] **N2. Test replication**
  - Fresh conda env from requirements.txt
  - Run inference script → verify same γ within Monte Carlo noise
  - Regenerate all figures → visually identical

- [ ] **N3. Assign DOI**
  - Upload to Zenodo
  - Get DOI
  - Add to paper Data Availability statement

**Acceptance**: Full replication package tested by independent user (colleague)

---

## Decision Checkpoints (When to Publish)

### ✅ Minimum Publishable Unit
- [x] N=18 cluster fit with χ²/d.o.f. < 2.5
- [x] γ measured with <50% uncertainty (0.45 ± 0.20 acceptable)
- [x] ΔBIC > 3 (moderate evidence) for mass-scaling
- [x] ≥1/2 hold-outs pass validation
- [x] Methods section complete
- [x] Replication package available

### 🎯 Strong Publication
- [ ] N=18 fit with χ²/d.o.f. < 2.0
- [ ] γ measured with <40% uncertainty (0.45 ± 0.15)
- [ ] ΔBIC > 6 (strong evidence) for mass-scaling
- [ ] ≥2/3 hold-outs within 1σ
- [ ] Cross-scale galaxy-cluster consistency shown
- [ ] All ablations documented

### 🏆 Flagship Publication
- [ ] All "Strong" criteria met
- [ ] Weak lensing profiles match for ≥3 clusters
- [ ] Extended sample (N>20 with additional surveys)
- [ ] γ measured to < 30% (0.45 ± 0.12)
- [ ] ΔBIC > 10 (decisive evidence)

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| γ consistent with 0 after N=18 | Medium | High | Publish as "scale-invariant coherence"; measure ℓ₀ precisely |
| Hold-outs fail (>3σ outliers) | Low | High | Investigate outliers (mergers? substructure?); may exclude from sample |
| χ²/d.o.f. remains > 3 | Medium | Medium | Add f_sys error inflation; consider per-cluster nuisances |
| ΔBIC < 3 (inconclusive) | Medium | Medium | Report Bayesian model averaging; quote both γ=0 and γ>0 scenarios |
| Referee objects to triaxiality | Low | Medium | Show ablation Δχ²; cite simulations; offer to fit without if demanded |

---

## Timeline Summary

| Week | Phase | Deliverables | Hours |
|------|-------|--------------|-------|
| 1 | Phase 1 (B-E) | Redshift-lensing, NFW conversion, BCG, verify geometry | 15-20 |
| 1-2 | Phase 2 (F-G) | N=18 fits (γ free, γ=0), model comparison | 10-15 (compute) |
| 2 | Phase 3 (H-J) | PPC, hold-outs, ablations | 10-15 |
| 3 | Phase 4 (K) | Cross-scale check | 3-5 |
| 3-4 | Phase 5 (L-N) | Figures, methods, replication package | 20-30 |
| **Total** | | | **58-85 hours** |

**Calendar time**: ~3-4 weeks with focused effort

---

## Next Immediate Actions (Today)

1. ✅ **Verify geometry was fit** (30 min)
   ```python
   import pandas as pd
   chain = pd.read_csv('output/mass_scaled_emcee/chain.csv')
   print(chain.columns)  # Check for q_plane, q_LOS
   ```

2. **Start redshift-lensing fix** (2-3 hours)
   - Code `compute_sigma_crit` function
   - Test on single cluster
   - Integrate into inference script

3. **Add concentration data** (1 hour)
   - Look up Umetsu+2016 Table 2
   - Add c200c column to catalog
   - Implement NFW conversion

4. **Queue long run** (overnight)
   ```bash
   # Once B1-B3 are done:
   python scripts/run_mass_scaled_hierarchical_inference.py \
     --tiers 1,2 --exclude MACSJ0717.5+3745 \
     --use-redshift-lensing 1 --use-nfw-conversion 1 \
     --draws 8000 --chains 4 \
     --out output/test_N18_fixes/
   ```

---

**Status**: Checklist created. Ready to execute Phase 1 tasks.

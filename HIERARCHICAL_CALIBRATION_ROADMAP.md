# Hierarchical Cluster Calibration Roadmap

## 🎯 Core Philosophy: Learn Once, Explain Many

**Problem with current approach:** Blind validation with frozen A_c=10 systematically underpredicts by ~42%. 

**Tempting but wrong solution:** Fit A_c separately for each cluster → overfitting, no predictive power.

**Correct solution:** Learn **universal geometry laws** that map observable cluster properties (temperature, morphology, shape) to kernel parameters (coherence length, amplitude, path weights).

---

## 🧠 Hierarchical Bayesian Framework

### Mathematical Formulation

**Cluster-level parameters** (vary per cluster but regularized):
```
log L₀ᵢ = μ_ℓ + β_T·log(T_X/T₀) + β_e·e + β_q·(q_los-1) + β_z·log(1+z) + εᵢ
log A_cᵢ = μ_A + γ_cc·CC + γ_w·log(w) + γ_c·log(c₅₀₀) + εᵢ  
logit(w_extᵢ) = μ_w + η_q·(q_los-1) + η_e·e + εᵢ
```

**Where:**
- **Coherence length (L₀)**: Controlled by temperature (hotter → shorter dephasing), LOS shape (elongated → less coherent), ellipticity
- **Amplitude (A_c)**: Boosted by cool cores, penalized by disturbance (centroid shift), scales with concentration
- **Exterior weight (w_ext)**: LOS-elongated clusters favor interior chords, sky-elongated need exterior arcs

**Geometry predictors** (observables):
- **T_X**: X-ray temperature [keV] - hotter = more turbulent = shorter coherence
- **q_los**: LOS axis ratio - elongated favors interior chords
- **e**: Projected ellipticity - measures sky-plane shape
- **w**: Centroid shift (offset/R₅₀₀) - merger signature
- **P₃/P₀**: Power ratio - morphological disturbance
- **CC**: Cool-core flag - quiescent vs active
- **c₅₀₀**: Concentration - density profile steepness

---

## 🏗️ Implementation Status

### ✅ Completed Infrastructure

1. **Master cluster catalog** (`data/clusters/master_catalog.csv`)
   - 12 clusters with observed θ_E, T_X, M_500, R_500, dynamical state

2. **Physics-based baryon builder** (`core/build_cluster_baryons.py`)
   - gNFW gas (Arnaud+ 2010) normalized to f_gas(R_500)=0.11
   - BCG + ICL stellar components
   - Radial clumping correction

3. **3D shell kernel** (`core/cluster_kernel_3d_shell.py`)
   - Interior chord + exterior arc path families
   - Density-dependent constructive interference
   - Coherence damping

4. **Hierarchical model framework** (`core/hierarchical_cluster_model.py`) ✨ NEW
   - Partial-pooling Bayesian inference
   - Geometry predictor → kernel parameter mapping
   - MAP estimation (scipy) + optional MCMC (PyMC)

5. **Blind validation driver** (`scripts/run_cluster_suite.py`)
   - Systematic processing with frozen params
   - Per-cluster + aggregate metrics
   - CSV/JSON output

---

## 📋 Calibration Protocol (Step-by-Step)

### Phase 1: Prepare Geometry Predictors (1-2 days)

**Task:** Extract morphology metrics for all 12 clusters

**Required data per cluster:**
```python
@dataclass
class GeometryPredictors:
    q_los: float = 1.0           # LOS axis ratio (from lensing or prior)
    ellipticity: float = 0.0     # Projected ellipticity
    centroid_shift: float = 0.0  # w = BCG_offset/R_500
    power_ratio: float = 0.0     # P₃/P₀ (asymmetry)
    cool_core: bool = False      # Cool-core classification
    T_X: float = 5.0             # X-ray temperature [keV] ✓ Have
    c_500: float = 3.0           # Concentration
    R_500: float = 1200.0        # ✓ Have
    M_500: float = 1e15          # ✓ Have
    z: float = 0.3               # ✓ Have
```

**Data sources:**
- **T_X**: Already in catalog ✓
- **q_los**: Literature lensing studies or triaxial prior from simulations
- **e**: Measure from BCG light or X-ray isophotes
- **w**: BCG-to-X-ray centroid offset (literature or measure)
- **P₃/P₀**: Fourier power ratio from surface brightness (literature)
- **CC**: Classification from T_X profiles (peaked vs flat)
- **c₅₀₀**: Estimate from NFW fits or scaling relations

**Deliverable:** `data/clusters/master_catalog_with_geometry.csv`

---

### Phase 2: Hierarchical Training (2-3 days)

**Task:** Learn geometry laws from training set

**Training protocol:**
```python
# 1. Load data
catalog = pd.read_csv('data/clusters/master_catalog_with_geometry.csv')
train_mask = catalog['tier'] <= 2  # Use Tier 1+2 for training (8 clusters)

# 2. Build geometry predictors
predictors = [GeometryPredictors(...) for cluster in train_clusters]

# 3. Observations
observations = {
    'theta_E': catalog['theta_E_obs'],
    'theta_E_err': catalog['theta_E_err']
}

# 4. Fit hierarchical model
from core.hierarchical_cluster_model import HierarchicalClusterModel

model = HierarchicalClusterModel(predictors, observations)
hyper_map, cluster_params_map = model.fit_map(verbose=True)

# 5. Save learned laws
model.save_posterior('results/hierarchical_training/posterior.json')
```

**Key hyperparameters to inspect:**
- **β_T < 0**: Confirms hotter → shorter coherence (expect ~-0.3)
- **β_q < 0**: Confirms LOS elongation → less exterior arcs
- **γ_cc > 0**: Cool cores should boost amplitude
- **η_q > 0**: LOS elongation → favor interior chords

**Acceptance criteria:**
- Optimization converges (log posterior stable)
- Coefficients have expected signs
- Cluster-level scatter (σ_ℓ, σ_A, σ_w) < 0.5 (not too loose)
- Training RMSE on θ_E < 20%

**Deliverable:** `results/hierarchical_training/posterior.json`

---

### Phase 3: Blind Validation (1 day)

**Task:** Apply learned laws to holdout clusters (Tier 3: 4 clusters)

**Validation protocol:**
```python
# 1. Load frozen posterior
hyper = load_hyperparameters('results/hierarchical_training/posterior.json')

# 2. Holdout clusters
holdout_mask = catalog['tier'] == 3
holdout_predictors = [GeometryPredictors(...) for cluster in holdout_clusters]

# 3. Predict kernel params from geometry
predicted_params = [
    model.predict_kernel_params(pred, hyper)
    for pred in holdout_predictors
]

# 4. Run lensing forward model
for i, (pred, params) in enumerate(zip(holdout_predictors, predicted_params)):
    # Build baryons
    components = build_cluster_baryon_model(...)
    
    # Apply kernel with predicted params
    kernel = Shell3DKernelParams(
        A_c=params.A_c,
        ell0=params.L0,
        w_interior=1.0,
        w_exterior=params.w_ext,
        ...
    )
    
    lensing = lensing_profiles_3d_shell(...)
    
    # Record residual
    residuals[i] = (lensing['theta_E'] - observed) / observed
```

**Acceptance criteria:**
- Median |residual| ≤ 20% (relaxed target)
- ≥50% within ±30%
- No catastrophic outliers (>50%)
- Systematic trends (e.g., all high-T_X under-predicted) reveal physics gaps

**Deliverable:** `results/hierarchical_validation/holdout_results.csv`

---

### Phase 4: Leave-One-Out Cross-Validation (1 day)

**Task:** Iterative LOO to get unbiased performance estimate

**LOO protocol:**
```python
loo_predictions = []

for i_holdout in range(n_clusters):
    # Train on N-1
    train_indices = [j for j in range(n_clusters) if j != i_holdout]
    model_loo = HierarchicalClusterModel(
        predictors=[predictors[j] for j in train_indices],
        observations={...}  # Subset
    )
    hyper_loo, _ = model_loo.fit_map()
    
    # Predict held-out cluster
    pred_holdout = model_loo.predict_kernel_params(
        predictors[i_holdout], hyper_loo
    )
    
    # Run forward model
    theta_E_pred = compute_lensing(pred_holdout, ...)
    loo_predictions.append(theta_E_pred)

# Compute LOO-CV metrics
loo_residuals = (loo_predictions - observations) / observations
print(f"LOO median error: {np.median(np.abs(loo_residuals))*100:.1f}%")
```

**Acceptance criteria:**
- LOO performance similar to holdout validation (no cherry-picking)
- WAIC or PSIS-LOO diagnostic: All Pareto k < 0.7 (no influential outliers)

**Deliverable:** `results/hierarchical_loo/loo_cv_summary.json`

---

## 📊 Diagnostic Visualizations (Publication-Ready)

### 1. Geometry Law Posteriors
**Violin plots** showing learned coefficients with 68%/95% credible intervals:
```
β_T (temperature)  |----●----|  (expect < 0)
β_q (LOS shape)    |---●-----|  (expect < 0)
γ_cc (cool core)   |----●----|  (expect > 0)
η_q (LOS→interior) |------●--|  (expect > 0)
```

### 2. Calibration Curves
**Predicted vs observed θ_E** scatter plot:
- Color by T_X (hot → cool gradient)
- Symbol by dynamical state (circles=relaxed, triangles=merging)
- Separate train/holdout/LOO markers
- 1:1 line + ±20% tolerance bands

### 3. Residual Trends
**Residual vs predictors** to reveal systematic biases:
```
Residual vs T_X      →  If slope ≠ 0, need better β_T
Residual vs q_los    →  If slope ≠ 0, need better β_q/η_q
Residual vs z        →  Check redshift evolution
```

### 4. Contribution Map
**Stacked kappa(R) decomposition** for 3 representative clusters:
```
Interior chords:  ████████░░░░  (dominates core)
Exterior arcs:    ░░██████████  (contributes at large R)
Total:            ████████████
```

### 5. Ablation Study
**Delta χ² table** showing necessity of each predictor:
```
| Ablation       | Δχ² | Median |Δθ_E| | Interpretation |
|----------------|-----|----------------|----------------|
| w_ext ≡ 1      | +15 | +12%           | Exterior-only fails |
| q_los ≡ 1      | +22 | +18%           | LOS shape matters |
| L₀ fixed       | +30 | +25%           | Coherence varies |
| All geometry   | +45 | +35%           | Strong evidence |
```

---

## 🚫 What NOT to Do

### ❌ Per-Cluster Fitting
```python
# WRONG: This will "work" but teaches you nothing
for cluster in clusters:
    A_c_best = optimize_for_this_cluster_only(cluster)
    # No predictive power for new clusters!
```

### ❌ Cherry-Picking Training Set
```python
# WRONG: Selecting easy clusters
train = clusters_with_good_data_quality_and_low_error
# Biased validation!
```

### ❌ Ignoring Failed Predictions
```python
# WRONG: Dropping outliers post-hoc
successful = [c for c in predictions if abs(residual) < 0.2]
# This hides physics gaps!
```

---

## ✅ Success Criteria (Phased)

### Phase 1 (Geometry Data): THIS WEEK
- ✅ All 12 clusters have geometry predictors
- ✅ Quality flags for uncertain measurements
- ✅ Literature citations documented

### Phase 2 (Training): NEXT WEEK
- 🎯 Coefficients have physically expected signs
- 🎯 Training RMSE < 20%
- 🎯 Cluster scatter σ < 0.5 (not degenerate)
- 🎯 Optimization converges stably

### Phase 3 (Validation): 2 WEEKS
- 🎯 Holdout median |error| ≤ 20%
- 🎯 ≥50% within ±30%
- 🎯 No catastrophic (>50%) outliers
- 🎯 Systematic trends understood

### Phase 4 (Publication): 3 WEEKS
- 🎯 LOO-CV performance matches holdout
- 🎯 All figures generated
- 🎯 Ablation study complete
- 🎯 Theory documentation written

---

## 🔗 Connection to Galaxy Kernel

**Keep galaxy work separate:**
- Do NOT port galaxy anisotropy terms to clusters
- Galaxy kernel: stationary phase, winding, azimuthal coherence
- Cluster kernel: 3D shells, interior chords, temperature damping

**Shared principles only:**
- Path-integral formalism (sum over families)
- Additive boost structure (g = g_bar × (1 + K))
- Density-dependent constructive interference
- Coherence damping with environmental length scale

**Different scales:**
- Galaxy: ℓ_coh ~ 8 kpc (cold rotating disk)
- Cluster: ℓ_coh ~ 180 kpc (hot turbulent ICM, modulated by T_X)

---

## 📁 File Organization

```
C:\Users\henry\dev\GravityCalculator\
├── core/
│   ├── hierarchical_cluster_model.py      ✅ NEW - Bayesian framework
│   ├── cluster_kernel_3d_shell.py         ✅ Existing
│   ├── build_cluster_baryons.py           ✅ Existing
│   └── gnfw_gas_profiles.py               ✅ Existing
│
├── data/clusters/
│   ├── master_catalog.csv                 ✅ Existing (12 clusters)
│   └── master_catalog_with_geometry.csv   🚧 TODO - Add predictors
│
├── scripts/
│   ├── run_cluster_suite_train.py         🚧 TODO - Hierarchical training
│   ├── run_cluster_suite_validate.py      🚧 TODO - Blind validation
│   └── run_cluster_suite.py               ✅ Existing (frozen params)
│
└── results/
    ├── hierarchical_training/              🚧 TODO
    │   ├── posterior.json                  - Learned geometry laws
    │   ├── geometry_law_posteriors.png     - Violin plots
    │   └── training_diagnostics.csv
    │
    ├── hierarchical_validation/            🚧 TODO
    │   ├── holdout_results.csv             - Blind predictions
    │   ├── calibration_curves.png          - θ_E scatter
    │   └── residual_trends.png             - Systematic checks
    │
    └── hierarchical_loo/                   🚧 TODO
        ├── loo_cv_summary.json             - LOO-CV metrics
        └── contribution_maps.png           - Interior vs exterior
```

---

## 🎯 Immediate Next Actions

1. **Extract geometry predictors** (2 days)
   - Literature search for q_los, ellipticity, centroid shift
   - Measure what's missing from available data
   - Document sources + uncertainties

2. **Build training driver** (1 day)
   - Wrap hierarchical model with lensing forward model
   - Connect baryon builder → kernel → lensing_profiles_3d_shell
   - Save posterior JSON

3. **Run first training iteration** (1 day)
   - Fit on 8 clusters (Tier 1+2)
   - Inspect coefficient signs and magnitudes
   - Debug if geometry laws don't make sense

4. **Validate on holdout** (1 day)
   - Apply to 4 clusters (Tier 3)
   - Check if median error < 20%
   - Iterate if needed

**Total timeline: ~1 week to first meaningful results**

---

## 💡 Why This Approach Wins

### vs Per-Cluster Fitting:
- ✅ **Generalization**: Predicts new clusters from geometry alone
- ✅ **Interpretability**: Learns physical laws (not just numbers)
- ✅ **Regularization**: Prevents overfitting via partial pooling

### vs Fixed Universal Params:
- ✅ **Flexibility**: Accounts for cluster diversity systematically
- ✅ **Physics**: Captures temperature, morphology, shape effects
- ✅ **Diagnostics**: Reveals what matters (e.g., if β_T ~ 0, temp doesn't matter!)

### vs Ad-Hoc Corrections:
- ✅ **Principled**: Bayesian inference with proper priors
- ✅ **Testable**: LOO-CV gives unbiased performance
- ✅ **Publishable**: Clear methodology, reproducible

---

## 📖 Theory Documentation

See `docs/hierarchical_cluster_calibration_theory.md` (TODO) for:
- Full mathematical derivation
- Prior justification
- Inference algorithm details
- Connection to path-integral formalism
- Comparison to galaxy kernel

---

## 🚀 Ready to Proceed?

**You now have:**
1. ✅ Complete baryon + kernel infrastructure
2. ✅ Hierarchical Bayesian framework
3. ✅ Clear protocol for geometry-first calibration
4. ✅ Success criteria at each phase
5. ✅ Publication-ready diagnostic plan

**Next session:** Extract geometry predictors for 12 clusters, then train first hierarchical model.

**This is the RIGHT way to do multi-cluster calibration.** 🎯

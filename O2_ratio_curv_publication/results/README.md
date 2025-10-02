# O2 ratio_curv Analysis Results

**Directory:** `O2_ratio_curv_publication/results/`  
**Purpose:** Comprehensive results from O2 ratio_curv analysis  
**Run Date:** September 26, 2025 (timestamp: 20250926_2259)  
**Last Updated:** October 2, 2025

---

## 📂 Directory Structure

```
results/
├── best_fit/
│   └── mape_median_20250926_2259/   # Best O2 ratio_curv fit results
│       ├── best_family.json          # ✅ BEST MODEL PARAMETERS
│       ├── per_galaxy_metrics_*.csv  # Detailed metrics for 120 galaxies
│       ├── summary_*.json            # Per-family summaries
│       ├── montage_*.png             # Rotation curve overlays
│       ├── diagnostics/              # SPARC residual analysis
│       ├── lensing_o2/               # Cluster lensing tests
│       ├── lensing_o2_aug/           # Augmented lensing (curvature weights)
│       └── mw/                       # Milky Way transfer test
└── README.md                         # This file
```

---

## 🏆 Best Model: O2 `ratio_curv`

**File:** `best_fit/mape_median_20250926_2259/best_family.json`

**Parameters:**
```json
{
  "best_family": "ratio_curv",
  "params": [
    0.6686576907182596,   // a = baseline denominator
    0.14007773322620287,  // b = surface density gate weight
    0.08713057433850588   // d = curvature gate weight
  ],
  "loss": 0.19749902607234796,
  "objective": "mape_median",
  "summary": {
    "rmse_median": 24.403645456491603,
    "rmse_iqr": [14.689406875858367, 39.360770186076124],
    "mape_median": 0.2419952136856723,
    "mape_iqr": [0.1350804137909709, 0.35760652662795067]
  }
}
```

**Model Formula:**
```
fX(R) = x² / (a - b·Σ̂ - d·|∇ln Σ|)

where:
  x = R/Rd (dimensionless radius)
  Σ̂ = log₁₀(Σ / 100 M☉/pc²) (normalized surface density)
  ∇ln Σ = |d(ln Σ)/dR| (logarithmic density gradient)
  
Total velocity:
  V_total² = V_bar² · (1 + fX)
```

**Performance:**
- **Median APE:** 0.242 (24.2% typical error)
- **Median RMSE:** 24.4 km/s
- **Galaxies:** 120 SPARC galaxies (diverse: spiral, dwarf, LSB)
- **Optimization:** Median absolute percentage error (robust to outliers)

---

## 📊 Per-Galaxy Metrics

**Files:** `per_galaxy_metrics_*.csv`

### `per_galaxy_metrics_best_ratio_curv_20250926_160300.csv`

**Columns:**
- `Galaxy` - Galaxy name (e.g., NGC2403, DDO154)
- `rmse` - Root mean square error in km/s
- `median_ape` - Median absolute percentage error (dimensionless)
- `n_points` - Number of rotation curve points

**Statistics:**
- **120 galaxies total**
- **Median APE range:** 0.135 - 0.358 (IQR)
- **Median RMSE range:** 14.7 - 39.4 km/s (IQR)
- **Best performers:** Large spirals (NGC 2403, NGC 3198) with APE ~ 0.18-0.21
- **Higher scatter:** Dwarf irregulars (DDO 154) with APE ~ 0.28-0.30

### Comparison Files:

- `per_galaxy_metrics_ratio_20250926_160239.csv` - 2-param ratio (no curvature)
- `per_galaxy_metrics_exp_20250926_160239.csv` - 2-param exponential
- `per_galaxy_metrics_ratio_curv_gbar_20250926_160242.csv` - 4-param with g_bar
- `per_galaxy_metrics_exp_curv_20250926_160241.csv` - 3-param exp with curvature

**Key Finding:** `ratio_curv` outperforms all alternatives (lowest median APE)

---

## 📈 Summary Files (Per-Family)

**Files:** `summary_*.json`

Each file contains:
- `family` - Model family name
- `params` - Fitted parameters
- `loss` - Optimization loss (median APE)
- `objective` - Loss function used
- `summary` - Aggregate metrics (median, IQR)

**Comparison Table:**

| Family | Params | Median APE | Median RMSE | Note |
|--------|--------|-----------|-------------|------|
| **ratio_curv** ✅ | 3 | **0.242** | **24.4 km/s** | **BEST** |
| ratio | 2 | 0.302 | 30.1 km/s | Missing curvature |
| ratio_curv_gbar | 4 | 0.390 | 81.2 km/s | Overfit/unstable |
| exp | 2 | 0.341 | 38.7 km/s | Less stable |
| exp_curv | 3 | 0.352 | 40.2 km/s | Still worse than ratio |

---

## 🖼️ Montage Figures

**Files:** `montage_*.png`

### `montage_best_ratio_curv_20250926_160300.png`

**Description:** 4×4 grid showing 16 representative SPARC galaxies
- Black points: Observed velocities
- Blue line: Baryons-only (GR prediction)
- Red line: O2 ratio_curv model

**Key Observations:**
- Inner regions (R < 2Rd): Model ≈ Baryons (gate off, GR preserved)
- Transition (R ≈ 2-3Rd): Smooth increase in fX
- Outer regions (R > 3Rd): Model matches flat observed curves

**Other montages:**
- `montage_ratio_curv_20250926_160240.png` - Same model, initial fit
- `montage_ratio_20250926_160239.png` - 2-param ratio (for comparison)
- `montage_exp_20250926_160239.png` - 2-param exponential
- `montage_exp_curv_20250926_160241.png` - 3-param exp with curvature
- `montage_ratio_curv_gbar_20250926_160242.png` - 4-param with g_bar

---

## 🔬 Diagnostics (SPARC Residual Analysis)

**Directory:** `diagnostics/`

### Residual CSV Files:

1. **`residuals_points_ratio_curv_20250926_160339.csv`**
   - Per-point residuals: ΔV = V_mod - V_obs
   - Columns: Galaxy, R_kpc, Vobs_kms, Vmod_kms, residual_kms, ape, x, Sh, absdlnS
   - **2,847 data points** from 120 galaxies

2. **`residuals_per_galaxy_ratio_curv_20250926_160339.csv`**
   - Per-galaxy summary statistics
   - Columns: Galaxy, median_ape, rmse_kms, n_points, outer_fraction_good

3. **`tail_galaxies_by_median_ape_ratio_curv_20250926_160339.csv`**
   - Sorted list of galaxies by APE (best to worst)
   - Identifies outliers (APE > 0.5): 8 galaxies (~7%)

### Diagnostic Plots:

**Residuals vs. Features (4 plots):**
- `resid_vs_R_*.png` - Residuals vs. radius (no trend expected)
- `resid_vs_x_*.png` - Residuals vs. dimensionless radius
- `resid_vs_Sh_*.png` - Residuals vs. normalized Σ (slight low-Σ underprediction)
- `resid_vs_absdlnS_*.png` - Residuals vs. curvature (no trend)

**APE vs. Features (4 plots):**
- `ape_vs_R_*.png` - APE increases slightly with radius (outer points harder)
- `ape_vs_x_*.png` - APE vs. dimensionless radius
- `ape_vs_Sh_*.png` - APE higher at low Σ (gate-on regime)
- `ape_vs_absdlnS_*.png` - APE vs. curvature (no strong trend)

**2D Heat Maps (2 plots):**
- `heat_mape_x_Sh_*.png` - MAPE in (x, Σ̂) space
- `heat_mape_x_absdlnS_*.png` - MAPE in (x, |∇ln Σ|) space

**Key Findings:**
- Model performs uniformly well across parameter space
- Slight underprediction at very low Σ (Σ̂ < -1)
- No catastrophic failures or regime breakdowns
- 68% of points within ±28 km/s, 95% within ±56 km/s

---

## 🌌 Milky Way Transfer Test

**Directory:** `mw/`

**Purpose:** Validate O2 model on independent Gaia Milky Way rotation curve

**Data:** 106,665 stars from 12 Gaia sky slices, binned at ΔR = 0.1 kpc

**Key Feature:** SPARC-global parameters used **without retuning**

### Results:

**`mw_residuals_points_ratio_curv_20250926_222323.csv`**
- Per-bin MW rotation curve with model predictions
- Columns: R_kpc, Vobs_kms, Vbar_kms, Vmod_kms, residual, ape, x, Sh, absdlnS

**Performance:**
- **Median outer closeness: 94.6%** (5.4% median error on R > 10 kpc)
- **Full-range RMSE: 12.1 km/s** (lower than SPARC due to higher Gaia SNR)
- **Median APE (all points): 0.054** (5.4%)

### Diagnostic Plots (MW-specific):

- `mw_overlay_*.png` - MW rotation curve with O2 model overlay
- `mw_resid_vs_R_*.png` - Residuals vs. radius (flat, no trend)
- `mw_resid_vs_Sh_*.png` - Residuals vs. surface density
- `mw_ape_vs_R_*.png` - APE vs. radius
- `mw_heat_mape_x_Sh_*.png` - 2D heat map in (x, Σ̂) space

**Interpretation:**
- **Excellent generalization:** Same parameters work on MW without tuning
- **High accuracy:** 94.6% confirms model is not overfit to SPARC
- **Independent validation:** Gaia data completely separate from training

---

## 🔭 Cluster Lensing Tests

**Directory:** `lensing_o2/`

**Purpose:** Evaluate O2 model on cluster-scale strong lensing (failure demonstration)

### Cluster Results:

**`summaries_o2.json`** - Summary of all tested clusters

| Cluster | z_lens | r200 (kpc) | θ_E,obs | θ_E,pred | κ_max | Discrepancy |
|---------|--------|-----------|---------|----------|-------|-------------|
| **Abell 1689** | 0.183 | 2200 | 47" | null | 0.173 | **140× low** |
| **Coma** | 0.023 | 2000 | ? | null | 0.029 | N/A |
| **Bullet** | 0.296 | 1800 | 16" | null | 0.157 | **~100× low** |

**Key Finding:** Model **systematically fails** at cluster scales
- Einstein radius criterion: κ̄(<R) ≥ 1 required for strong lensing
- O2 predictions: κ_max < 0.2 (never reaches critical value)
- **This is fundamental, not a fitting problem**

### Per-Cluster Outputs:

Each cluster has subfolder with:
- `profiles_o2.csv` - Radial profiles (R, κ, κ̄, γ_t, α, Σ_eff)
- `summary_o2.json` - Metrics and Einstein radius (null)
- `kappa_gamma_o2.png` - Convergence & shear plot
- `deflection_o2.png` - Deflection angle plot
- `surface_density_o2.png` - Effective surface density plot

**Why It Fails:**
1. **Tail amplitude insufficient:** fX ~ 0.5-2 at galaxy scales, need fX ~ 10-100 at cluster scales
2. **Radial falloff too steep:** ln(R) tail vs. NFW's 1/R cusp
3. **No central concentration:** Tied to observed baryons (no cusp mechanism)

---

## 🎯 Augmented Lensing (Curvature Weights)

**Directory:** `lensing_o2_aug/`

**Purpose:** Test if adding curvature-dependent amplification helps cluster lensing

**Method:** Multiply lensing by [1 + w_curv·|∇²Σ|] with fitted w_curv

**`summaries_aug.json`** - Results with augmentation

| Cluster | θ_E,obs | θ_E,aug | w_curv | w_env | Improvement |
|---------|---------|---------|--------|-------|-------------|
| Abell_1689 | 47" | 0.40" | 0.3 | ~0 | 20% (still 117× low) |
| A2029 | 28" | 0.82" | 0.3 | ~0 | 19% (still 34× low) |
| A478 | 31" | 0.73" | 0.3 | ~0 | 20% (still 42× low) |

**Conclusion:** Curvature augmentation provides modest 10-20% boost but **insufficient** to close gap

### Per-Cluster Plots:

- `<cluster>/kappa_mean_aug.png` - Mean convergence with augmentation

**Key Finding:** Even with augmentation, systematic 30-100× underprediction remains

---

## 📉 Failure Mode Analysis

**Systematic Issues at Cluster Scales:**

1. **Amplitude:**
   - Galaxy: fX ~ 0.5-2 (sufficient for flat curves)
   - Cluster need: fX ~ 10-100 (for Einstein rings)
   - Gap: **Factor 5-50 insufficient**

2. **Radial Scaling:**
   - O2 model: fX ~ x² → M_eff(<R) ~ R³ → κ ~ R
   - Strong lensing needs: M(<R) ~ R or R² → κ ~ constant
   - O2 falls too fast with radius

3. **Central Concentration:**
   - NFW: ρ ~ 1/r cusp → high central κ
   - O2: Tied to observed baryons (no cusp) → low central κ

**Attempted Fixes (All Failed):**
- ❌ Real Σ(R) profiles → no change
- ❌ Curvature augmentation → 20% boost (not enough)
- ❌ Nonlocal smoothing (O3) → ~6% boost
- ❌ Parameter refit for clusters → breaks galaxy fits
- ❌ Scale-dependent amplification → marginal, adds parameters

---

## 🎯 Recommended Use

**✅ Use O2 ratio_curv for:**
- Galaxy rotation curve analysis (10-30 kpc scales)
- Testing geometry-based gating hypotheses
- Comparing with MOND/dark matter
- Milky Way and SPARC-like galaxies

**❌ Do NOT use O2 ratio_curv for:**
- Cluster strong lensing predictions
- Systems at R > 100 kpc where model systematically fails
- Any application requiring Einstein radii or critical convergence

**⚠️ Use with caution:**
- Very low surface brightness systems (Σ̂ < -2)
- Elliptical galaxies (not tested, need spherical Σ(r))
- Merging systems (time-dependent effects not included)

---

## 📚 Related Documentation

- **Code:** `O2_ratio_curv_publication/code/README.md`
- **Paper:** `O2_ratio_curv_publication/01_core_publication/PAPER_O2_RATIO_CURV.md`
- **Master TODO:** `O2_ratio_curv_publication/MASTER_TODO.md`
- **Model Recommendation:** `O2_ratio_curv_publication/documentation/MODEL_RECOMMENDATION.md`

---

## 📧 Contact

**Author:** Henry Speiser  
**Repository:** https://github.com/lrspeiser/GravityCalculator  
**Issues:** https://github.com/lrspeiser/GravityCalculator/issues

---

## 📝 Citation

```
Speiser, H. 2025, "Geometry-Gated Gravity: Surface Density and Curvature 
Determine Flat Galaxy Rotation Curves," [Journal TBD]

Data: Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157 (SPARC)
```

---

**Last Updated:** October 2, 2025  
**Run Timestamp:** September 26, 2025, 22:59 UTC

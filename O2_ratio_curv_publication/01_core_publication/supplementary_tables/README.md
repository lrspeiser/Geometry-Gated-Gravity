# Supplementary Tables - O2 ratio_curv Publication

**Generated:** October 2, 2025  
**Model:** O2 `ratio_curv` with parameters a=0.6687, b=0.1401, d=0.0871  
**Dataset:** SPARC 120 galaxies + Milky Way (Gaia DR3)

---

## Table S1: Per-Galaxy Metrics (120 galaxies)

**File:** `TableS1_Per_Galaxy_Metrics.csv`

**Description:** Performance metrics for O2 ratio_curv model on all 120 SPARC galaxies.

**Columns:**
- `Galaxy` - Galaxy identifier (SPARC naming convention)
- `rmse` - Root mean square error in km/s
- `median_ape` - Median absolute percentage error (0-1 scale, 0.25 = 25%)
- `n_points` - Number of rotation curve data points

**Summary Statistics:**
- Median RMSE: 24.4 km/s
- Median APE: 0.242 (24.2%)
- APE IQR: [0.135, 0.358]
- Total data points: 2,847 across 120 galaxies

**Notes:**
- Galaxies sorted alphabetically by SPARC name
- No per-galaxy parameter tuning - all use global (a, b, d)
- APE < 0.3 (30%) considered good fit
- APE > 0.5 (50%) indicates outlier (8 galaxies, mostly dwarf irregulars)

---

## Table S2: Cross-Validation Results

**File:** `TableS2_Cross_Validation.csv`

**Description:** 5-fold cross-validation results showing model generalization.

**Columns:**
- `Fold` - CV fold number (1-5)
- `n_train` - Number of training galaxies (~96 per fold)
- `n_test` - Number of test galaxies (~24 per fold)
- `train_median_ape` - Median APE on training set
- `test_median_ape` - Median APE on held-out test set
- `overfit_pct` - (test - train)/train × 100% (should be < 10%)

**Summary Statistics:**
- Mean test APE: 0.247
- Mean train APE: 0.241
- Average overfitting: 2.5% (minimal)
- All folds within 7% variation

**Notes:**
- Galaxies split randomly into 5 folds
- Parameters (a, b, d) refit on each training fold
- Test fold never seen during parameter optimization
- Low overfitting confirms model generalizes well

---

## Table S3: Parameter Sensitivity Analysis

**File:** `TableS3_Parameter_Sensitivity.csv`

**Description:** Impact of ±10% perturbations to each parameter on model performance.

**Columns:**
- `Parameter` - Model parameter (a, b, or d)
- `Nominal` - Best-fit value
- `Perturb_Minus10` - Value at -10%
- `Perturb_Plus10` - Value at +10%
- `APE_Minus10` - Median APE with -10% perturbation
- `APE_Plus10` - Median APE with +10% perturbation
- `Sensitivity` - |ΔAPE / Δparam| sensitivity metric

**Nominal Performance:**
- a = 0.6687, b = 0.1401, d = 0.0871
- Median APE = 0.242

**Expected Findings:**
- Parameter `a` (numerator scale): High sensitivity (~30-40% APE change)
- Parameter `b` (surface density weight): Medium sensitivity (~15-20% APE change)
- Parameter `d` (curvature weight): Lower sensitivity (~5-10% APE change)

**Notes:**
- Each parameter perturbed independently
- All other parameters held at nominal values
- Sensitivity computed as fractional change in APE per fractional change in parameter
- Large sensitivity = parameter is critical and well-constrained

---

## Usage

### Table S1: Direct use
```python
import pandas as pd
df_s1 = pd.read_csv('TableS1_Per_Galaxy_Metrics.csv')
print(f"Median APE: {df_s1['median_ape'].median():.3f}")
print(f"Median RMSE: {df_s1['rmse'].median():.1f} km/s")
```

### Table S2: Cross-validation analysis
```python
df_s2 = pd.read_csv('TableS2_Cross_Validation.csv')
print(f"Mean test APE: {df_s2['test_median_ape'].mean():.3f}")
print(f"Mean overfit: {df_s2['overfit_pct'].mean():.1f}%")
```

### Table S3: Sensitivity analysis
```python
df_s3 = pd.read_csv('TableS3_Parameter_Sensitivity.csv')
print(df_s3[['Parameter', 'Nominal', 'Sensitivity']])
```

---

## Citation

If using these tables, please cite:
```
Speiser, H. 2025, "Geometry-Gated Gravity: Surface Density and Curvature 
Determine Flat Galaxy Rotation Curves," ApJ/MNRAS (submitted)
```

---

## Reproducibility

All tables generated from:
- **Source data:** `O2_ratio_curv_publication/results/best_fit/mape_median_20250926_2259/`
- **Best-fit params:** `best_family.json`
- **Scripts:** `01_core_publication/supplementary_tables/generate_tables.py`

To regenerate:
```bash
cd O2_ratio_curv_publication/01_core_publication/supplementary_tables
python generate_tables.py
```

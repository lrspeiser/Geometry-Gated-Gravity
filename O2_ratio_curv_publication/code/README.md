# O2 ratio_curv Analysis Code

**Directory:** `O2_ratio_curv_publication/code/`  
**Purpose:** Core analysis scripts used to generate O2 ratio_curv results  
**Date:** October 2, 2025

---

## 📂 Directory Structure

```
code/
├── core_analysis/        # Main O2 fitting and diagnostic scripts
├── data_loaders/         # SPARC data loading utilities
├── features/             # Geometry feature computation
└── README.md            # This file
```

---

## 🔬 Core Analysis Scripts (`core_analysis/`)

### 1. **`global_fit_o2.py`** - Main O2 Family Fitting

**Purpose:** Fits all O2 families (ratio, exp, ratio_curv, exp_curv, ratio_curv_gbar) globally on SPARC galaxies.

**Key Features:**
- Implements 5 functional families with different feature combinations
- Supports multiple objective functions: MSE, median APE, Huber loss
- Uses L-BFGS-B optimization with bounds
- Generates per-galaxy metrics, montages, and summary JSON

**Usage:**
```bash
python -m gravity_learn.eval.global_fit_o2 \
    --objective mape_median \
    --outdir gravity_learn/experiments/eval/global_fit/run_YYYYMMDD_HHMMSS
```

**Outputs:**
- `best_family.json` - Best model parameters and metrics
- `summary_<family>_<timestamp>.json` - Per-family results
- `per_galaxy_metrics_<family>_<timestamp>.csv` - Detailed per-galaxy fits
- `montage_<family>_<timestamp>.png` - 16-galaxy rotation curve overlays

**O2 Families Defined:**
1. **ratio** (2 params): fX = x²/(a - b·Σ̂)
2. **exp** (2 params): fX = α·x²·(exp(Σ̂) + c)
3. **ratio_curv** (3 params): fX = x²/(a - b·Σ̂ - d·|∇ln Σ|) ✅ **BEST**
4. **exp_curv** (3 params): fX = α·x²·(exp(Σ̂) + c + d·|∇ln Σ|)
5. **ratio_curv_gbar** (4 params): fX = x²/(a - b·Σ̂ - d·|∇ln Σ| + e·√g_bar)

**Best Model:** `ratio_curv` with median APE optimization
- a = 0.6686576907182596
- b = 0.14007773322620287
- d = 0.08713057433850588
- Median APE = 0.242 (24.2%)
- Median RMSE = 24.4 km/s

---

### 2. **`plot_rc_overlays_best.py`** - Rotation Curve Visualization

**Purpose:** Generate publication-quality rotation curve overlay plots for best-fit O2 model.

**Key Features:**
- Reads best_family.json for parameters
- Creates N-galaxy montage (default 16, configurable)
- Shows observed data, baryons-only, and O2 model
- Includes per-galaxy MAPE and RMSE

**Usage:**
```bash
python -m gravity_learn.eval.plot_rc_overlays_best \
    --best_json path/to/best_family.json \
    --outdir path/to/output \
    --limit_galaxies 16
```

**Outputs:**
- `montage_best_<family>_<timestamp>.png` - Publication-quality figure

---

### 3. **`mw_residual_diagnostics.py`** - Milky Way Transfer Test

**Purpose:** Validate O2 model on independent Gaia Milky Way rotation curve data.

**Key Features:**
- Uses SPARC-global parameters (no MW-specific tuning)
- Computes residuals vs. geometry features (x, Σ̂, ∇ln Σ)
- Generates heat maps and scatter plots
- Tests generalization beyond training set

**Usage:**
```bash
python -m gravity_learn.eval.mw_residual_diagnostics \
    --best_json path/to/best_family.json \
    --outdir path/to/output
```

**Outputs:**
- `mw/mw_overlay_<family>_<timestamp>.png` - MW rotation curve overlay
- `mw/mw_residuals_points_<family>_<timestamp>.csv` - Per-point residuals
- `mw/mw_resid_vs_<feature>_<family>_<timestamp>.png` - Diagnostic plots (4 features)
- `mw/mw_heat_mape_<features>_<family>_<timestamp>.png` - 2D heat maps

**Key Result:** 94.6% median outer-bin closeness with frozen SPARC parameters

---

### 4. **`lensing_o2_diagnostics.py`** - Cluster Lensing Test

**Purpose:** Evaluate O2 model on cluster-scale strong lensing (failure demonstration).

**Key Features:**
- Computes lensing profiles (convergence κ, shear γ, deflection α)
- Finds Einstein radius (θ_E) from κ̄(<R) ≥ 1 criterion
- Tests on Abell 1689, Bullet, Coma clusters
- Documents systematic underprediction (40-140× too small)

**Usage:**
```bash
python -m gravity_learn.eval.lensing_o2_diagnostics \
    --best_json path/to/best_family.json \
    --outdir path/to/output
```

**Outputs:**
- `lensing_o2/summaries_o2.json` - Summary of all clusters
- `lensing_o2/<cluster>/profiles_o2.csv` - Radial profiles
- `lensing_o2/<cluster>/summary_o2.json` - Per-cluster metrics
- `lensing_o2/<cluster>/kappa_gamma_o2.png` - Convergence & shear plot
- `lensing_o2/<cluster>/deflection_o2.png` - Deflection angle plot
- `lensing_o2/<cluster>/surface_density_o2.png` - Σ_eff plot

**Key Finding:** Model systematically fails at cluster scales
- Abell 1689: θ_E,obs = 47", θ_E,pred = 0.33" (140× low)
- This is a fundamental limitation, not a tuning issue

---

## 📊 Data Loaders (`data_loaders/`)

### **`data.py`** - SPARC Data Loading

**Purpose:** Load and parse SPARC (Spitzer Photometry and Accurate Rotation Curves) dataset.

**Key Functions:**
- `load_sparc()` - Main loader, returns SPARCDataset object
- Reads parquet file: `data/sparc_rotmod_ltg.parquet`
- Extracts: R_kpc, Vobs_kms, Verr_kms, Vbar_kms, Sigma_bar, Rd_kpc, Type
- Handles missing data and quality filtering

**Usage:**
```python
from rigor.rigor.data import load_sparc

ds = load_sparc()
for galaxy in ds.galaxies:
    print(galaxy.name, galaxy.Type, galaxy.Rd_kpc)
```

**Data Source:**
- **SPARC:** Lelli et al. (2016), AJ, 152, 157
- **URL:** http://astroweb.cwru.edu/SPARC/
- **License:** CC BY 4.0

---

## 🧮 Features (`features/`)

### **`geometry.py`** - Geometry Feature Computation

**Purpose:** Compute geometry features for O2 model: dimensionless radius, normalized surface density, logarithmic gradient.

**Key Functions:**

1. **`dimensionless_radius(R_kpc, Rd=None)`**
   - Returns: x = R / Rd
   - Rd estimated from Σ(R) exponential fit if not provided
   
2. **`sigma_hat(Sigma_Msun_pc2)`**
   - Returns: Σ̂ = log₁₀(Σ / 100)
   - Reference: 100 M☉/pc² (typical dwarf central density)
   
3. **`grad_log_sigma(R_kpc, Sigma_Msun_pc2)`**
   - Returns: |d(ln Σ)/dR|
   - Uses centered finite differences with 3-point smoothing

**Usage:**
```python
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

x = dimensionless_radius(R_kpc, Rd=2.5)
Sigma_hat = sigma_hat(Sigma_Msun_pc2)
grad_ln_Sigma = grad_log_sigma(R_kpc, Sigma_Msun_pc2)
```

**Feature Ranges (typical):**
- x: [0, 10] (captures 3-5 scale lengths)
- Σ̂: [-2, 2] (inner galaxies ~+1, outer ~-1)
- |∇ln Σ|: [0, 1] kpc⁻¹ (steeper gradients → larger values)

---

## 🔄 Workflow: From Data to Results

**Step 1:** Load SPARC data
```python
from rigor.rigor.data import load_sparc
ds = load_sparc()
```

**Step 2:** Compute geometry features for each galaxy
```python
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

for g in ds.galaxies:
    x = dimensionless_radius(g.R_kpc, Rd=g.Rd_kpc)
    Sigma_hat = sigma_hat(g.Sigma_bar)
    grad_ln_Sigma = grad_log_sigma(g.R_kpc, g.Sigma_bar)
```

**Step 3:** Fit O2 families globally
```bash
python -m gravity_learn.eval.global_fit_o2 --objective mape_median --outdir results/
```

**Step 4:** Select best family (ratio_curv) based on loss

**Step 5:** Generate diagnostics
```bash
python -m gravity_learn.eval.plot_rc_overlays_best --best_json results/best_family.json --outdir figures/
python -m gravity_learn.eval.mw_residual_diagnostics --best_json results/best_family.json --outdir results/
python -m gravity_learn.eval.lensing_o2_diagnostics --best_json results/best_family.json --outdir results/
```

---

## 📈 Performance Summary

**Best Model:** O2 `ratio_curv` (3 parameters, median APE optimization)

| Metric | Value | Comparison |
|--------|-------|------------|
| **Median APE** | 0.242 (24.2%) | MOND ~0.25, GR ~0.36 |
| **Median RMSE** | 24.4 km/s | MOND ~25, GR ~45 |
| **Parameters** | 3 global | NFW: ~3 per galaxy (360 total) |
| **MW Transfer Test** | 94.6% accuracy | No retuning |
| **Cluster Lensing** | 40-140× low | Fundamental limitation |

---

## 🛠️ Dependencies

**Python 3.9+**

**Required packages:**
- numpy >= 1.24
- scipy >= 1.10
- pandas >= 1.5
- matplotlib >= 3.7

**Optional (for symbolic regression):**
- PySR >= 0.16

**Install:**
```bash
pip install numpy scipy pandas matplotlib
```

---

## 📝 Citation

If you use this code, please cite:

```
Speiser, H. 2025, "Geometry-Gated Gravity: Surface Density and Curvature 
Determine Flat Galaxy Rotation Curves," [Journal TBD]

Data: Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157 (SPARC)
```

---

## 🔗 Related Files

- **Results:** `O2_ratio_curv_publication/results/best_fit/`
- **Paper:** `O2_ratio_curv_publication/01_core_publication/PAPER_O2_RATIO_CURV.md`
- **Master TODO:** `O2_ratio_curv_publication/MASTER_TODO.md`

---

## 📧 Contact

**Author:** Henry Speiser  
**Repository:** https://github.com/lrspeiser/GravityCalculator  
**Issues:** https://github.com/lrspeiser/GravityCalculator/issues

---

**Last Updated:** October 2, 2025

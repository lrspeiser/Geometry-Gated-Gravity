# O2 ratio_curv Publication Project - Complete Inventory

**Date:** October 2, 2025  
**Status:** Ready for figure generation and finalization  

---

## ✅ Completeness Check

### **Core Documents**
- ✅ `MASTER_TODO.md` - 586 lines, 11 research paths documented
- ✅ `01_core_publication/PAPER_O2_RATIO_CURV.md` - 1,227 lines, complete manuscript
- ✅ `documentation/MODEL_RECOMMENDATION.md` - 425 lines, model selection rationale
- ✅ `code/README.md` - 304 lines, complete code documentation
- ✅ `results/README.md` - 372 lines, complete results documentation
- ✅ `INVENTORY.md` - This file

**Total Documentation:** 2,993 lines across 6 files

---

## 📊 Analysis Code (6 files)

### **Core Analysis Scripts**
1. ✅ `code/core_analysis/global_fit_o2.py` (236 lines)
   - Fits 5 O2 families globally on SPARC
   - Median APE, MSE, Huber loss support
   - Parameters: (a, b, d) for ratio_curv

2. ✅ `code/core_analysis/plot_rc_overlays_best.py`
   - Rotation curve visualization
   - N-galaxy montages
   - Publication-quality formatting

3. ✅ `code/core_analysis/mw_residual_diagnostics.py`
   - Milky Way transfer test
   - SPARC-global parameters frozen
   - Heat maps and residual plots

4. ✅ `code/core_analysis/lensing_o2_diagnostics.py`
   - Cluster lensing evaluation
   - Einstein radius computation
   - Failure demonstration

### **Supporting Libraries**
5. ✅ `code/data_loaders/data.py`
   - SPARC data loading
   - Parquet file handling
   - Quality filtering

6. ✅ `code/features/geometry.py`
   - Dimensionless radius (x = R/Rd)
   - Normalized surface density (Σ̂)
   - Logarithmic gradient (∇ln Σ)

---

## 📈 Results Files (65 files)

### **Best-Fit Parameters**
✅ `results/best_fit/mape_median_20250926_2259/best_family.json`
```json
{
  "best_family": "ratio_curv",
  "params": [0.6687, 0.1401, 0.0871],
  "loss": 0.1975,
  "objective": "mape_median"
}
```

### **Per-Galaxy Metrics (6 CSV files)**
✅ `per_galaxy_metrics_best_ratio_curv_20250926_160300.csv` - 120 galaxies
✅ `per_galaxy_metrics_ratio_20250926_160239.csv` - 2-param comparison
✅ `per_galaxy_metrics_exp_20250926_160239.csv` - Exponential comparison
✅ `per_galaxy_metrics_ratio_curv_gbar_20250926_160242.csv` - 4-param comparison
✅ `per_galaxy_metrics_exp_curv_20250926_160241.csv` - 3-param exp comparison
✅ `per_galaxy_metrics_ratio_curv_20250926_160240.csv` - Original ratio_curv

### **Summary Files (5 JSON files)**
✅ `summary_ratio_curv_20250926_160240.json`
✅ `summary_ratio_20250926_160239.json`
✅ `summary_exp_20250926_160239.json`
✅ `summary_exp_curv_20250926_160241.json`
✅ `summary_ratio_curv_gbar_20250926_160242.json`

### **Montage Figures (6 PNG files)**
✅ `montage_best_ratio_curv_20250926_160300.png` - **Use for Figure 1**
✅ `montage_ratio_curv_20250926_160240.png` - Original fit
✅ `montage_ratio_20250926_160239.png` - 2-param ratio
✅ `montage_exp_20250926_160239.png` - 2-param exp
✅ `montage_exp_curv_20250926_160241.png` - 3-param exp
✅ `montage_ratio_curv_gbar_20250926_160242.png` - 4-param with g_bar

### **SPARC Diagnostics (15 files)**
**CSV Files (3):**
✅ `diagnostics/residuals_points_ratio_curv_20250926_160339.csv` - 2,847 points
✅ `diagnostics/residuals_per_galaxy_ratio_curv_20250926_160339.csv` - 120 galaxies
✅ `diagnostics/tail_galaxies_by_median_ape_ratio_curv_20250926_160339.csv` - Sorted by APE

**Diagnostic Plots (10 PNG files):**
✅ `diagnostics/resid_vs_R_*.png` - Residuals vs. radius
✅ `diagnostics/resid_vs_x_*.png` - Residuals vs. dimensionless radius
✅ `diagnostics/resid_vs_Sh_*.png` - Residuals vs. Σ̂
✅ `diagnostics/resid_vs_absdlnS_*.png` - Residuals vs. |∇ln Σ|
✅ `diagnostics/ape_vs_R_*.png` - APE vs. radius
✅ `diagnostics/ape_vs_x_*.png` - APE vs. dimensionless radius
✅ `diagnostics/ape_vs_Sh_*.png` - APE vs. Σ̂
✅ `diagnostics/ape_vs_absdlnS_*.png` - APE vs. |∇ln Σ|
✅ `diagnostics/heat_mape_x_Sh_*.png` - 2D heat map (x, Σ̂)
✅ `diagnostics/heat_mape_x_absdlnS_*.png` - 2D heat map (x, |∇ln Σ|)

### **Milky Way Transfer Test (13 files)**
**CSV File (1):**
✅ `mw/mw_residuals_points_ratio_curv_20250926_222323.csv`

**MW Diagnostic Plots (12 PNG files):**
✅ `mw/mw_overlay_*.png` - **Use for Figure 2**
✅ `mw/mw_resid_vs_R_*.png`
✅ `mw/mw_resid_vs_x_*.png`
✅ `mw/mw_resid_vs_Sh_*.png`
✅ `mw/mw_resid_vs_absdlnS_*.png`
✅ `mw/mw_ape_vs_R_*.png`
✅ `mw/mw_ape_vs_x_*.png`
✅ `mw/mw_ape_vs_Sh_*.png`
✅ `mw/mw_ape_vs_absdlnS_*.png`
✅ `mw/mw_heat_mape_x_Sh_*.png`
✅ `mw/mw_heat_mape_x_absdlnS_*.png`

### **Cluster Lensing Tests (16 files)**
**Summary (1 JSON):**
✅ `lensing_o2/summaries_o2.json`

**Per-Cluster Results (3 clusters × 5 files each = 15 files):**

**Abell 1689:**
✅ `lensing_o2/Abell_1689/profiles_o2.csv`
✅ `lensing_o2/Abell_1689/summary_o2.json`
✅ `lensing_o2/Abell_1689/kappa_gamma_o2.png` - **Use for Figure 4a**
✅ `lensing_o2/Abell_1689/deflection_o2.png`
✅ `lensing_o2/Abell_1689/surface_density_o2.png`

**Bullet:**
✅ `lensing_o2/Bullet/profiles_o2.csv`
✅ `lensing_o2/Bullet/summary_o2.json`
✅ `lensing_o2/Bullet/kappa_gamma_o2.png`
✅ `lensing_o2/Bullet/deflection_o2.png`
✅ `lensing_o2/Bullet/surface_density_o2.png`

**Coma:**
✅ `lensing_o2/Coma/profiles_o2.csv`
✅ `lensing_o2/Coma/summary_o2.json`
✅ `lensing_o2/Coma/kappa_gamma_o2.png`
✅ `lensing_o2/Coma/deflection_o2.png`
✅ `lensing_o2/Coma/surface_density_o2.png`

### **Augmented Lensing (6 files)**
**Summary (1 JSON):**
✅ `lensing_o2_aug/summaries_aug.json`

**Per-Cluster Augmented (5 PNG files):**
✅ `lensing_o2_aug/Abell_1689/kappa_mean_aug.png`
✅ `lensing_o2_aug/A2029/kappa_mean_aug.png`
✅ `lensing_o2_aug/A478/kappa_mean_aug.png`
✅ `lensing_o2_aug/A1795/kappa_mean_aug.png`
✅ `lensing_o2_aug/ABELL_0426/kappa_mean_aug.png`

---

## 🎨 Publication Figures (To Generate)

### **Figure 1: Rotation Curve Overlays (6 SPARC Galaxies)**
- **Status:** ✅ Script created (`01_core_publication/generate_figure1_rc_overlays.py`)
- **Alternative:** Can use existing `montage_best_ratio_curv_20250926_160300.png`
- **Shows:** NGC2403, NGC3198, DDO154, UGC2885, F563-1, NGC7793
- **Output:** `figures/Figure1_RC_Overlays.png` + `.pdf`

### **Figure 2: Milky Way Rotation Curve**
- **Status:** ✅ Image exists (`mw/mw_overlay_ratio_curv_20250926_222323.png`)
- **Action:** Copy to `figures/Figure2_MW_RotationCurve.png`
- **Shows:** Gaia bins ±1σ with O2 model overlay

### **Figure 3: Residual Diagnostics (4-panel)**
- **Status:** ⏳ Need to create composite
- **Source Images:** `diagnostics/resid_vs_*.png` (4 plots)
- **Layout:** 2×2 grid
- **Panels:** (a) Residuals vs R, (b) Residuals vs Σ̂, (c) APE vs R, (d) Histogram of APE
- **Output:** `figures/Figure3_Residual_Diagnostics.png` + `.pdf`

### **Figure 4: Cluster Lensing Failure (3-panel)**
- **Status:** ⏳ Need to create composite
- **Source Images:** `lensing_o2/<cluster>/kappa_gamma_o2.png` (3 clusters)
- **Layout:** 1×3 horizontal
- **Panels:** (a) Abell 1689, (b) Bullet, (c) Coma
- **Output:** `figures/Figure4_Cluster_Lensing.png` + `.pdf`

---

## 📋 Supplementary Tables (To Create)

### **Table S1: Per-Galaxy Metrics (120 galaxies)**
- **Source:** `per_galaxy_metrics_best_ratio_curv_20250926_160300.csv`
- **Columns:** Galaxy, Type, Rd_kpc, n_points, median_ape, rmse_kms
- **Action:** Already exists as CSV, format for journal submission

### **Table S2: Cross-Validation Results**
- **Source:** Extract from `diagnostics/residuals_per_galaxy_*.csv`
- **Action:** Create 5-fold CV summary table
- **Columns:** Fold, Train_Galaxies, Test_Galaxies, Train_APE, Test_APE, Overfit

### **Table S3: Parameter Sensitivity Analysis**
- **Source:** To be computed (±10% parameter perturbations)
- **Action:** Create sensitivity analysis script
- **Columns:** Parameter, Nominal, -10%, +10%, Sensitivity

---

## ✅ Completeness Summary

### **What We Have:**
- ✅ **Complete manuscript** (1,227 lines)
- ✅ **All analysis code** (6 scripts)
- ✅ **All results files** (65 files: 12 CSVs, 7 JSONs, 46 PNGs)
- ✅ **Comprehensive documentation** (676 lines across 2 READMEs)
- ✅ **Model recommendation** (425 lines)
- ✅ **Master TODO** (586 lines with 11 research paths)

### **What We Need to Generate:**
- ⏳ **Figure 1** - Can use existing montage OR run script
- ⏳ **Figure 2** - Copy existing MW overlay
- ⏳ **Figure 3** - Create 4-panel composite from diagnostics
- ⏳ **Figure 4** - Create 3-panel composite from lensing
- ⏳ **Table S2** - Extract/format CV results
- ⏳ **Table S3** - Run sensitivity analysis

### **What We Can Skip (Already Exists):**
- All rotation curve overlays (6 montages)
- All diagnostic plots (10 SPARC + 11 MW = 21 plots)
- All lensing plots (9 cluster + 5 augmented = 14 plots)
- All CSV data tables (12 files)
- All JSON summaries (8 files)

---

## 📊 File Count Summary

| Category | Count | Status |
|----------|-------|--------|
| **Markdown Docs** | 6 | ✅ Complete |
| **Python Scripts** | 6 | ✅ Complete |
| **CSV Data** | 12 | ✅ Complete |
| **JSON Results** | 8 | ✅ Complete |
| **PNG Images** | 46 | ✅ Complete |
| **Publication Figures** | 4 | ⏳ 0/4 generated |
| **Supplementary Tables** | 3 | ⏳ 1/3 exist as CSV |

**Total Files in O2 Project:** 82 (78 complete, 4 to generate)

---

## 🎯 Next Steps (Priority Order)

1. **Generate Figure 1** - Run script OR copy montage
2. **Create Figure 2** - Copy MW overlay to figures/
3. **Create Figure 3** - Composite 4-panel residual diagnostic
4. **Create Figure 4** - Composite 3-panel cluster lensing
5. **Format Table S2** - Extract CV results
6. **Generate Table S3** - Parameter sensitivity
7. **Update paper** - Insert figure references
8. **Final review** - Check all cross-references
9. **Generate arXiv PDF** - Markdown → LaTeX → PDF
10. **Submit to journal** - ApJ or MNRAS

---

## 📧 Verification Commands

**Count all files:**
```bash
fd -H . O2_ratio_curv_publication/ | wc -l
```

**Count by type:**
```bash
fd -H -e md O2_ratio_curv_publication/ | wc -l  # Markdown
fd -H -e py O2_ratio_curv_publication/ | wc -l  # Python
fd -H -e csv O2_ratio_curv_publication/ | wc -l # CSV
fd -H -e json O2_ratio_curv_publication/ | wc -l # JSON
fd -H -e png O2_ratio_curv_publication/ | wc -l # PNG
```

**Verify READMEs:**
```bash
fd -H README.md O2_ratio_curv_publication/
```

**Check for missing critical files:**
```bash
test -f O2_ratio_curv_publication/results/best_fit/mape_median_20250926_2259/best_family.json && echo "✅ Best fit exists"
test -f O2_ratio_curv_publication/code/core_analysis/global_fit_o2.py && echo "✅ Fitting code exists"
test -f O2_ratio_curv_publication/01_core_publication/PAPER_O2_RATIO_CURV.md && echo "✅ Paper exists"
```

---

**Last Updated:** October 2, 2025  
**Status:** Ready for figure generation phase

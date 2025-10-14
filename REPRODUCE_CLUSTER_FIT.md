# Reproducibility Guide: Sigma-Gravity Cluster Hierarchical Fit

**Date:** 2025-01-14  
**Version:** 1.0  
**Status:** Ready for execution after cluster catalog preparation

---

## Overview

This document provides complete reproducibility for the Sigma-Gravity baryons-only cluster lensing calibration. The fit learns a **universal many-paths kernel** from 9 training clusters and validates on 3 blind holdouts.

**Key Result:** Baryons + geometry + Σ-Gravity path accumulation → Einstein radii WITHOUT dark matter.

---

## Prerequisites

### Software Requirements

```bash
# Python 3.9+
python --version  # Should be ≥3.9

# Required packages
pip install numpy scipy pandas matplotlib astropy
pip install scikit-learn  # for train/test splits
pip install corner emcee  # optional, for MCMC posteriors
```

### Data Requirements

**Cluster catalog:** `data/cluster_catalog.csv`

Required columns:
- `name` (str): Cluster identifier
- `M_500` (float): Mass [Msun]
- `R_500` (float): R500 radius [kpc]
- `z_lens` (float): Cluster redshift
- `z_source` (float): Source redshift for lensing
- `theta_E_obs` (float): Observed Einstein radius [arcsec]
- `theta_E_err` (float): Error on theta_E [arcsec]
- `f_gas` (float): Gas fraction (optional, defaults to 0.11)

**Optional weak lensing:**
- `gamma_t_R` (array): Radii for shear measurements [kpc]
- `gamma_t_obs` (array): Tangential shear
- `gamma_t_err` (array): Errors

---

## Phase 0: Validation (Already Complete ✓)

The triaxial lensing module has been fixed and validated:

```bash
# Rerun validation if needed
python scripts/validate_triaxial_lensing.py
```

**Expected output:**
- All 5 tests PASS
- ~60% Einstein radius sensitivity to q_LOS
- Validation report: `docs/triaxial_lensing_fix_report.md`

**Status:** ✓ PASSED (2025-01-14)

---

## Phase 1: Single-Cluster Sanity Check

Before running the full hierarchical fit, verify physics on one well-studied cluster (e.g., MACS0416):

```bash
# Run standalone MACS0416 test
python scripts/test_cluster_macs0416_triaxial.py
```

**Acceptance criteria:**
1. With unified physics (clumping correction, f_gas=0.11):
   - Spherical (q=1.0): theta_E within ~10% of observed
   - Prolate (q_LOS ~ 1.15-1.2): theta_E matches observed within error
2. Geometry sensitivity: ~20-30% swing across q_LOS ∈ [0.7, 1.3]
3. No NaNs, no negative densities, mass conservation holds

**If this fails:** Debug single-cluster physics before proceeding to hierarchical fit.

---

## Phase 2: Hierarchical Training Fit (9/3 Split)

### Step 2.1: Run Training Fit

```bash
# Full hierarchical calibration with strong lensing only
python scripts/run_cluster_hierarchical_fit.py \
    --catalog data/cluster_catalog.csv \
    --output results/cluster_fit_v1 \
    --mode train
```

**What this does:**
1. Loads cluster catalog and splits into 9 train / 3 holdout
2. Initializes global kernel from priors (A_c=10, ell0=180, etc.)
3. Alternates optimization:
   - Fix geometry → optimize global kernel (shared by all 9)
   - Fix kernel → optimize each cluster's geometry (q_plane, q_LOS, kappa_ext)
4. Applies physically motivated priors:
   - Sparsity on w_exterior (Laplace at 0)
   - Broad but bounded geometry (q_LOS ∈ [0.7, 1.4])
   - Tight prior on external sheet (kappa_ext ~ N(0, 0.03²))
5. Saves results to `results/cluster_fit_v1/`

**Expected runtime:** ~30-60 minutes (depends on cluster count and n_iterations)

### Step 2.2: Check Training Metrics

```bash
# View results summary
cat results/cluster_fit_v1/results.json
```

**Success criteria:**
- **Training chi²/dof < 1.2** (good fit to training data)
- **Training median fractional error < 15%** (θ_E predictions accurate)
- **Holdout chi²/dof < 1.5** (generalization to unseen clusters)
- **Holdout median fractional error < 20%** (blind predictions reasonable)
- **w_exterior < 0.15** (exterior arcs stay sparse due to prior)
- **Geometry parameters physical** (not hitting prior bounds)

**If training fails:**
- Check for NaNs in predictions → debug baryon model
- If chi²/dof >> 2 → may need to relax priors or add systematic uncertainties
- If w_exterior → 0.3 → exterior arcs being forced, check interior-only kernel first

### Step 2.3: Inspect Geometry Posteriors

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('results/cluster_fit_v1/results.json') as f:
    results = json.load(f)

# Extract q_LOS values
q_LOS_values = [results['best_geometry'][name]['q_LOS'] 
                for name in results['best_geometry']]

# Plot distribution
plt.hist(q_LOS_values, bins=10, edgecolor='black')
plt.xlabel('q_LOS')
plt.ylabel('Count')
plt.title('Fitted LOS Axis Ratios (Training Clusters)')
plt.axvline(1.0, color='red', linestyle='--', label='Spherical')
plt.legend()
plt.savefig('results/cluster_fit_v1/q_LOS_distribution.png', dpi=150)
```

**Expected:**
- q_LOS distribution centered near 1.0 (spherical) with scatter ~0.2
- Possible tail toward prolate (q_LOS > 1.1) for merging clusters
- No extreme outliers (all within [0.7, 1.4])

---

## Phase 3: Add Weak Lensing Constraints

Once strong lensing is working, add weak lensing to break degeneracies:

```bash
# Run with joint strong + weak loss
python scripts/run_cluster_hierarchical_fit.py \
    --catalog data/cluster_catalog_with_WL.csv \
    --output results/cluster_fit_v2_WL \
    --mode train \
    --lambda_WL 1.0
```

**What changes:**
- Loss = chi²_SL + lambda_WL × chi²_WL
- Weak lensing constrains outer profile → tighter bounds on A_c and ell0
- Reduces q_LOS ↔ A_c degeneracy (WL sees projected mass, not just θ_E)

**Success criteria:**
- chi²_SL/dof and chi²_WL/dof both < 1.5
- Geometry parameters tighten (smaller posterior widths)
- Holdout performance improves or stays stable

---

## Phase 4: Ablation Studies (Degeneracy Mapping)

Run ablation tests to prove the result is robust and not hiding dark matter in geometry:

### Ablation 1: Interior vs Exterior Paths

Test how much exterior arcs contribute:

```bash
python scripts/run_cluster_ablation.py \
    --type interior_vs_exterior \
    --output results/ablations/interior_vs_exterior
```

**Produces:**
- Plot: theta_E vs w_exterior for each cluster
- Quantifies: How much does w_ext=0.15 improve fit over w_ext=0?
- Interpretation: If w_ext is needed, verify it's small and consistent with WL

### Ablation 2: Triaxial vs External Sheet

Map (q_LOS, kappa_ext) degeneracy:

```bash
python scripts/run_cluster_ablation.py \
    --type triaxial_vs_kappa_ext \
    --output results/ablations/triaxial_vs_kappa_ext
```

**Produces:**
- 2D contour plot: chi² in (q_LOS, kappa_ext) plane for MACS0416
- Shows: Two parameters are NOT perfectly degenerate
- Interpretation: Geometry and mass sheet have different radial signatures

### Ablation 3: Clumping vs Kernel Amplitude

Test if clumping can fake the kernel boost:

```bash
python scripts/run_cluster_ablation.py \
    --type clumping_vs_amplitude \
    --output results/ablations/clumping_vs_amplitude
```

**Produces:**
- Grid scan: theta_E vs (C0, A_c)
- Shows: Clumping alone cannot match observations (suppresses mass)
- Interpretation: Path accumulation (A_c) is physically distinct from clumping

---

## Phase 5: Galaxy RAR Consistency Check

After fitting cluster kernel, verify it doesn't break galaxies:

```bash
# Rerun SPARC RAR fit with updated cluster kernel
python many_path_model/sparc_hierarchical_search_v2.py \
    --kernel_from results/cluster_fit_v2_WL/results.json \
    --output results/galaxies_with_cluster_kernel
```

**Success criteria:**
- RAR scatter stays ≤ 0.13 dex (allow ~50% degradation from galaxy-only 0.087 dex)
- No systematic bias in v_circ predictions
- Newtonian limit (r << ell0) still matches Solar System

**If RAR degrades badly (>0.20 dex):**
- Cluster and galaxy kernels may need scale-dependent tuning
- Consider separate ell0 for clusters (ell0_cluster ~ 180 kpc) vs galaxies (ell0_gal ~ 5 kpc)
- Document any differences as scale-dependent physics

---

## Phase 6: Final Diagnostics & Plots for Paper

### Generate Figure Set

```bash
# Master plotting script
python scripts/generate_cluster_paper_figures.py \
    --results results/cluster_fit_v2_WL \
    --output figures/cluster_paper_figs
```

**Produces:**
1. **Fig 1:** Triaxial validation plots (already in `figures/triaxial_validation.png`)
2. **Fig 2:** Training fit convergence (loss vs iteration)
3. **Fig 3:** Theta_E predictions vs observations (train + holdout)
4. **Fig 4:** Weak lensing profiles (gamma_t vs R) for 3 example clusters
5. **Fig 5:** Geometry posterior distributions (q_plane, q_LOS histograms)
6. **Fig 6:** Ablation study (interior vs exterior, triaxial vs kappa_ext)
7. **Fig 7:** Galaxy+cluster RAR consistency (0.087 dex galaxies + cluster predictions)

### Tables for Paper

```bash
# Generate LaTeX tables
python scripts/generate_cluster_paper_tables.py \
    --results results/cluster_fit_v2_WL \
    --output tables/
```

**Produces:**
- **Table 1:** Cluster catalog (name, M500, R500, z, theta_E_obs, theta_E_pred)
- **Table 2:** Best-fit global kernel parameters with uncertainties
- **Table 3:** Per-cluster geometry parameters (q_plane, q_LOS, kappa_ext)
- **Table 4:** Train/holdout chi² breakdown

---

## Phase 7: Error Budget & Systematics

Document all sources of uncertainty:

### Statistical Uncertainties

From MCMC posteriors (if run):
```bash
# Optional: full Bayesian inference with emcee
python scripts/run_cluster_mcmc.py \
    --results results/cluster_fit_v2_WL \
    --output results/cluster_mcmc_posteriors \
    --nwalkers 32 --nsteps 10000
```

**Produces:**
- Corner plots for (A_c, ell0, p, n_coh, w_ext)
- Credible intervals (16th, 50th, 84th percentiles)

### Systematic Uncertainties

Test sensitivity to assumptions:

1. **Clumping model:** Vary (C0, C_max) within literature ranges → Δθ_E
2. **f_gas normalization:** Vary 0.10-0.12 → Δθ_E
3. **Cosmology:** Vary H0 = 67-73 km/s/Mpc → Δθ_E
4. **BCG/ICL masses:** Vary by ±30% → Δθ_E
5. **Critical density Σ_crit:** Recompute with astropy → Δθ_E

**Document in table:**
| Systematic | Method | Δθ_E (median) |
|------------|--------|---------------|
| Clumping   | C0 ± 0.2 | ~2% |
| f_gas      | ±0.01 | ~5% |
| Cosmology  | H0 ± 3 | ~4% |
| BCG/ICL    | ±30% | ~3% |
| **Total**  | Quadrature | ~8% |

---

## Expected Final Results

### Training Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Training chi²/dof | < 1.2 | TBD |
| Training median error | < 15% | TBD |
| Holdout chi²/dof | < 1.5 | TBD |
| Holdout median error | < 20% | TBD |

### Best-Fit Kernel

| Parameter | Prior Mean | Fitted Value | Interpretation |
|-----------|------------|--------------|----------------|
| A_c | 10.0 | TBD ± TBD | Cluster boost amplitude |
| ell0 [kpc] | 180.0 | TBD ± TBD | Coherence length |
| p_density | 1.2 | TBD ± TBD | Density scaling |
| n_coh | 1.5 | TBD ± TBD | Coherence damping |
| w_exterior | 0.0 | TBD ± TBD | Exterior arc weight (sparse!) |

### Geometry Population

| Statistic | q_plane | q_LOS |
|-----------|---------|-------|
| Median | TBD | TBD |
| 16-84% range | TBD | TBD |
| Physical? | ✓ (0.6-1.0) | ✓ (0.7-1.4) |

---

## Troubleshooting

### Problem: Training chi² > 3

**Diagnosis:** Underfitting - model cannot match observations.

**Solutions:**
1. Check baryon model normalization (f_gas, M500)
2. Relax priors on A_c (allow 5-30 range)
3. Add small w_exterior (0.05-0.15)
4. Check for outlier clusters (high chi² contribution)

### Problem: Holdout chi² >> Training chi²

**Diagnosis:** Overfitting - geometry masquerading as physics.

**Solutions:**
1. Tighten priors on q_LOS (use narrower population prior)
2. Increase regularization (stronger Laplace prior on w_ext)
3. Reduce n_iterations (stop earlier)
4. Cross-validate with different train/holdout splits

### Problem: w_exterior → 0.3 (hitting bound)

**Diagnosis:** Exterior arcs being forced to contribute.

**Solutions:**
1. Check if interior-only kernel is physically correct
2. Verify weak lensing constraints (WL disfavors large w_ext)
3. May indicate real physics (exterior paths matter for some clusters)
4. Document as finding, not bug

### Problem: Geometry at prior edges (q_LOS → 0.7 or 1.4)

**Diagnosis:** Prior too tight or data demands extreme geometry.

**Solutions:**
1. Widen prior cautiously (e.g., q_LOS ∈ [0.6, 1.6])
2. Check X-ray/optical morphology for those clusters (are they really extreme?)
3. May indicate mergers or non-virialized systems
4. Consider hierarchical prior on q_LOS_std (learn from data)

---

## Archiving Results

After successful fit:

```bash
# Create versioned archive
cd results/
tar -czf cluster_fit_v2_WL_FINAL.tar.gz cluster_fit_v2_WL/
shasum -a 256 cluster_fit_v2_WL_FINAL.tar.gz > cluster_fit_v2_WL_FINAL.sha256

# Upload to repository
git add results/cluster_fit_v2_WL_FINAL.tar.gz
git add results/cluster_fit_v2_WL_FINAL.sha256
git commit -m "Final cluster hierarchical fit results (v2 with WL)"
git push origin main
```

**Include in archive:**
- `results.json` (all parameters and metrics)
- `fit_history.csv` (convergence trace)
- Diagnostic plots (PNG)
- `REPRODUCE.md` (this file)
- Git commit hash of code used

---

## Citation

If using these results, cite:

```bibtex
@article{sigma_gravity_clusters_2025,
  title={Baryons-Only Cluster Lensing via Σ-Gravity Path Accumulation},
  author={[Your Name]},
  journal={TBD},
  year={2025},
  note={Reproducibility: github.com/[repo]/REPRODUCE_CLUSTER_FIT.md}
}
```

---

## Contact

Questions or issues:
- Open GitHub issue: github.com/lrspeiser/Geometry-Gated-Gravity/issues
- Email: [your email]

---

**Status:** Ready to execute  
**Last updated:** 2025-01-14  
**Next action:** Prepare cluster catalog CSV and run Phase 1 single-cluster sanity check.

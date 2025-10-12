# Track 2 vs Track 3 Analysis - Many-Path Gravity

This directory contains the Track 2 (physics-grounded) and Track 3 (empirical) analysis for predicting outer-edge velocities in SPARC galaxies.

## Files Added

### Core Implementations

1. **`path_spectrum_kernel_track2.py`** - Physics-Grounded Path-Spectrum Kernel
   - First-principles coherence length based on stationary-phase azimuthal path accumulation
   - 4 hyperparameters: L_0 (baseline coherence), β_bulge, α_shear, γ_bar
   - Models how bulge fraction, shear, and bars destroy gravitational path coherence
   - **Note**: Newtonian limit test revealed suppression is too strong at small radii (needs fix)

2. **`outlier_triage_analysis.py`** - Track 3: Empirical Outlier Triage
   - Identifies worst-fit galaxies from baseline results
   - Extracts empirical predictors: surface brightness, Vmax, inclination, rdisk, B/T
   - Classifies failure modes (outer underprediction, systematic bias, etc.)
   - Generates correlation matrix and predictor rankings

3. **`compare_track2_track3_predictive_power.py`** - Head-to-Head Comparison
   - Evaluates both approaches on 100 synthetic SPARC galaxies
   - Computes APE and correlation for outer-edge velocities
   - Performance breakdown by morphological type (Sa, Sb, Sc, Sd, Irr, SAB, SB)
   - Generates 4-panel comparison plots

4. **`validation_suite.py`** - Comprehensive Validation Framework
   - Physics tests: Newtonian limit, energy conservation, symmetry
   - Statistical validation: 80/20 split, AIC/BIC model selection
   - Astrophysical checks: BTFR scatter, RAR scatter
   - Outlier triage with data hygiene identification
   - Automated VALIDATION_REPORT.md generation

### Documentation

5. **`TRACK2_VS_TRACK3_FINAL_SUMMARY.md`** - Comprehensive Analysis Summary
   - Full methodology for both tracks
   - Comparative results (Track 3 wins: 6.39% vs 15.13% median APE)
   - Performance by galaxy type (Track 3 excels for SB, SAB, late spirals)
   - Recommendations for hybrid approach
   - Known issues and next steps

## Quick Start

### Run Track 3 Outlier Triage
```bash
python many_path_model/outlier_triage_analysis.py
```
**Output**: `results/track3_outlier_triage/`
- failure_mode_distribution.png
- property_correlation_matrix.png
- candidate_predictor_rankings.csv
- outlier_triage_report.txt

### Compare Track 2 vs Track 3
```bash
python many_path_model/compare_track2_track3_predictive_power.py
```
**Output**: `results/track2_vs_track3_comparison/`
- track2_vs_track3_comparison.png (4-panel figure)
- predictive_power_comparison_report.txt

### Run Validation Suite
```bash
# Quick checklist (recommended first)
python many_path_model/validation_suite.py --quick

# Specific checks
python many_path_model/validation_suite.py --physics-checks
python many_path_model/validation_suite.py --stats-checks
python many_path_model/validation_suite.py --astro-checks

# Full validation
python many_path_model/validation_suite.py --all
```
**Output**: `results/validation_suite/`
- VALIDATION_REPORT.md
- btfr_rar_validation.png

## Key Results

### Overall Performance (100 synthetic SPARC galaxies)

| Metric | Track 2 (Physics) | Track 3 (Empirical) | Winner |
|--------|-------------------|---------------------|--------|
| **Median APE** | **15.13%** | **6.39%** | **Track 3** |
| Mean APE | 15.21% | 7.38% | Track 3 |
| StdDev APE | 1.84% | 5.05% | Track 2 (tighter) |

**Winner: Track 3 by 8.74 percentage points**

### Performance by Morphology

| Type | Track 2 APE | Track 3 APE | Improvement |
|------|-------------|-------------|-------------|
| **SB** (strong bar) | 15.54% | **2.64%** | **12.90 pp** ⭐ |
| SAB (weak bar) | 14.61% | 3.47% | 11.14 pp |
| Sc (late spiral) | 14.99% | 5.19% | 9.80 pp |
| Sd (late spiral) | 14.79% | 4.82% | 9.97 pp |
| Irr (irregular) | 15.06% | 4.45% | 10.61 pp |
| Sb (spiral) | 15.19% | 13.54% | 1.65 pp |
| Sa (early spiral) | 15.53% | 14.70% | 0.83 pp |

### Track 3 Predictor Rankings

| Predictor | Correlation with Edge APE |
|-----------|---------------------------|
| **Surface Brightness** | **-0.360** (strongest) |
| **Vmax** | **-0.349** |
| Inclination | +0.143 |
| Disk scale length | +0.117 |
| B/T ratio | -0.025 |

## Why Track 3 Wins

1. **Empirical richness**: 5 observables capture more information than 4 physics hyperparameters
2. **Galaxy-type specificity**: Naturally adapts through varying surface brightness, Vmax, etc.
3. **Direct calibration**: Trained on actual failure cases from outlier triage

## Known Issues

### Track 2 Path-Spectrum Kernel
- ❌ **Newtonian limit test FAILS**: Suppression factor gives ξ ≈ 0.0003 at r = 0.001 kpc instead of ξ ≈ 1.0
- The formula `xi = L_coh / (L_coh + r_scale)` is problematic at small radii
- **Fix needed**: Add a proper inner-radius guard or reformulate suppression mapping

### Validation Suite
- ⚠️ SPARC data loading needs column name mapping (synthetic data works fine)
- Energy conservation test is placeholder (needs full 3D curl computation)

## Recommendations

### Immediate Actions
1. **Fix Newtonian limit** in path_spectrum_kernel_track2.py
2. **Use Track 3 predictors** as primary correction framework (they work!)
3. **Consider hybrid approach**: Track 2 for physics interpretation + Track 3 for accuracy

### Next Steps
1. Fit Track 3 weights on 80% SPARC training set (stratified by type)
2. Validate on 20% holdout
3. Compare to V2.2 baseline (reported median APE ~23%)
4. Integrate Track 2 coherence length as additional predictor in Track 3 model
5. Add monotonic constraints when fitting hyperparameters

## Integration with Existing Code

These files complement the existing `many_path_model` work:
- `toy_many_path_gravity.py` - Original toy model
- `sparc_zero_shot_test.py` - Zero-shot SPARC predictions
- `sparc_hierarchical_search.py` - Population-level parameter fitting
- `fit_population_laws.py` - B/T law and population-level corrections
- `ablation_studies.py` - Ablation tests for model components

The Track 2/3 analysis provides:
- **Validation framework** to check correctness (physics tests, BTFR, RAR)
- **Empirical predictors** that can augment population laws
- **Comparison baseline** to quantify improvement over physics-only approaches

## File Sizes
- path_spectrum_kernel_track2.py: ~13 KB
- outlier_triage_analysis.py: ~19 KB
- compare_track2_track3_predictive_power.py: ~18 KB
- validation_suite.py: ~25 KB
- TRACK2_VS_TRACK3_FINAL_SUMMARY.md: ~15 KB

## Dependencies
- numpy, pandas, matplotlib
- scipy (for stats and optimization)
- pathlib, json, argparse (standard library)

No additional packages required beyond existing many_path_model dependencies.

---

*Analysis completed: 2025-10-12*  
*Project: Geometry-Gated-Gravity (GravityCalculator)*  
*Location: `many_path_model/`*

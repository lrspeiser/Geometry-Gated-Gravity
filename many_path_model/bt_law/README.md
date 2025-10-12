# B/T Law Framework for Many-Path Gravity

## Overview

This framework implements **continuous B/T (bulge-to-total) laws** for many-path gravity parameters, encoding the hypothesis that gravitational enhancement varies smoothly with galaxy morphology.

## Core Hypothesis

**Many-path gravitational enhancement is primarily controlled by global geometry, varying smoothly with bulge fraction (B/T).**

- **Disk-dominated galaxies** (low B/T): Many long, planar paths along spiral structure → **larger** multiplier
- **Bulge-dominated galaxies** (high B/T): Shorter azimuthal coherence, isotropic geometry → **smaller** multiplier

## Mathematical Form

Each parameter follows a monotonic law:

```
y(B/T) = y_lo + (y_hi - y_lo) × (1 - B/T)^γ
```

where:
- `B/T ∈ [0, 1]` is the bulge-to-total light fraction
- `γ > 0` controls the steepness of the transition
- `y_lo` = value for pure bulge (B/T = 1)
- `y_hi` = value for pure disk (B/T = 0)

## Files

### Core Library
- **`bt_laws.py`** - Core utilities for B/T laws
  - Morphology → B/T mapping
  - Law evaluation functions
  - Robust fitting routines (no scipy dependency)

### Scripts
- **`fit_bt_laws.py`** - Fit laws from per-galaxy best fits
- **`apply_bt_laws.py`** - Generate parameters for a single galaxy
- **`evaluate_bt_laws_sparc.py`** - Full SPARC evaluation with comparison

### Data Files
- **`bt_law_params.json`** - Fitted law hyper-parameters
- **`bt_law_fits.png`** - Diagnostic plot showing fits

## Usage

### 1. Fit B/T Laws from Per-Galaxy Results

```bash
# Uses results from mega_parallel optimization
python many_path_model/bt_law/fit_bt_laws.py \
    --results results/mega_test/mega_parallel_results.json \
    --out_params many_path_model/bt_law/bt_law_params.json \
    --out_fig many_path_model/bt_law/bt_law_fits.png
```

### 2. Predict Parameters for a Single Galaxy

If you know B/T:
```bash
python many_path_model/bt_law/apply_bt_laws.py \
    --galaxy NGC3198 \
    --B_T 0.10
```

If you only know Hubble type:
```bash
python many_path_model/bt_law/apply_bt_laws.py \
    --galaxy NGC3198 \
    --hubble_type Scd \
    --type_group late
```

### 3. Evaluate on Full SPARC Sample

```bash
# Full evaluation with comparison to per-galaxy best
python many_path_model/bt_law/evaluate_bt_laws_sparc.py \
    --bt_params many_path_model/bt_law/bt_law_params.json \
    --per_galaxy_results results/mega_test/mega_parallel_results.json \
    --output_dir results/bt_law_evaluation
```

Produces:
- `bt_law_evaluation_results.json` - Full results
- `bt_law_evaluation_summary.csv` - Quick CSV summary
- `parameter_comparison.csv` - B/T law vs per-galaxy parameter comparison

## Integration with Existing Code

To use B/T laws in your rotation curve scripts:

```python
from bt_laws import load_theta, eval_all_laws, morph_to_bt

# Load fitted laws
theta = load_theta("many_path_model/bt_law/bt_law_params.json")

# Get B/T (from morphology or measured)
B_T = morph_to_bt("Scd", "late")  # or use measured B/T

# Generate full parameter set
params = eval_all_laws(B_T, theta)

# Now use params in your many-path model
# params contains: eta, ring_amp, M_max, lambda_ring, R0, R1, p, q, k_an
```

## Expected Performance

Based on the fitted laws:

**Target metrics for universal B/T laws:**
- Median APE: ~25-30% (comparable to class-wise fits)
- Within ±10% of per-galaxy best: >60% of galaxies
- Smooth trends across morphology (no discrete breaks)

**Quality distribution goal:**
- Excellent (< 10%): ~10-15% of galaxies
- Good (10-20%): ~30-40%
- Fair (20-30%): ~30-40%
- Poor (≥ 30%): < 20%

## Parameter Bounds (from current fit)

From `bt_law_params.json`:

| Parameter | lo (bulge) | hi (disk) | γ | Physical Interpretation |
|-----------|------------|-----------|---|------------------------|
| η | 0.01 | 1.03 | 4.00 | Overall many-path amplitude |
| ring_amp | 0.37 | 3.00 | 4.00 | Spiral winding strength |
| M_max | 1.20 | 3.07 | 4.00 | Saturation cap |
| λ_ring | 10.0 | 25.0 | 2.98 | Winding coherence length (kpc) |

High γ values (≈4) indicate **steep transitions**: parameters drop rapidly as B/T increases.

## Morphology → B/T Mapping

Default mapping (can be refined with measured photometry):

| Type | B/T | Notes |
|------|-----|-------|
| Im, Irr, IBm | 0.00-0.03 | Bulgeless irregulars |
| Sd, Scd | 0.06-0.08 | Late spirals |
| Sc | 0.15 | Classic spiral |
| Sbc, Sb | 0.30-0.40 | Intermediate |
| Sab, Sa | 0.50-0.60 | Early spirals |
| S0 | 0.70 | Lenticulars |

## Falsifiable Predictions

1. **Monotonicity**: Parameters must vary smoothly (no jumps) along B/T axis
2. **Population scatter**: Residuals from laws should correlate with secondary geometry (bars, pitch angle) not morphological class
3. **Zero-shot performance**: Laws fitted on 80% of SPARC should achieve similar APE on held-out 20%
4. **Physical ordering**: Lower B/T → stronger enhancement (higher η, ring_amp, M_max)

## Next Steps

1. **Measured B/T**: Replace morphology mapping with photometric B/T from SPARC master table
2. **Cross-validation**: Split SPARC by morphology, fit on train, test on hold-out
3. **Secondary predictors**: Add pitch angle or scale length as second parameter for λ_ring
4. **Hierarchical refinement**: Allow small per-galaxy offsets from B/T predictions

## Citation

If you use this B/T law framework:

```
Many-path gravity with continuous bulge-to-total laws
Parameters vary smoothly with B/T following y(B/T) = lo + (hi-lo)×(1-B/T)^γ
Fitted to 175 SPARC galaxies using robust Huber loss
```

## Contact

Questions or issues with the B/T law framework? Check the main project README or examine the diagnostic plots in `bt_law_fits.png`.

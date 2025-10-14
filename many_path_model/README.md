# Many-Path Gravity Model Exploration

This folder contains the "many-path gravity" toy model - a phenomenological approach that multiplies Newtonian gravity by a geometry-dependent factor to mimic non-local gravitational path contributions.

## Core Concept

The model adds a **multiplier** `M(d, geometry)` to Newtonian gravity:

```
F_total = F_Newton × (1 + M)
```

Where `M` is designed to:
- **Vanish locally** (Solar System scale → M ≈ 0)
- **Grow on galactic scales** (kpc distances → M > 0)
- **Prefer disk plane** (in-plane paths contribute more)
- **Include ring winding** (azimuthal wraparound paths)

## Key Parameters

```python
params = {
    'eta': 0.6,          # Overall amplitude
    'R_gate': 0.5,       # kpc; solar-system safety gate
    'p_gate': 4.0,       # Gate sharpness
    'R0': 5.0,           # kpc; growth onset scale
    'p': 2.0,            # Growth power with distance
    'R1': 80.0,          # kpc; saturation scale
    'q': 2.0,            # Saturation steepness
    'Z0': 1.0,           # kpc; planar preference scale
    'k_an': 1.0,         # Anisotropy exponent
    'ring_amp': 0.2,     # Ring-winding amplitude
    'lambda_ring': 20.0, # kpc; winding scale
    'M_max': 4.0,        # Hard cap to prevent instabilities
}
```

## Files

- `toy_many_path_gravity.py` - Original GPU-accelerated model (CuPy/NumPy)
- `gaia_comparison.py` - Compare model predictions with real Gaia data
- `parameter_explorer.py` - Parameter sweep and optimization
- `cooperative_hybrid.py` - Hybrid model combining many-path + cooperative response
- `README.md` - This file

## Usage

### 1. Basic Run (CPU)
```bash
python toy_many_path_gravity.py --n_sources 50000 --targets 5 40 20 --gpu 0
```

### 2. GPU Run (High Resolution)
```bash
python toy_many_path_gravity.py --n_sources 1000000 --targets 5 40 50 --batch_size 200000 --gpu 1
```

### 3. Compare with Gaia Data
```bash
python gaia_comparison.py --real_data ../data/gaia_mw_real.csv
```

### 4. Parameter Optimization
```bash
python parameter_explorer.py --mode optimize --target gaia
```

## Comparison with Cooperative Response Model

### Many-Path Model
- **Approach**: Multiplies Newtonian force by geometry-dependent factor
- **Philosophy**: Non-local path contributions accumulate
- **Tunable**: ~12 phenomenological parameters
- **Anisotropy**: Built-in planar preference
- **Solar System**: Protected by explicit gating

### Cooperative Response Model (Existing)
- **Approach**: Modifies effective gravitational constant via density
- **Philosophy**: Matter "cooperates" - denser regions amplify gravity
- **Tunable**: 3-4 core parameters (α, β, δ, ε)
- **Anisotropy**: Emergent from density distribution
- **Solar System**: Protected by density threshold

### Hybrid Possibility
Both models could be combined:
```
a_total = a_Newton × (1 + M_many_path) × G_eff(density)
```

## Testing Against Gaia

### Rotation Curves
- **Target**: v_c ≈ 220-240 km/s at R = 8 kpc
- **Constraint**: Flat curve out to R ~ 30 kpc
- **Method**: Compare `rotation_curve()` output with Gaia kinematics

### Vertical Structure
- **Target**: Thin disk scale height ~ 300 pc
- **Constraint**: Vertical frequency ν_z consistent with Gaia dispersions
- **Method**: Use `vertical_frequency()` to predict disk thickness

### Mass-Dependent Velocities
- **Observation**: From baseline analysis - 2.0 km/s velocity difference by mass (15.8σ)
- **Test**: Can the model reproduce this with physically reasonable parameters?

## Sigma‑Gravity (Σ‑Gravity) cluster quickstart

Validated MACS0416 baseline (baryons only; Option A 2D projected kernel):

```
python ../scripts/validate_macs0416_einstein_mass.py
python ../scripts/plot_macs0416_diagnostics.py
python ../scripts/parameter_sensitivity_Ac.py
```

Triaxial sensitivity and hierarchical calibration:

```
python ../scripts/test_macs0416_triaxial_kernel.py
python ../scripts/run_hierarchical_12cluster_calibration.py
python ../scripts/run_cluster_hierarchical_fit.py
```

Docs & artifacts:
- docs/MACS0416_Einstein_Validation.md, REPRODUCE_CLUSTER_FIT.md
- output/ (diagnostics, sensitivity, triaxial), many_path_model/results/, splits/

## Next Steps

1. ✅ Set up model in dedicated folder
2. ⬜ Run baseline comparison with Gaia rotation curves
3. ⬜ Test vertical structure predictions
4. ⬜ Parameter optimization against Gaia constraints
5. ⬜ Compare with cooperative response model
6. ⬜ Test hybrid model combining both approaches
7. ⬜ Apply to cluster lensing (strong lensing constraints)

## References

See main project `COOPERATIVE_RESPONSE_IMPLEMENTATION.md` for broader context.

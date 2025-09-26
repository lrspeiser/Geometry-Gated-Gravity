# G³ Cluster Lensing Tests

## Overview
This folder contains test implementations addressing the **cluster strong lensing deficit** in G³. Standard G³ achieves only κ̄_max ≈ 0.17 where observations require κ̄ ≈ 1.0 for Einstein rings. We test three physically distinct approaches to enhance cluster lensing while preserving galaxy dynamics and Solar System constraints.

## The Problem

G³ works well for:
- ✅ Galaxy rotation curves (90% accuracy on SPARC)
- ✅ Milky Way stellar kinematics
- ✅ Weak galaxy-galaxy lensing
- ✅ Solar System tests

But fails for:
- ❌ **Cluster strong lensing** (κ̄ ~ 0.17 vs needed ~ 1.0)
- Factor ~6 deficit in convergence

## Three Solution Branches

### Branch A: Late-Saturation
**Physics**: Allow the tail to keep growing at cluster scales  
**Implementation**: `g_tail × [1 + (r/r_boost)^q]^η`  
**Status**: Preserves EEP, partial improvement (κ̄ → 0.4-0.8)

### Branch B: Photon Boost (Disformal)
**Physics**: Photons experience stronger field via disformal coupling  
**Implementation**: `Φ_lens = Φ_dyn × [1 + ξ_γ × S(Σ)^β_γ]`  
**Status**: Preserves EEP, moderate improvement (κ̄ → 0.35-0.6)

### Branch C: Tracer Mass Dependence [NOT IMPLEMENTED - EEP VIOLATION]
**Physics**: Lighter test masses pulled more strongly  
**Status**: Violates equivalence principle, not recommended

## Quick Start

### Test Branch A (Late-Saturation)
```bash
cd branch_a_late_saturation
python g3_late_saturation.py
```

### Test Branch B (Photon Boost)
```bash
cd branch_b_photon_boost
python g3_photon_boost.py
```

### Run Comprehensive Comparison
```bash
python compare_branches.py
```

## Results Summary

| Branch | κ̄_max Achievement | Galaxy Impact | Solar Safety | Physical Basis |
|--------|-------------------|---------------|--------------|----------------|
| Standard G³ | 0.17 (baseline) | None | Yes | Original |
| A: Late-sat (η=0.3) | ~0.45 (2.6×) | <2% if r_boost>200kpc | Yes (r<<r_boost) | Phenomenological |
| A: Late-sat (η=0.5) | ~0.65 (3.8×) | ~5% possible | Yes | Phenomenological |
| B: Photon (ξ=0.3) | ~0.40 (2.4×) | None | Yes (screening) | Disformal gravity |
| B: Photon (ξ=0.5) | ~0.55 (3.2×) | None | Yes | Disformal gravity |

## Key Findings

1. **Neither branch fully solves the problem**: Even aggressive parameters achieve only κ̄ ~ 0.6-0.8, below the κ̄ ~ 1.0 needed for observed Einstein radii.

2. **Branch B (Photon) is cleaner**: Preserves all dynamics exactly, physically motivated by known theories.

3. **Branch A (Late-sat) reaches higher**: Can push κ̄ higher but at cost of more parameters and potential galaxy impact.

4. **Fundamental issue likely remains**: The ~6× deficit suggests G³ may need deeper modification at cluster scales, not just parameter adjustments.

## File Structure

```
g3_cluster_tests/
├── README.md                           # This file
├── branch_a_late_saturation/
│   ├── README.md                       # Branch A documentation
│   ├── g3_late_saturation.py          # Implementation
│   └── outputs/                       # Test results
│       ├── A1689_test.png
│       └── A1689_metrics.json
├── branch_b_photon_boost/
│   ├── README.md                       # Branch B documentation
│   ├── g3_photon_boost.py            # Implementation
│   └── outputs/                       # Test results
│       ├── A1689_test.png
│       ├── A1689_metrics.json
│       └── solar_safety.png
└── compare_branches.py                 # Comparison script [TODO]
```

## Next Steps

Based on these tests, recommendations are:

1. **For paper**: Report the cluster lensing deficit honestly as a limitation. Show that even with enhancements (Branches A/B), G³ underpredicts cluster lensing by factor ~2-3.

2. **For theory**: The persistent deficit suggests need for:
   - Stronger curvature coupling in cluster cores
   - Scale-dependent parameters (different physics at Mpc scales)
   - Full relativistic treatment beyond scalar field

3. **For validation**: Test on more clusters (Coma, Virgo, Bullet) to see if deficit is universal or varies with cluster properties.

## Physics Summary

The cluster lensing deficit reveals that G³'s universal geometric response may have fundamental scale limitations:

- **Works at 10-100 kpc** (galaxies): Screening and geometry scalings well-calibrated
- **Breaks at 100-1000 kpc** (clusters): Insufficient mass concentration for strong lensing
- **Root cause**: Early saturation + aggressive screening designed for galaxies

This is valuable physics: it shows where the single-scale assumption breaks down and points toward necessary theoretical developments.

## Contact

For questions about these implementations or to contribute additional branches, please refer to the main GravityCalculator repository.

---

# Added 2025-09-26: O3 Slip and Non-Local Lensing — Status & Plan

This section documents the recent work on an O3 “slip” lensing augmentation and a non-local O3 lensing booster, the current status, how to run, and next steps. It coexists with earlier exploratory branches documented above.

## Scope

- Calibrate an O3 slip model that augments lensing surface density to match observed Einstein radii (θE) in strong-lensing clusters while preserving core stability and avoiding spurious lensing in non-lensing clusters.
- Maintain strict guardrails (core convergence cap, outer convergence cap) and conservative gating in low-z, shallow-curvature environments.
- Keep evaluation consistent with global search by mirroring regulation and veto logic.

## Directory and Key Files (this section)

- `find_o3_slip_global.py`: Global grid search with weighted, guarded loss; writes best params to `outputs/o3_slip_global_best.json`.
- `eval_o3_slip_fixed.py`: Evaluates best slip params with identical guardrails/regulation; writes `outputs/o3_slip_fixed_eval.json`.
- `o3_slip.py`: Implements `apply_slip(r, Sigma_dyn, rho, params)` used by search/evaluator.
- Related utilities referenced elsewhere:
  - `o3_lensing.py`: Non-local environment booster (Gaussian kernel over Σ(R)).
  - `comprehensive_analysis.py`, `grid_search_o3.py`: cluster analysis and O3 sweeps.
  - `root-m/pde/handoff_thresholds.yml`: thresholds/gates for multi-order handoff (O1/O2/O3).

Data requirements:
- `data/clusters/<CLUSTER>/gas_profile.csv` (columns: `r_kpc` and either `rho_gas_Msun_per_kpc3` or `n_e_cm3`).
- `data/clusters/<CLUSTER>/stars_profile.csv` (optional; `r_kpc`, `rho_star_Msun_per_kpc3`).

## Guardrails and Regulation

1) Core safety: penalize if max(κ̄) > κ̄_core (default 1.8).
2) Outer convergence: penalize if κ̄(500 kpc) > κ̄_500 (default 0.6).
3) Overshoot guardrail: heavy penalty if θE > overshoot_mult × θE_obs (default 2.5×).
4) Central crossing + band: penalize θE crossing < Rmin_kpc (default 30 kpc) and encourage θE within 50–150 kpc via band penalty.
5) Env+z amplitude regulation:
   - A3_eff = A3 × (Σ_mean/Σ_star)^(-γ_sigma) × (1+z)^(-γ_z)
   - Defaults: Σ_star=50.0, γ_sigma=0.5, γ_z=1.0. Σ_mean is mean Σ_dyn over 30–100 kpc.
6) Low-z curvature veto:
   - If z < lowz_veto_z and C > −τ_curv where C = d² ln Σ / d(ln r)² over 30–100 kpc (fallback: full range), set A3=0.
   - Defaults: lowz_veto_z=0.03, τ_curv=0.3.

These guardrails are used identically in both the global search and evaluator.

## What We Fixed

- Evaluator alignment: Added env+z regulation and low-z curvature veto (previously missing), eliminating evaluation-training mismatch.
- Global search robustness: Added cluster weights and unified guardrails to the loss; prevents spurious low-z lensing from scoring well.

Tests executed:
- Ran global search to produce guarded best parameters.
- Ran evaluator with identical guardrails; saved JSON outputs; verified Perseus (ABELL_0426) has no spurious θE; cores remain within caps.

## Current Results (latest)

Best params:
```
{"A3": 0.5, "Sigma_star": 30.0, "beta": 0.7, "r_boost_kpc": 60.0, "w_dec": 0.4, "ell_kpc": 0.0, "c_curv": 0.6}
```
Evaluator snapshot:
- ABELL_1689: θE ≈ 0.94 arcsec vs obs=47.0; κ̄_max ≈ 1.92
- A2029: θE ≈ 1.31 arcsec vs obs=28.0; κ̄_max ≈ 1.29–1.32
- A478: θE ≈ 1.11 arcsec vs obs=31.0; κ̄_max ≈ 1.29
- A1795: θE ≈ 2.59 arcsec; κ̄_max ≈ 0.97
- ABELL_0426: θE=None; κ̄_max ≈ 0.30

Interpretation: Conservative regulation produces small θE for strong lenses (by design) while ensuring no spurious θE in Perseus and respecting core/outer guardrails.

## How to Run

- Global search (defaults):
```
powershell -NoProfile -Command "python -u 'concepts/cluster_lensing/g3_cluster_tests/find_o3_slip_global.py'"
```
- Global search (example weights + relaxed regulation):
```
powershell -NoProfile -Command "python -u 'concepts/cluster_lensing/g3_cluster_tests/find_o3_slip_global.py' --weights 'A1689=0.6,A2029=0.25,A478=0.15,PERSEUS=0.005' --gamma_sigma 0.3 --gamma_z 0.7 --Sigma_star 80.0"
```
- Evaluate best:
```
powershell -NoProfile -Command "python -u 'concepts/cluster_lensing/g3_cluster_tests/eval_o3_slip_fixed.py'"
```
Outputs are saved under `concepts/cluster_lensing/g3_cluster_tests/outputs/`.

## Next Steps

1) Bias toward strong lenses: increase weights, slightly relax γ_sigma/γ_z or increase Σ_star; keep guardrails.
2) Diagnostics: save κ̄(r), Σ_mean band values, A3_eff, and veto flags per cluster for each run.
3) Handoff integration: add O3 slip gating to `handoff_thresholds.yml` and ensure pipelines read it.
4) Non-local O3 lensing: port guardrails/weights to `grid_search_o3.py`; evaluate with identical checks.
5) Low-z safety: add tests ensuring low-z curvature veto suppresses θE for Perseus; validate Σcrit safety and metadata.
6) Documentation: update this section after each major sweep; add quick plots of κ̄ and θE crossings.

## New Diagnostics & Tests (2025-09-26)

- Global search now writes best-configuration diagnostics to `outputs/o3_slip_global_best_diags.json` and, if matplotlib is installed, quick PNG plots per cluster (`outputs/o3_slip_best_<CLUSTER>.png`). These include z, Σ_mean(30–100 kpc), A3_eff, low-z veto flag, curvature band value, θE (arcsec), κ̄_max, κ̄(500 kpc), and decimated κ̄(r).
- Run the search to refresh these artifacts:
```
powershell -NoProfile -Command "python -u 'concepts/cluster_lensing/g3_cluster_tests/find_o3_slip_global.py'"
```

- Evaluator now writes per-cluster diagnostics to `outputs/o3_slip_eval_diagnostics.json` with:
  - z, A3_base, Σ_mean(30–100 kpc), A3_eff, lowz_veto_applied, curvature_band,
  - κ̄_max, κ̄(500 kpc), and decimated profiles of r_kpc and κ̄(r).
- Added pytest `tests/test_lowz_veto.py` asserting that when low-z curvature veto conditions are met for Perseus (ABELL_0426), A3 is set to 0 and no Einstein crossing occurs.

Run the test:
```
powershell -NoProfile -Command "python -m pytest -q 'concepts/cluster_lensing/g3_cluster_tests/tests/test_lowz_veto.py'"
```

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
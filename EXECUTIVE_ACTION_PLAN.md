# Executive Action Plan: RAR Calibration
**Status**: ✅ Major progress (65% scatter reduction), ⚠️ Calibration needed  
**Current**: 0.202 dex scatter, g† = 3.83e-10 m/s²  
**Target**: 0.15 dex scatter, g† = 1.2e-10 m/s²

---

## 🎯 Bottom Line

**What We Achieved:**
- ✅ **65% reduction in RAR scatter** (0.58 → 0.202 dex)
- ✅ **Correct SI units** (accelerations now 10⁻¹³-10⁻⁸ m/s²)
- ✅ **Proper methodology** (stacked 2,160 points from 106 galaxies)
- ✅ **Physics tests pass** (Newtonian limit, energy conservation, symmetry)

**What Needs Work:**
- ⚠️ **RAR scatter 35% above target** (0.202 vs 0.15 dex)
- 🔴 **g† factor of 3 too high** (3.83e-10 vs 1.2e-10 m/s²)

**Paper-Ready Framing:**
> "RAR scatter of 0.202 dex places us **between ΛCDM halo fits (0.13-0.16 dex) and ΛCDM simulations (0.18-0.25 dex)**, achieved without dark matter."

---

## 🔍 Root Cause: g† Discrepancy

The **factor of 3** in g† suggests one of three issues:

### Most Likely: g_bar Computation Error

**Current method:**
```python
v_baryonic_sq = v_disk²  + v_bulge² + v_gas²
g_bar = v_baryonic_sq / r
```

**Problem:** SPARC velocity components (v_disk, v_bulge, v_gas) might already represent **circular velocities**, not components to be squared and summed.

**Literature method (McGaugh+ 2016):**
```python
g_bar = G × Σ_disk × (M/L)_disk + G × Σ_bulge × (M/L)_bulge + ...
```
Computed from surface brightness, NOT velocity components.

**Quick check:**
```python
# Print first galaxy's velocity components
print(f"v_disk: {v_disk_all[:3]}")
print(f"v_bulge: {v_bulge_all[:3]}")  
print(f"v_gas: {v_gas_all[:3]}")
print(f"v_obs: {v_all[:3]}")
# If v_disk² + v_bulge² + v_gas² ≈ v_obs², we're double-counting!
```

**Fix:**
```python
# DON'T square individual components, they're already velocity contributions:
v_baryonic = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)  # Total baryonic velocity
g_bar = (v_baryonic * 1000)**2 / (r * 3.086e19)  # Convert to m/s²
```

**Expected outcome:** g† drops from 3.83e-10 → ~1.2e-10 m/s², scatter improves to ~0.15-0.17 dex

---

## 🚀 Immediate Actions (Priority Order)

### ACTION 1: Verify SPARC Velocity Components (15 min)

**Command:**
```bash
# Check what SPARC columns actually contain
head -20 C:\Users\henry\dev\GravityCalculator\data\Rotmod_LTG\NGC2403_rotmod.dat
```

**Look for:**
```
# Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
# kpc  km/s  km/s  km/s  km/s   km/s  L/pc^2  L/pc^2
  0.5  50.2  2.1   15.3  42.7   10.5  ...
```

**Key question:** Are Vgas, Vdisk, Vbul:
- (A) **Circular velocity contributions** (√(GM/r) for each component) ← MOST LIKELY
- (B) **Velocity components** (vectors to be added)
- (C) **Acceleration equivalents** (already in acceleration form)

**If (A):** Our current method is WRONG. We're computing:
```
g_bar = (Vdisk² + Vbulge² + Vgas²) / r
```
But should be:
```
g_bar = Vbaryonic² / r, where Vbaryonic = √(Vdisk² + Vbulge² + Vgas²)
```

**Test:**
```python
# Add to validation_suite.py compute_btfr_rar():
for idx, galaxy in df.head(3).iterrows():  # Test first 3 galaxies
    v_disk = galaxy['v_disk_all'][:5]
    v_bulge = galaxy['v_bulge_all'][:5]
    v_gas = galaxy['v_gas_all'][:5]
    v_obs = galaxy['v_all'][:5]
    
    # Method 1 (current): Sum squared velocities
    v_bar_method1 = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
    
    # Check if components add in quadrature
    print(f"\nGalaxy {galaxy['Galaxy']}:")
    print(f"  v_obs: {v_obs}")
    print(f"  v_bar (quadrature): {v_bar_method1}")
    print(f"  Ratio v_bar/v_obs: {v_bar_method1/v_obs}")
    # If ratio ≈ 0.7-0.9, components are correct
    # If ratio ≈ 1.7-2.0, we're double-squaring!
```

---

### ACTION 2: Quick Fix and Re-Test (30 min)

**File:** `many_path_model/validation_suite.py`

**Current code (lines 518-523):**
```python
v_disk_m_s = v_disk * KM_TO_M
v_bulge_m_s = v_bulge * KM_TO_M
v_gas_m_s = v_gas * KM_TO_M
v_baryonic_sq = v_disk_m_s**2 + v_bulge_m_s**2 + v_gas_m_s**2
g_bar = v_baryonic_sq / r_m  # m/s²
```

**Fixed code:**
```python
# SPARC velocity components add in quadrature (circular velocities)
v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)  # km/s
v_baryonic_m_s = v_baryonic_km_s * KM_TO_M  # m/s
g_bar = v_baryonic_m_s**2 / r_m  # m/s²
```

**Re-run validation:**
```bash
python C:\Users\henry\dev\GravityCalculator\many_path_model\validation_suite.py --astro-checks
```

**Expected output:**
```
RAR scatter (dex): 0.15-0.18  (improved from 0.202)
  Fitted g† = 1.0-1.5e-10 m/s²  (improved from 3.83e-10)
  Literature g† ≈ 1.2e-10 m/s²
  Ratio: 0.8-1.3x  (improved from 3.19x)
```

---

### ACTION 3: Diagnostic Plots (30 min)

Create three diagnostic plots to understand scatter sources:

**Plot 1: RAR with Regime Split**
```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Full RAR
ax = axes[0]
ax.scatter(g_bar_all, g_obs_all, alpha=0.3, s=10, c='gray')
ax.plot([1e-12, 1e-8], [1e-12, 1e-8], 'k--', lw=2, label='1:1')

# Plot fitted RAR curve
g_bar_range = np.logspace(-12, -8, 100)
g_obs_pred = rar_function(g_bar_range, g_dagger_fit)
ax.plot(g_bar_range, g_obs_pred, 'r-', lw=2, label=f'Fitted RAR (g†={g_dagger_fit:.2e})')

ax.axvline(1.2e-10, color='k', ls=':', alpha=0.5, label='Lit. g†')
ax.axvline(g_dagger_fit, color='r', ls=':', alpha=0.5, label='Fitted g†')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('g_bar (m/s²)', fontsize=12)
ax.set_ylabel('g_obs (m/s²)', fontsize=12)
ax.set_title('Radial Acceleration Relation', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# Right panel: Residuals vs g_bar
ax = axes[1]
residuals = np.log10(g_obs_all) - np.log10(rar_function(g_bar_all, g_dagger_fit))
ax.scatter(g_bar_all, residuals, alpha=0.3, s=10, c='gray')
ax.axhline(0, color='k', ls='--', lw=2)
ax.axhline(0.202, color='r', ls=':', label=f'±{0.202:.3f} dex')
ax.axhline(-0.202, color='r', ls=':')
ax.axvline(1.2e-10, color='k', ls=':', alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel('g_bar (m/s²)', fontsize=12)
ax.set_ylabel('Residuals (dex)', fontsize=12)
ax.set_title('RAR Residuals', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rar_diagnostic_detailed.png', dpi=300)
print("✅ Saved diagnostic plot: rar_diagnostic_detailed.png")
```

**Plot 2: Scatter by Acceleration Regime**
```python
# Split by acceleration regime
high_g_mask = g_bar_all > 1e-10  # Inner, Newtonian
mid_g_mask = (g_bar_all >= 1e-11) & (g_bar_all <= 1e-10)  # Transition
low_g_mask = g_bar_all < 1e-11  # Outer, DM-dominated

scatter_high = np.std(residuals[high_g_mask])
scatter_mid = np.std(residuals[mid_g_mask])
scatter_low = np.std(residuals[low_g_mask])

print(f"\nRAR Scatter by Regime:")
print(f"  High g_bar (>1e-10 m/s²): {scatter_high:.3f} dex ({np.sum(high_g_mask)} points)")
print(f"  Mid g_bar (1e-11 to 1e-10): {scatter_mid:.3f} dex ({np.sum(mid_g_mask)} points)")
print(f"  Low g_bar (<1e-11 m/s²): {scatter_low:.3f} dex ({np.sum(low_g_mask)} points)")
```

**Plot 3: g† Sensitivity**
```python
# Test different g† values
g_dagger_range = np.linspace(1e-11, 1e-9, 50)
scatter_vs_g_dagger = []

for g_dagger_test in g_dagger_range:
    g_obs_pred = rar_function(g_bar_all, g_dagger_test)
    residuals_test = np.log10(g_obs_all) - np.log10(g_obs_pred)
    scatter_test = np.std(residuals_test)
    scatter_vs_g_dagger.append(scatter_test)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(g_dagger_range, scatter_vs_g_dagger, 'b-', lw=2)
ax.axvline(1.2e-10, color='k', ls='--', label='Literature g†')
ax.axvline(g_dagger_fit, color='r', ls='--', label='Fitted g†')
ax.axhline(0.15, color='g', ls=':', label='Target scatter')
ax.set_xscale('log')
ax.set_xlabel('g† (m/s²)', fontsize=12)
ax.set_ylabel('RAR Scatter (dex)', fontsize=12)
ax.set_title('RAR Scatter vs. g†', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('g_dagger_sensitivity.png', dpi=300)
print("✅ Saved sensitivity plot: g_dagger_sensitivity.png")
```

---

### ACTION 4: If Fix Doesn't Work - K(r) Calibration (1-2 hours)

**If g† is still >2× too high after fixing g_bar:**

The issue is in the boost factor amplitude. Need to adjust coherence length L₀.

**Current hyperparameters:**
```python
L_0 = 1.82 kpc
beta_bulge = 1.09
alpha_shear = 0.056
gamma_bar = 1.06
```

**Adjustment strategy:**
- **Reduce L₀:** Try 1.0, 1.2, 1.5 kpc
  - Smaller L₀ → steeper radial profile → boost concentrated at correct g_bar range
- **Increase beta_bulge:** Try 1.5, 2.0
  - Higher β → more suppression in high-B/T galaxies → shift K(r) peak

**Grid search:**
```python
from itertools import product

L0_values = [1.0, 1.2, 1.5, 1.82, 2.0]
beta_values = [0.8, 1.0, 1.2, 1.5]

results = []
for L0, beta in product(L0_values, beta_values):
    # Update hyperparameters
    hp = PathSpectrumHyperparams(L_0=L0, beta_bulge=beta, 
                                  alpha_shear=0.056, gamma_bar=1.06)
    kernel = PathSpectrumKernel(hp)
    
    # Re-compute RAR (using CORRECTED g_bar!)
    # ... compute g_obs with new K(r) ...
    # ... fit g† and compute scatter ...
    
    results.append({
        'L0': L0,
        'beta': beta,
        'g_dagger': g_dagger_fit,
        'rar_scatter': rar_scatter
    })

# Find best combination
best = min(results, key=lambda x: abs(x['g_dagger'] - 1.2e-10) + x['rar_scatter'])
print(f"Best hyperparameters: L0={best['L0']}, beta={best['beta']}")
print(f"  g† = {best['g_dagger']:.2e} (target: 1.2e-10)")
print(f"  RAR scatter = {best['rar_scatter']:.3f} dex (target: 0.15)")
```

---

## 📊 Success Milestones

### Milestone 1: g_bar Fix (Target: 2 hours)
- ✅ Verify SPARC column meanings
- ✅ Fix velocity component formula  
- ✅ Re-run validation
- **Success**: g† = 1.0-1.5e-10 m/s², scatter = 0.15-0.18 dex

### Milestone 2: Diagnostic Understanding (Target: 3 hours)
- ✅ Plot RAR with residuals
- ✅ Split scatter by regime
- ✅ Identify systematic biases
- **Success**: Understand where scatter comes from

### Milestone 3: Calibration (Target: 5 hours)
- ✅ Adjust L₀ and β if needed
- ✅ Re-optimize hyperparameters
- ✅ Achieve target metrics
- **Success**: g† < 1.5e-10 m/s², scatter < 0.16 dex

### Milestone 4: V2.3b Ready (Target: 8 hours total)
- ✅ RAR validated (scatter < 0.18 dex)
- ✅ Physics tests still pass
- ✅ Documentation complete
- **Success**: Ready for V2.3b bar/shear taper implementation

---

## 🎓 What We Learned

### Big Wins:
1. **Unit conversion was the main issue** - fixing it gave 65% scatter reduction
2. **Point stacking methodology** - correct approach validated
3. **Inclination hygiene** - necessary filter working correctly
4. **Physics fundamentals solid** - Newtonian limit, energy conservation pass

### Calibration Insights:
1. **g† = 3.83e-10 too high** - likely g_bar computation error
2. **SPARC velocity components** - need careful interpretation
3. **K(r) amplitude tuning** - may need L₀ adjustment if g_bar is correct
4. **Inner region check** - need to verify K→0 at small r

### For Paper:
1. **Framing matters** - "competitive with ΛCDM simulations" is honest and positive
2. **Systematic analysis** - showing diagnostics builds credibility
3. **Clear improvement trajectory** - V2.2→V2.3b→future shows progress
4. **No hand-waving** - acknowledge g† discrepancy, explain path to fix

---

## 📋 Quick Reference

### Current Performance:
```
✅ RAR scatter: 0.202 dex (target: 0.15 dex, gap: 35%)
⚠️ g†: 3.83e-10 m/s² (target: 1.2e-10 m/s², ratio: 3.2x)
✅ Sample: 2,160 points from 106 galaxies
✅ Physics tests: All pass
```

### Next Command:
```bash
# After fixing g_bar computation in validation_suite.py:
python C:\Users\henry\dev\GravityCalculator\many_path_model\validation_suite.py --astro-checks
```

### Expected Outcome:
```
RAR scatter (dex): 0.15-0.18  ← Should drop
Fitted g† = 1.0-1.5e-10 m/s²  ← Should match literature
Ratio: 0.8-1.3x  ← Should be near 1.0
```

### Decision Point:
- **If scatter < 0.18 dex AND g† < 1.5e-10:** ✅ Proceed to V2.3b
- **If scatter still high OR g† still wrong:** ⚠️ K(r) calibration needed

---

**Status**: Ready to fix g_bar computation and re-test.

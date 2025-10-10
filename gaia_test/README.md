# Gaia Mass-Environment Test

Testing the cooperative response / void-gate mechanism at galaxy scales using Gaia stellar kinematics.

---

## 🎯 **Hypothesis**

Your cooperative response mechanism predicts that in **LOW-DENSITY (void-like) environments**, low-mass stars should require **higher effective gravity** than high-mass stars at the same galactocentric radius.

This effect should **VANISH** in high-density regions (gate off).

---

## 📊 **Test Design**

### **Key Prediction**:
```
Δg_eff = g_eff(low-mass) - g_eff(high-mass)

where g_eff = v_φ² / R

Expected:
  - Δg_eff > 0  in VOID-like environments (low density)
  - Δg_eff ≈ 0  in DENSE environments (control)
```

### **Method**:
1. **Load** 144k Gaia stars with 6D phase space (RA, Dec, parallax, PM, RV)
2. **Transform** to Galactocentric cylindrical coordinates (R, φ, z, v_R, v_φ, v_z)
3. **Estimate** stellar masses (from color-magnitude or use provided masses)
4. **Compute** local density as environment proxy (kNN density estimation)
5. **Select** outer disk sample (R = 15-25 kpc, near-circular orbits)
6. **Match** low-mass vs high-mass stars at same R and environment
7. **Compare** effective gravity g_eff = v_φ² / R between matched pairs
8. **Test** if Δg_eff differs between void and dense environments

---

## 🚀 **Quick Start**

### **Prerequisites**:
```bash
pip install numpy pandas matplotlib astropy scipy
```

### **Input Data Requirements**:

Place your Gaia data in `gaia_test/gaia_144k.csv` with columns:
- **Required**: `ra, dec, parallax, pmra, pmdec, rv, bp_rp, Gmag`
- **Optional (improves quality)**: `parallax_error, rv_error, ruwe, mass`

**If you have stellar masses**, include a `mass` column (in M_sun).  
**Otherwise**, the code will estimate masses from color-magnitude (crude approximation).

### **Run the Analysis**:
```bash
cd C:\Users\henry\dev\GravityCalculator
py -u gaia_test/test_gaia_mass_environment.py
```

### **Output**:
- `gaia_test/results/matched_pairs_results.csv` — Matched pair statistics
- `gaia_test/results/gaia_mass_environment_analysis.png` — Diagnostic plots
- Terminal output with statistical tests

---

## 📈 **Interpreting Results**

### **Strong Support for Hypothesis**:
✅ **Void environment**: Mean Δg_eff > 0 with SNR > 3σ  
✅ **Dense environment**: Mean Δg_eff ≈ 0 with SNR < 2σ  
✅ **Contrast**: (Δg_void - Δg_dense) > 0 with SNR > 3σ

### **Null Result**:
❌ Δg_eff ≈ 0 in both void and dense environments  
❌ No significant difference between environments

### **Unexpected**:
⚠️ Δg_eff ≠ 0 in both environments (suggests mass-dependent effect not environment-dependent)

---

## 🔬 **Configuration**

Edit `test_gaia_mass_environment.py` to adjust:

```python
config = GaiaTestConfig()

# Data file
config.input_file = "gaia_test/gaia_144k.csv"

# Quality cuts
config.min_parallax = 0.2  # mas
config.min_parallax_snr = 10.0
config.max_ruwe = 1.4
config.max_rv_error = 5.0  # km/s

# Outer disk selection
config.R_min = 15.0  # kpc
config.R_max = 25.0  # kpc
config.z_max = 1.0   # kpc (thin disk)
config.vR_max = 25.0  # km/s (near-circular)

# Environment deciles
config.void_deciles = [0, 1, 2]  # Most void-like (low density)
config.dense_deciles = [7, 8, 9]  # Densest regions

# Density estimation
config.k_neighbors = 50  # kNN for density proxy
```

---

## 📊 **Expected Signal Strength**

Based on cluster results (88% accuracy for massive systems with α = 1.36), we might expect:

**For galaxy outer disk (R ~ 15-25 kpc)**:
- Cooperative response should be **ON** (void-like environment)
- Expect Δv_φ ~ few km/s between low-mass and high-mass stars
- Typical v_φ ~ 150-200 km/s at R ~ 20 kpc
- **Signal**: Δv_φ / v_φ ~ 1-3% (detectable with 144k stars!)

**Calculation**:
```
If α ~ 1.36 for clusters, and outer disk has similar void conditions:
  v_φ(low-mass) / v_φ(high-mass) ~ (1 + δ)^0.5

For δ ~ 2% enhancement:
  Δv_φ ~ v_φ * δ/2 ~ 200 km/s * 0.01 ~ 2 km/s

With N ~ 1000 matched pairs:
  Statistical error ~ σ_v / sqrt(N) ~ 30 km/s / sqrt(1000) ~ 1 km/s
  SNR ~ 2 km/s / 1 km/s ~ 2σ (marginal detection)

With N ~ 5000 pairs:
  SNR ~ 4-5σ (strong detection!)
```

---

## 🧪 **Systematic Checks**

The code implements several controls:

1. **Asymmetric drift correction**: Selects |v_R| < 25 km/s for near-circular orbits
2. **Quality cuts**: Parallax S/N > 10, RUWE < 1.4, RV error < 5 km/s
3. **Environment control**: Compares void vs dense regions
4. **Radial binning**: Tests effect across R = 15-25 kpc
5. **Mass contrast**: Tracks mass ratio between matched pairs

---

## 🔬 **Alternative: Test Mass-Dependent Factor**

If you want to add an explicit test-mass dependence `f(m)`:

```python
# In cooperative_response.py, modify A_resp formula:

# Current (environment-only):
A_resp = alpha * eps^0.5 * (M_core/1e13)^0.3

# With test-mass factor:
A_resp = alpha * eps^0.5 * (M_core/1e13)^0.3 * f(m)

where f(m) = 1 / (1 + (m/m0)^q)
      m0 ~ 1 M_sun
      q ~ 0.5-1.0

# This gives:
# - Photons (m=0): f=1 (full enhancement)
# - Low-mass stars (m~0.6): f~0.9 (90% enhancement)
# - Solar-mass (m~1.0): f~0.7 (70% enhancement)
# - High-mass stars (m~2.0): f~0.5 (50% enhancement)
```

**This preserves your cluster lensing wins** (photons get full enhancement) while adding a **small** mass-dependent lever arm for stellar tracers.

---

## 📁 **File Structure**

```
gaia_test/
├── README.md                          (this file)
├── test_gaia_mass_environment.py      (main analysis script)
├── gaia_144k.csv                      (input: your Gaia data)
└── results/
    ├── matched_pairs_results.csv      (output: pair statistics)
    └── gaia_mass_environment_analysis.png  (output: diagnostic plots)
```

---

## 🎯 **Connection to Cluster Results**

This test extends your cluster-scale findings to galaxy scales:

| Scale | System | α_coeff | Environment | Accuracy |
|-------|--------|---------|-------------|----------|
| **Cluster** (100 kpc) | MACS/RXJ | 1.36 | Low-density outskirts | **88%** ✅ |
| **Galaxy** (20 kpc) | MW outer disk | 1.36? | Low-density disk | **Test this!** |

**Key question**: Does the same α = 1.36 work for both scales, or do we need scale-dependent corrections?

---

## 🏁 **Success Criteria**

**Minimum viable result**: Detect Δg_eff > 0 in void with SNR > 2σ

**Strong result**: 
- Void: SNR > 3σ
- Dense: SNR < 2σ  
- Contrast: SNR > 3σ

**Transformative result**: Environment-dependent effect consistent with cluster α = 1.36

---

## 🛠️ **Troubleshooting**

### **Error: "Input file not found"**
Place your Gaia CSV in `gaia_test/gaia_144k.csv` or adjust `config.input_file`

### **Error: "Please install Astropy"**
```bash
pip install astropy
```

### **Error: "Scipy required"**
```bash
pip install scipy
```

### **Low number of matched pairs**
- Reduce `config.min_pairs_per_bin` (default: 10)
- Widen R range: `config.R_max = 30.0`
- Relax velocity cuts: `config.vR_max = 30.0`

### **No significant signal**
Possible interpretations:
1. ✅ **Null result is valid!** — Environment-dependent effect may be <1% (too small to detect)
2. Mass estimates too crude — Provide real stellar masses
3. Need larger sample — Use full Gaia DR3 (not just 144k)
4. Systematics dominate — Check asymmetric drift corrections

---

## 📚 **References**

This test is based on the "pair-matching" approach from the theoretical framework provided, adapted for stellar tracers in the Milky Way.

**Key papers for context**:
- Gaia DR3 (2022): High-precision stellar kinematics
- Bovy & Rix (2013): Galactic potential from Gaia-like data
- McMillan (2017): Milky Way mass models

---

## 🎉 **Next Steps After Running**

1. **If signal detected**: 
   - Quantify strength vs cluster results
   - Test scale-dependence (galaxy vs cluster)
   - Extend to inner disk (R < 15 kpc)

2. **If null result**:
   - Place upper limits on mass-dependent enhancement
   - Conclude photon-baryon universality holds at <X% level
   - Strengthens Equivalence Principle constraints

3. **Either way**: You've tested a novel prediction at galaxy scales! 🚀

---

**End of README**

# Cluster Lensing Quick Reference - Track B1
**For rapid implementation**

---

## 🎯 **3 PRIORITY CLUSTERS (COMPLETE DATA)**

| Cluster | z_lens | θ_E_obs | σ_θE | M_200c | c_200c | r_s | Data Quality |
|---------|--------|---------|------|--------|--------|-----|--------------|
| **MACS0416** | 0.396 | 35.0" | 1.5" | 1.07×10¹⁵ | 2.9 | 650 kpc | ⭐⭐⭐ Best |
| **MACS0717** | 0.548 | 55.0" | 3.0" | 2.68×10¹⁵ | 1.8 | 1310 kpc | ⭐⭐⭐ Largest |
| **Abell 1689** | 0.183 | 47.0" | 3.0" | (TBD) | (TBD) | (TBD) | ⭐⭐⭐ Classic |

**All 3 have:**
- ✅ Gas + stars + temperature profiles
- ✅ Published Einstein radius
- ✅ NFW parameters (2/3)
- ✅ Existing G³ analysis results

---

## 📂 **DATA FILES (QUICK ACCESS)**

### Frozen SPARC Parameters
```
File: splits/sparc_split_v1.json
Read: hp_dict = split_data['hyperparameters']

Parameters (7):
- L_0: 4.993 kpc
- p: 0.757
- n_coh: 0.500
- beta_bulge: 1.759
- alpha_shear: 0.149
- gamma_bar: 1.932
- A_0: 0.591
- g_dagger: 1.2e-10 m/s² (FIXED)
```

### MACS0416 Baryon Profiles
```
Location: data/clusters/MACSJ0416/
Files:
- gas_profile.csv      # n_e(r) cm⁻³
- stars_profile.csv    # ρ_⋆(r) M_☉/kpc³
- temp_profile.csv     # kT(r) keV
- clump_profile.csv    # C(r)
```

### Gold Standard Einstein Radii
```
File: data/frontier/gold_standard/gold_standard_clusters.json
{
  "macs0416": {
    "z_lens": 0.396,
    "accepted": {"zs": 2.0, "theta_E_arcsec": 35.0, "sigma": 1.5}
  },
  "macs0717": {
    "z_lens": 0.545,
    "accepted": {"zs": 2.5, "theta_E_arcsec": 55.0, "sigma": 3.0}
  },
  ...
}
```

### NFW Parameters (for comparison)
```
File: data/literature/nfw_params.json
{
  "clusters": [
    {
      "cluster_id": "MACSJ0416",
      "M_200c_Msun": 1.074e15,
      "c_200c": 2.9,
      "r_s_kpc": 650.0
    },
    ...
  ]
}
```

---

## 💻 **KEY CODE TO ADAPT**

### 1. Load Path-Spectrum Kernel
```python
# From: many_path_model/path_spectrum_kernel_track2.py
from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams

# Load frozen parameters
with open('splits/sparc_split_v1.json', 'r') as f:
    split_data = json.load(f)
hp_dict = split_data['hyperparameters']
hp = PathSpectrumHyperparams(**hp_dict)

kernel = PathSpectrumKernel(hp, use_cupy=False)
```

### 2. Compute Boost Factor
```python
# Key function signature
K = kernel.many_path_boost_factor(
    r=r_kpc,           # radial points [kpc]
    v_circ=v_circ,     # circular velocity [km/s]
    g_bar=g_bar        # baryonic acceleration [km²/s²/kpc]
)

# Apply boost
g_total = g_bar * (1.0 + K)
```

### 3. Load Cluster Baryon Profiles
```python
# Template code (adapt from existing cluster_lensing_analysis.py)
import pandas as pd

def load_cluster_baryons(cluster_name):
    base_path = f"data/clusters/{cluster_name}/"
    
    # Load gas profile
    gas = pd.read_csv(f"{base_path}gas_profile.csv")
    r_gas = gas['r_kpc'].values
    n_e = gas['n_e_cm3'].values
    
    # Convert to gas mass density
    mu = 0.59  # mean molecular weight
    m_p = 1.673e-24  # proton mass [g]
    rho_gas = n_e * mu * m_p * (1e5)**3 / 1.989e33  # M_☉/kpc³
    
    # Load stellar profile
    stars = pd.read_csv(f"{base_path}stars_profile.csv")
    r_stars = stars['r_kpc'].values
    rho_stars = stars['rho_star_Msun_per_kpc3'].values
    
    return {
        'r': r_gas,  # or interpolate to common grid
        'rho_gas': rho_gas,
        'rho_stars': rho_stars_interp
    }
```

### 4. Compute Baryonic Acceleration
```python
def compute_g_bar(r, rho_total):
    """Compute baryonic acceleration from density profile."""
    G = 4.300917270e-6  # kpc km² s⁻² M_☉⁻¹
    
    # Integrate to get enclosed mass
    M_enc = np.zeros_like(r)
    for i in range(len(r)):
        integrand = rho_total[:i+1] * r[:i+1]**2
        M_enc[i] = 4 * np.pi * np.trapz(integrand, r[:i+1])
    
    # g = GM/r²
    g_bar = G * M_enc / (r**2 + 1e-12)
    return g_bar, M_enc
```

### 5. Abel Transform for Projection
```python
# From existing code (cluster_lensing_analysis.py)
def abel_project_sigma(r, rho, R):
    """Project spherical density to surface density.
    
    Σ(R) = 2 ∫_R^∞ ρ(r) r / sqrt(r² - R²) dr
    """
    Sigma = np.zeros_like(R)
    for i, R_i in enumerate(R):
        mask = r > R_i
        if np.sum(mask) > 1:
            r_int = r[mask]
            rho_int = rho[mask]
            integrand = rho_int * r_int / np.sqrt(r_int**2 - R_i**2 + 1e-20)
            Sigma[i] = 2 * np.trapz(integrand, r_int)
    return Sigma
```

### 6. Compute Einstein Radius
```python
def compute_einstein_radius(R, kappa, z_lens, z_source):
    """Find θ_E where mean convergence = 1."""
    from scipy.interpolate import interp1d
    
    # Compute mean convergence <κ>(<R)
    kappa_mean = np.zeros_like(R)
    for i in range(len(R)):
        if i == 0:
            kappa_mean[i] = kappa[i]
        else:
            kappa_mean[i] = np.trapz(kappa[:i+1] * R[:i+1], R[:i+1]) / (0.5 * R[i]**2)
    
    # Find R where <κ> = 1
    if kappa_mean.max() < 1.0:
        return None  # No Einstein radius
    
    f = interp1d(kappa_mean, R, kind='linear', bounds_error=False, fill_value='extrapolate')
    R_E_kpc = f(1.0)
    
    # Convert to arcsec
    from lensing_cosmology import angular_diameter_distance_kpc
    D_A = angular_diameter_distance_kpc(z_lens)
    theta_E_arcsec = (R_E_kpc / D_A) * (180 / np.pi) * 3600
    
    return theta_E_arcsec
```

### 7. Cosmology Utilities
```python
# From existing lensing_cosmology.py or inline:
def angular_diameter_distance_kpc(z):
    """Angular diameter distance in kpc (flat ΛCDM)."""
    from scipy.integrate import quad
    H0 = 70.0  # km/s/Mpc
    Om = 0.3
    Ol = 0.7
    c = 299792.458  # km/s
    
    E = lambda z: np.sqrt(Om * (1 + z)**3 + Ol)
    D_c = (c / H0) * quad(lambda z: 1/E(z), 0, z)[0]  # Mpc
    D_A = D_c / (1 + z)  # Mpc
    return D_A * 1000  # kpc

def sigma_crit(z_lens, z_source):
    """Critical surface density [M_☉/kpc²]."""
    if z_source <= z_lens:
        return np.inf
    
    D_d = angular_diameter_distance_kpc(z_lens)
    D_s = angular_diameter_distance_kpc(z_source)
    D_ds = (D_s * (1 + z_lens) - D_d * (1 + z_lens)) / (1 + z_source)
    
    c = 299792.458  # km/s
    G = 4.300917270e-6  # kpc km² s⁻² M_☉⁻¹
    
    return (c**2 / (4 * np.pi * G)) * (D_s / (D_d * D_ds))
```

---

## 🔄 **TYPICAL WORKFLOW**

```python
# 1. Load frozen hyperparameters
hp = load_sparc_hyperparameters()
kernel = PathSpectrumKernel(hp)

# 2. Load cluster data
cluster_data = load_cluster_baryons("MACSJ0416")
r = cluster_data['r']
rho_bar = cluster_data['rho_gas'] + cluster_data['rho_stars']

# 3. Compute baryonic quantities
g_bar, M_bar = compute_g_bar(r, rho_bar)
v_bar = np.sqrt(g_bar * r)

# 4. Apply path-spectrum boost
K = kernel.many_path_boost_factor(r, v_bar, g_bar)
g_total = g_bar * (1.0 + K)

# 5. Get effective density
M_eff = M_bar * (1.0 + K)  # approximate
rho_eff = enclosed_mass_to_density(r, M_eff)

# 6. Project to surface density
R = np.logspace(1, 3, 100)  # kpc
Sigma_eff = abel_project_sigma(r, rho_eff, R)

# 7. Compute lensing quantities
Sigma_crit = sigma_crit(z_lens=0.396, z_source=2.0)
kappa = Sigma_eff / Sigma_crit

# 8. Find Einstein radius
theta_E_pred = compute_einstein_radius(R, kappa, 0.396, 2.0)
theta_E_obs = 35.0  # arcsec

print(f"Predicted: {theta_E_pred:.1f}\"")
print(f"Observed:  {theta_E_obs:.1f}\"")
print(f"Ratio:     {theta_E_pred/theta_E_obs:.2f}")
```

---

## 📊 **EXPECTED RESULTS (BALLPARK)**

### If Model Works (Track B1 Success):
- θ_E prediction within factor of 1.5-2 of observations
- No per-cluster tuning (0 parameters)
- Median APE across 3 clusters: ~30-40%
- Comparable to NFW (which uses 2-3 params/cluster)

### If Model Fails:
- θ_E off by factor of 3-10
- Systematic bias (all too high or too low)
- Would indicate: scale issues OR mechanism doesn't extend to clusters

---

## ⚠️ **CRITICAL ASSUMPTIONS TO VERIFY**

1. **Scale Invariance:** Does L_0 = 4.99 kpc work at cluster scale?
2. **Spherical Symmetry:** Clusters may be more elliptical than galaxies
3. **Boost Factor Application:** Is K(r) additive for g_total?
4. **Temperature Effects:** Do we need pressure support correction?

---

## 🚀 **START HERE**

### Day 1: Single Cluster Test (MACS0416)
```bash
# Create script
touch many_path_model/run_cluster_lensing_single.py

# Run on MACS0416
python many_path_model/run_cluster_lensing_single.py --cluster MACSJ0416
```

**Goal:** Get first θ_E prediction, compare to 35"

### Day 2-3: Debug & Iterate
- If θ_E too small → Check scale factors
- If θ_E too large → Check boost application
- Plot radial profiles for diagnostics

### Day 4-5: 3-Cluster Validation
- Run on MACS0416, MACS0717, Abell 1689
- Compute statistics
- Compare to NFW benchmarks

**Success:** Competitive with NFW, zero tuning

---

## 📖 **USEFUL EXISTING CODE SNIPPETS**

### From `cluster_lensing_analysis.py`:
- Lines 136-177: Cosmology functions
- Lines 179-250: Abel projection
- Lines 300-400: Full lensing pipeline

### From `validation_suite.py` (SPARC):
- Data loading patterns
- Metrics computation
- Result formatting

### From `path_spectrum_kernel_track2.py`:
- Lines 80-120: Boost factor computation
- Lines 150-200: Coherence function
- Lines 250-300: Geometry gating

---

**Ready to implement Track B1!**

**Start with:** Single cluster (MACS0416), frozen params, compare θ_E to 35"

**Timeline:** 1-2 days for first result, 1-2 weeks for full validation

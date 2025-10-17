# Cluster Lensing Catalog Construction Notes

**File**: `data/cluster_lensing_catalog.csv`  
**Created**: 2025  
**Purpose**: Mass-scaled hierarchical inference on cluster strong lensing

---

## Data Sources

### 1. CLASH Survey (Umetsu et al. 2016)
**Reference**: Umetsu et al. 2016, ApJ, 821, 116  
**DOI**: 10.3847/0004-637X/821/2/116

**Provided**:
- Total mass M₂₀₀c and concentration c₂₀₀c from weak+strong lensing
- 20 clusters (16 X-ray selected + 4 high-magnification)
- NFW fits from combined lensing analysis

**Note**: Catalog uses M₂₀₀c values; M₅₀₀ and R₅₀₀ were derived assuming:
```
M₅₀₀ ≈ 0.65 × M₂₀₀c  (typical for NFW profiles)
R₅₀₀ = R₂₀₀c × (200/500)^(1/3) × f(c)
```

### 2. Einstein Radii (Strong Lensing)
**Sources**:
- CLASH strong lensing analysis (Zitrin et al. 2015, ApJ, 801, 44)
- Hubble Frontier Fields (Lotz et al. 2017, ApJ, 837, 97)
- Individual cluster lensing papers (see notes column)

**Method**:
- θ_E measured from critical curves at z_source ~ 2-3
- Typical uncertainties: ~8-10% for tier 1, ~10-15% for tier 2
- Complex mergers (tier 3): uncertainties ~12-18%

**Effective source redshift**:
- Used median z_source from lensed arc catalogs
- Range: z_s = 1.0-2.8 (higher for distant clusters)

### 3. Tier Assignments

**Tier 1 (Gold)**: N=7 clusters
- Clean strong lensing morphology
- No ongoing mergers
- θ_E uncertainty < 10%
- Examples: Abell 2261, RX J1347.5-1145, MACS J1206.2-0847

**Tier 2 (Silver)**: N=11 clusters  
- Good lensing quality with mild systematics
- Minor substructure or projection effects
- θ_E uncertainty < 15%
- Examples: MS 2137-2353, MACS J1149.5+2223

**Tier 3 (Complex)**: N=2 clusters (excluded from main analysis)
- MACS J0416.1-2403: Bimodal merger, complex mass distribution
- MACS J0717.5+3745: Extreme four-way merger, highly irregular

---

## Catalog Schema

| Column | Units | Description |
|--------|-------|-------------|
| `cluster_name` | — | CLASH identifier |
| `z_lens` | — | Cluster redshift (spectroscopic) |
| `z_source` | — | Effective source redshift for strong lensing |
| `theta_E_obs` | arcsec | Observed Einstein radius |
| `sigma_theta_E` | arcsec | 1σ uncertainty on θ_E |
| `M500_1e14Msun` | 10¹⁴ M☉ | Halo mass at R₅₀₀ |
| `R500_Mpc` | Mpc | Radius enclosing 500×ρ_crit |
| `tier` | — | Quality flag (1=Gold, 2=Silver, 3=Complex) |
| `has_weak_lensing` | 0/1 | Weak lensing data available |
| `notes` | — | Additional context |

---

## Systematic Considerations

### 1. Source Redshift Distribution
**Issue**: Real lensing comes from extended z_source distribution, not single z_s  
**Mitigation**: Used median z_s from arc catalogs; systematic bias < 5% on θ_E

### 2. Mass Definition (M₂₀₀c vs M₅₀₀)
**Issue**: CLASH reports M₂₀₀c; we need M₅₀₀ for mass-scaling relation  
**Conversion**: 
```python
# Approximate conversion (depends on c₂₀₀c)
M500 = M200c * (R500/R200c)**3
R500 = R200c * (500/200)**(1/3) / f_nfw(c)
```
**Uncertainty**: ~15% propagated to M₅₀₀

### 3. Triaxiality and Projection
**Issue**: Real clusters are triaxial; lensing sees projected mass  
**Model handles**: Per-cluster geometry factors q_LOS and q_plane in inference

### 4. Line-of-Sight Contamination
**Issue**: Large-scale structure along LOS can boost lensing  
**Model handles**: External convergence κ_ext ~ N(0, 0.03²) per cluster

---

## Usage Example

```python
import pandas as pd

# Load catalog
catalog = pd.read_csv('data/cluster_lensing_catalog.csv')

# Filter to tier 1+2, exclude MACS0717
analysis_sample = catalog[
    (catalog['tier'].isin([1, 2])) &
    (catalog['cluster_name'] != 'MACSJ0717.5+3745')
]

print(f"Analysis sample: N={len(analysis_sample)} clusters")
print(f"Mass range: {analysis_sample['M500_1e14Msun'].min():.1f} - {analysis_sample['M500_1e14Msun'].max():.1f} × 10¹⁴ M☉")
print(f"Redshift range: {analysis_sample['z_lens'].min():.3f} - {analysis_sample['z_lens'].max():.3f}")
```

---

## Future Improvements

### High Priority
1. **Add Abell 1689**: Famous strong lens (z=0.184), θ_E ~ 45-50"
2. **Add Abell 2029**: Nearby cluster (z=0.077), θ_E ~ 28"
3. **Incorporate BUFFALO survey**: Extended HFF data for 4 clusters

### Medium Priority
4. **Refine M₅₀₀ estimates**: Use cluster-specific NFW conversions from Table 2
5. **Per-cluster κ_ext priors**: From ray-tracing simulations (Meneghetti+2014)
6. **Weak lensing constraints**: Add shear profile data where available

### Low Priority
7. **Substructure fraction**: Flag clusters with significant subhalos
8. **BCG stellar mass**: Correct for central galaxy contribution

---

## Validation

### Cross-Check 1: Einstein Radius vs Mass
Expected scaling: θ_E ∝ √M at fixed z  
```python
import numpy as np
log_theta = np.log10(catalog['theta_E_obs'])
log_M = np.log10(catalog['M500_1e14Msun'])
slope, intercept = np.polyfit(log_M, log_theta, 1)
print(f"θ_E ∝ M^{slope:.2f}")  # Should be ~0.5
```

### Cross-Check 2: Comparison with Literature
| Cluster | Our θ_E | Literature | Source |
|---------|---------|------------|--------|
| RX J1347.5-1145 | 52.6" | 51-54" | Bradač+2008 |
| MACS J1206.2-0847 | 41.2" | 40-42" | Zitrin+2012 |
| Abell 2261 | 48.5" | 47-50" | Coe+2012 |

---

## References

1. **Umetsu et al. 2016**: CLASH weak+strong lensing masses
2. **Zitrin et al. 2015**: CLASH strong lensing analysis
3. **Lotz et al. 2017**: Hubble Frontier Fields overview
4. **Merten et al. 2015**: CLASH combined lensing methodology

---

## Notes on Tier 3 Clusters

### MACS J0717.5+3745 (z=0.548)
- **Structure**: Four-way merger in progress
- **Lensing**: Highly complex critical curves
- **Issue**: θ_E varies by 20-30% depending on model assumptions
- **Recommendation**: Exclude from main analysis; use as robustness test

### MACS J0416.1-2403 (z=0.396)
- **Structure**: Binary merger, two main halos separated by ~150 kpc
- **Lensing**: Two distinct critical curves
- **Issue**: Single θ_E poorly defined; need multi-component model
- **Recommendation**: Exclude or treat as two separate lenses

---

**Contact**: For questions about catalog construction or to report issues, see repository README.

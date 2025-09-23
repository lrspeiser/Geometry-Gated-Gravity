# Cluster Data Documentation

## Data Schemas

### Required Files
- **gas_profile.csv**: Gas density profile
  - Format: `r_kpc,n_e_cm3` or `r_kpc,rho_gas_Msun_per_kpc3`
  - Electron density or gas mass density
  
- **temp_profile.csv**: Temperature profile  
  - Format: `r_kpc,kT_keV,kT_err_keV`
  - X-ray spectroscopic temperature in keV
  
- **clump_profile.csv**: Clumping factor profile
  - Format: `r_kpc,C`
  - C = sqrt(<rho^2>/<rho>^2) ; C=1 if no clumping
  
- **stars_profile.csv**: Stellar density profile
  - Format: `r_kpc,rho_star_Msun_per_kpc3`
  - BCG + ICL combined 3D mass density

## Units
- Distances: kpc
- Densities: Msun/kpc^3 (3D volume density)
- Surface densities: Msun/pc^2 (projected)
- Temperatures: keV
- Clumping: dimensionless

## Cluster Details

### ABELL_0426 (Perseus)
- **Type**: Cool-core cluster with AGN feedback
- **z**: 0.0179
- **Data Sources**: 
  - Gas/temp: Chandra/XMM observations
  - Clumping: Urban+2014 (A&A 558, A33)
  - BCG/ICL: Digitized from optical/NIR data
- **Notes**: Central AGN cavity structures, multi-phase gas

### ABELL_1689
- **Type**: Massive lensing cluster
- **z**: 0.183
- **Data Sources**:
  - Gas/temp: Chandra deep observations
  - Clumping: Eckert+2013 (A&A 551, A22)
  - BCG/ICL: HST + ground-based photometry
- **Notes**: Strong lensing arcs, high central concentration

### A1795
- **Type**: Relaxed cool-core cluster
- **z**: 0.0625
- **Data Sources**:
  - Gas/temp: Representative ACCEPT-quality profiles
  - Clumping: Moderate central enhancement (C_0=1.8)
  - BCG: M_tot=7.5e11 Msun, Re=12.5 kpc, n=4
  - ICL: M_icl=3.0e11 Msun, a=100 kpc (Hernquist)
- **Notes**: Well-studied AGN feedback, cool core

### A478
- **Type**: Intermediate-mass regular cluster
- **z**: 0.0881
- **Data Sources**:
  - Gas/temp: Representative ACCEPT-quality profiles
  - Clumping: Moderate central enhancement (C_0=1.9)
  - BCG: M_tot=8.5e11 Msun, Re=14 kpc, n=4
  - ICL: M_icl=3.5e11 Msun, a=120 kpc (Hernquist)
- **Notes**: Regular morphology, moderate central density

### A2029
- **Type**: Massive relaxed cluster
- **z**: 0.0773
- **Data Sources**:
  - Gas/temp: Representative ACCEPT-quality profiles
  - Clumping: Strong central enhancement (C_0=2.0)
  - BCG: M_tot=1.2e12 Msun, Re=18 kpc, n=4.2
  - ICL: M_icl=5.0e11 Msun, a=150 kpc (Hernquist)
- **Notes**: Prominent BCG, high central mass concentration

## Profile Generation Tools

### BCG/ICL Profile Builder
```bash
py -u rigor/scripts/build_bcg_profile.py \
  --cluster_dir data/clusters/<NAME> \
  --Mtot_Msun <BCG_mass> --Re_kpc <effective_radius> \
  --sersic_n <index> --icl_Mtot_Msun <ICL_mass> \
  --icl_a_kpc <scale_length>
```

### Data Validation
```bash
py -u rigor/scripts/check_cluster_data.py --cluster <NAME>
```

Checks:
- File existence and format
- Radius monotonicity
- Profile positivity
- Mass integral convergence
- Unit consistency

## References
- Urban et al. 2014, A&A 558, A33 (Perseus clumping)
- Eckert et al. 2013, A&A 551, A22 (A1689 clumping)
- Cavagnolo et al. 2009, ApJS 182, 12 (ACCEPT catalog)
- Prugniel & Simien 1997, A&A 321, 111 (Sérsic deprojection)

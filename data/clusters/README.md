# Cluster data schemas and provenance

Schemas
- gas_profile.csv: either
  - r_kpc,n_e_cm3
  - r_kpc,rho_gas_Msun_per_kpc3
- temp_profile.csv: r_kpc,kT_keV[,kT_err_keV]
- clump_profile.csv: r_kpc,C
  - If no measured clumping is available, C=1 is assumed; placeholder files contain only headers.
- stars_profile.csv: r_kpc,rho_star_Msun_per_kpc3
  - BCG+ICL; deprojection helper provided in rigor/scripts/build_bcg_profile.py

Provenance notes
- ABELL_0426 (Perseus): clumping from Urban+14; BCG/ICL from digitized profile; see paper methods.
- ABELL_1689: clumping from Eckert+13; BCG/ICL from digitized profile; see paper methods.
- A1795, A478, A2029: placeholders created; replace with ACCEPT-quality profiles and measured BCG/ICL. Use build_bcg_profile.py and document mass/radius in-line.

Units
- r_kpc in kpc; densities in Msun/kpc^3; surface densities in Msun/pc^2; temperatures in keV.

Validation
- Run rigor/scripts/check_cluster_data.py --clusters <comma-list> to validate monotonicity and basic integrals.

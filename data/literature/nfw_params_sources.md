# NFW parameters extracted from Umetsu et al. 2016 (ApJ 821, 116)

- Source PDF: data/Umetsu_2016_ApJ_821_116.pdf
- Tables used: Table 2 (NFW parameters), Table 3 (mass references)
- Cosmology: h=0.7, Omega_m=0.27, Omega_L=0.73 (h_70 = 1)
- Conversions:
  - M200c (10^14 Msun/h70) -> Msun by ×1e14 (h70=1)
  - r_-2 (Mpc/h70) -> r_s (kpc) by ×1000 (h70=1)
- File written: data/literature/nfw_params.json
- Fields: M_200c_Msun, c_200c, r_s_kpc, symmetric 1σ errors

Notes
- Cluster IDs follow repository conventions (e.g., MACSJ0416, RXJ2248, ABELL_0209).
- High-magnification subset explicitly included (MACSJ0416, MACSJ0717, MACSJ1149, MACSJ0647).
- These are single spherical NFW fits to combined strong+weak lensing surface mass density profiles, per Umetsu+16.

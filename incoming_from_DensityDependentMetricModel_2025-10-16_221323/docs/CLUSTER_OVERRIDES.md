# Cluster Overrides

This repository supports per-cluster configuration overrides to inject cluster-specific realism during inference and holdout validation.

- Location: `data/overrides/{SANITIZED_NAME}.json`
- Name sanitization: uppercase, remove spaces/hyphens/dots, replace `+` with `PLUS`.
  - Example: `MACS J1149.5+2223` → `MACSJ11495PLUS2223.json`

## Supported Fields

- `kappa_ext_sigma` (number): External convergence prior width for the cluster (default 0.03).
- `bcg` (object): BCG Hernquist parameters
  - `M_Msun` (number): Stellar mass in solar masses (e.g., 2.0e12)
  - `a_kpc` (number): Scale radius in kpc (e.g., 15)
- `extra_baryon_components` (array): Additional projected components
  - Each component object:
    - `type`: currently `hernquist` supported
    - `M_Msun`: mass in solar masses
    - `a_kpc`: scale radius in kpc
- `z_source` (number): Override single effective source redshift
- `source_distribution` (object): Source redshift distribution P(z_s)
  - `type`: `mixture_normal`
  - `components`: array of `{ "weight": w, "mu": z, "sigma": dz }`
  - `z_min`, `z_max`: integration bounds (optional; defaults sensible)
  - `n_grid`: integration points (optional; default 400)
- `geometry` (object): Optional triaxial population override for validation
  - `mu_q` (number): mean axis ratio (default 1.0)
  - `sigma_q` (number): scatter (default 0.1)

The P(z_s) is used to compute an effective lensing efficiency β = ⟨D_ls/D_s⟩. The inference/validation then uses an effective pair `(D_source_eff=1, D_LS_eff=β)` so that Σ_crit ∝ 1/(D_lens·β).

## Example: MACS J1149.5+2223

```json
{
  "kappa_ext_sigma": 0.05,
  "bcg": { "M_Msun": 2.2e12, "a_kpc": 16.0 },
  "extra_baryon_components": [
    { "type": "hernquist", "M_Msun": 6.0e12, "a_kpc": 120.0 }
  ],
  "source_distribution": {
    "type": "mixture_normal",
    "components": [
      { "weight": 0.6, "mu": 1.7, "sigma": 0.2 },
      { "weight": 0.4, "mu": 3.0, "sigma": 0.3 }
    ],
    "z_min": 0.3, "z_max": 6.0, "n_grid": 400
  }
}
```

## How it’s used

- Inference: `run_mass_scaled_hierarchical_inference.py` loads overrides per cluster to:
  - Use effective P(z_s) distances when present
  - Adjust κ_ext prior width per cluster
  - Add extra baryon components to Σ(R)
- Validation: `run_holdout_validation.py` does the same for hold-out predictions.

## Notes

- If both `z_source` and `source_distribution` are present, the distribution is used.
- Only `hernquist` extra components are supported at present; extend as needed.
- Distances require `astropy`; install via `pip install astropy`.

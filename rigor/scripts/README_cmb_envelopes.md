# Clean-slate CMB envelopes: Planck plik_lite TT

This helper reads the Planck 2018 plik_lite binned TT bandpowers and their covariance (no `clik` runtime needed), constructs simple \ell-space distortion templates, and computes 1σ / 95% bounds on how large such late-time imprints could be without being ruled out by Planck TT alone.

It does not assume ΛCDM, a specific expansion history, or a photon–baryon plasma; it’s a data-driven consistency bound for phenomenological late-time effects (e.g., gravity-in-voids reprocessing).

## Inputs (from plik_lite `_external` directory)

- `cl_cmb_plik_v22.dat` — columns: (ℓ_eff, C_ℓ[µK²], σ[µK²])
- `c_matrix_plik_v22.dat` — covariance matrix for the binned vector (text or binary)
- Optional: `blmin.dat`, `blmax.dat`, `bweight.dat` — binning metadata (not required here)

See `data/baseline/readme_baseline.md` for repository-specific file locations and column notes.

## Templates

Let `y` be the measured binned TT vector (µK²) and `C` its covariance. For a template `F(ℓ)`, we bound the amplitude `A` (at A=0) via

```
σ_A = 1 / sqrt( F^T C^{-1} F )
A_95 = 1.96 * σ_A
```

We provide three templates:

1) `lens` — lensing-like smoothing
   - Build a smoothed copy of the measured vector, `S(y)` (Gaussian in ℓ with width `lens_sigma`), and take `F = S(y) - y`.

2) `gk` — low-ℓ gravitational kernel
   - `F = y * G_low(ℓ) * (ℓ/ℓ0)^power`, where `G_low(ℓ) = 0.5 * [1 + tanh((ℓ0 - ℓ)/width)]`.

3) `vea` — high-ℓ void-envelope amplification
   - `F = y * G_high(ℓ) * (ℓ/ℓ0)^alpha`, where `G_high(ℓ) = 0.5 * [1 + tanh((ℓ - ℓ0)/width)]`.

These are intended as stand-ins for late-time reprocessing: peak blurring, low-ℓ reweighting (ISW-like), and high-ℓ small-angle enhancements.

## Usage

```
# lensing-like smoothing envelope
python -u rigor/scripts/cmb_static_expts.py \
  --plik_lite_dir <.../plik_lite_v22_TT.clik/clik/lkl_0/_external> \
  --mode lens \
  --out_dir out/cmb_envelopes

# low-ℓ gate (GK)
python -u rigor/scripts/cmb_static_expts.py \
  --plik_lite_dir <.../plik_lite_v22_TT.clik/clik/lkl_0/_external> \
  --mode gk --gk_l0 80 --gk_width 40 --gk_power -1.0 \
  --out_dir out/cmb_envelopes

# high-ℓ gate (VEA)
python -u rigor/scripts/cmb_static_expts.py \
  --plik_lite_dir <.../plik_lite_v22_TT.clik/clik/lkl_0/_external> \
  --mode vea --vea_l0 800 --vea_width 150 --vea_alpha 0.15 \
  --out_dir out/cmb_envelopes
```

If `matplotlib` is available, a quicklook PNG is saved alongside JSON/CSV.

## Outputs

- `cmb_envelope_<mode>.json`
  - `{ "sigma_A": <float>, "A95": <float>, "nbins": <int>, "params": {…} }`
- `cmb_templates_<mode>.csv`
  - columns: `ell_eff, Cl_uK2, template_F, Cl_plus_1sigma, Cl_minus_1sigma`
- `cmb_peaks_metrics.json`
  - crude peak positions/contrasts extracted from the data vector (handy for acoustic-scale checks)
- (optional) `cmb_envelope_<mode>.png`

## Notes & caveats

- This is **additive** and evaluated about `A=0`. It answers: “How large could a late-time imprint of shape `F(ℓ)` be, given Planck TT covariance?”
- Because `F` is constructed directly from the measured `y`, there’s no need to assume an unlensed TT or a ΛCDM baseline within this tool.
- For a physical interpretation (e.g., mapping `lens_sigma` to a predicted deflection power spectrum), replace the toy template by one derived from your gravity model.

## Next steps

- Add TE/EE and Planck lensing bandpowers (analogous loaders and templates).
- Replace `lens` with a template built from a predicted `C_L^{\phi\phi}` or `A_\mathrm{lens}`.
- Replace `gk`/`vea` gates with line-of-sight integrals of a metric slip or growth response tied to your LogTail / MuPhi phenomenology.

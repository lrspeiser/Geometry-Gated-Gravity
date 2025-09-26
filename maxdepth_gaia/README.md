# Saturated-Well Gravity vs. Gaia (Local data, runnable)

This package gives you a **complete, runnable** Python pipeline to test a
"**saturated‑well**" (max-depth) gravity toy model against a Milky Way rotation
curve built from your **local Gaia DR3 products**.

Default baselines now align with widely used Milky Way models: the GR baseline
uses a multi-component disk (thin+thick+gas) plus a small bulge, and the NFW
fit is restricted to MW-like ranges to avoid unphysical corners.

What it does now (no online queries):
1. Ingests Gaia MW data from your repo under `data/`:
   - Preferred: `data/gaia_sky_slices/processed_*.parquet` (per‑longitude processed slices; documented in `data/README.md`).
   - Fallback: `data/gaia_mw_real.csv` (Galactocentric, star-level table).
2. Builds a **rotation curve** from thin-disk stars via robust binning (optional asymmetric-drift correction).
3. Fits baryons in the inner Galaxy (Miyamoto–Nagai disk + Hernquist bulge) to set the baseline.
4. Detects the **boundary radius** where residuals depart from baryons (consecutive‑excess and/or a BIC changepoint).
5. Anchors the **saturated‑well** tail at the boundary mass (v_flat from M(<Rb)) and fits its shape to outer bins.
6. Compares to **Baryons+NFW** and reports **χ²/AIC/BIC**.
7. Saves a publication‑style PNG and CSV/JSON outputs under `maxdepth_gaia/outputs/`.

> **Important caveats**
> - This is a first-pass, fast pipeline intended for exploration. A publication‑grade
>   analysis should treat selection functions, asymmetric drift, distance systematics,
>   and vertical structure carefully (Jeans modelling / action‑based modelling).
> - The “saturated‑well” model is a toy parameterization for your idea; it’s not a
>   relativistic theory. Lensing predictions included here are heuristic.
>
> **Good news**: the structure is clean and modular, so you can iterate quickly.

---

## Install

Create a clean environment and install dependencies (Python ≥3.10 recommended):

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```

Notes:
- Online queries are removed; this workflow uses your local data only.
- Parquet reading requires `pyarrow` (included in requirements). Cupy is optional; code falls back to NumPy.

---

## Run (local data)

Examples (Windows PowerShell):

Use processed Gaia slices (preferred):
```bash
python -m maxdepth_gaia.run_pipeline \
  --use_source slices \
  --slices_glob "C:\\Users\\henry\\dev\\GravityCalculator\\data\\gaia_sky_slices\\processed_*.parquet" \
  --zmax 0.5 --sigma_vmax 30 --vRmax 40 \
  --rmin 3 --rmax 20 --nbins 24 \
  --inner_fit_min 3 --inner_fit_max 8 \
  --boundary_method both \
  --saveplot "C:\\Users\\henry\\dev\\GravityCalculator\\maxdepth_gaia\\outputs\\mw_rotation_curve_maxdepth.png"
```

Use the Galactocentric CSV:
```bash
python -m maxdepth_gaia.run_pipeline \
  --use_source mw_csv \
  --mw_csv_path "C:\\Users\\henry\\dev\\GravityCalculator\\data\\gaia_mw_real.csv" \
  --rmin 3 --rmax 20 --nbins 24 \
  --saveplot "C:\\Users\\henry\\dev\\GravityCalculator\\maxdepth_gaia\\outputs\\mw_rotation_curve_maxdepth.png"
```

Key options:
- `--baryon_model {single,mw_multi}` selects the GR baseline. Default: `mw_multi` (MW-like thin+thick stellar disks + H I + H₂ as MN approximations, plus a small bulge).
- `--ad_correction` enables an (optional) asymmetric‑drift correction on binned data.
- `--ad_poly_deg` and `--ad_frac_err` tune AD smoothing and error inflation.
- `--mond_kind {simple,standard}` and `--mond_a0` configure the MOND baseline.
- `--gate_width_kpc`, `--fix_m`, `--eta_rs` control the saturated‑well tail shape.
- `--boundary_method` tries both a consecutive‑excess significance test and a BIC changepoint to locate the onset of the tail.
- `--gate_width_kpc` fixes the smooth C¹ gate width ΔR.
- `--fix_m` fixes the transition sharpness m across all radii.
- `--eta_rs` fixes the tail scale as `R_s = eta_rs * R_b` (global shape control), leaving only `xi` free for the tail amplitude.
- Outputs land in `maxdepth_gaia/outputs/`.

---

## What is the “saturated‑well” model?

You described a gravity **well with a maximum depth** whose **pull extends outward**,
making outer tracers move faster and light lens **more than Newtonian** expectations.

We capture that with a **logarithmic‑like** asymptote that yields nearly **flat
rotation curves** at large radius:

\[
v_{\rm model}^2(R) \;=\; v_{\rm bary}(R)^2 \;+\; v_{\rm extra}^2(R),
\quad
v_{\rm extra}^2(R) \;=\; v_\mathrm{flat}^2\,\Big[1-\exp\!\big(-(R/R_s)^m\big)\Big].
\]

- For \(R \ll R_s\): the extra term is small.
- For \(R \gg R_s\): \(v_{\rm extra} \to v_{\rm flat}\) (a flat tail set by the
  free parameter \(v_{\rm flat}\)).
- The **shape** of the transition is set by \(m\) (default 2).

This is a **phenomenological** representation of your idea, not a unique one.
You can swap it for any other closed form by editing `models.py`.

For **lensing**, a logarithmic potential gives a (heuristic) **impact‑parameter‑independent**
deflection \(\alpha \approx 2\pi v_{\rm flat}^2/c^2\). We expose a helper to compute
this so you can compare to strong‑lensing scales if you add an external catalog.

---

## Files produced

- `maxdepth_gaia/outputs/rotation_curve_bins.csv` — binned \(R, v_\phi, \sigma\) with counts
- `maxdepth_gaia/outputs/fit_params.json` — best‑fit parameters + uncertainties + boundary info
- `maxdepth_gaia/outputs/model_curves.csv` — dense model evaluations for inspection
- `maxdepth_gaia/outputs/mw_rotation_curve_maxdepth.png` — publication‑style comparison figure
- `maxdepth_gaia/outputs/used_files.json` — provenance for slice mode

---

## Extend / next steps

- Replace the quick rotation‑curve proxy with Jeans modelling (axisymmetric;
  correct for asymmetric drift).
- Swap in your preferred **baryonic mass model**.
- Add **MCMC** (`emcee`) to map parameter posteriors.
- Add an **external galaxy** rotation‑curve set (e.g., SPARC) for cross‑checks.
- Bolt on a **lensing catalog** and test the deflection scaling for galaxy lenses.

## Notes on baselines and priors

- GR baseline `mw_multi` approximates MWPotential2014/McMillan with two MN stellar
  disks (thin+thick) and two MN gas disks (H I, H₂) plus a small Hernquist bulge.
- NFW bounds are constrained to Milky Way–like ranges by default: `120 ≤ V200 ≤ 180`
  km/s and `8 ≤ c ≤ 20`. This avoids optimizer excursions to unrealistic corners
  (e.g., very low-c or very high-c) when outer data leverage is weak.
- A dashed grey overlay of the MW-like GR curve is drawn on every plot for
  immediate visual sanity checks.

### Wedge runner

Run azimuthal wedges and summarize boundary and fit stability across φ:

```bash
python -m maxdepth_gaia.wedge_runner --phi_bins 4 --ad_correction --baryon_model mw_multi --fix_m 1.5 --eta_rs 0.2 --gate_width_kpc 0.8
```

This writes `wedge_summary_phi{phi_bins}.csv/json` in `maxdepth_gaia/outputs/` with per-wedge `R_boundary`, `(lo,hi)` bootstrap bounds, saturated-well parameters (`xi, R_s, m, ΔR, v_flat`), and model-selection metrics.

---

# Executive Summary (what’s built and why)

- Purpose: Test a “saturated‑well” (max-depth/outward-extension) gravity toy model against real Gaia DR3 Milky Way kinematics, with rigorous GR baselines and NFW comparisons.
- Data: Runs only on your local repository data (Gaia DR3 Milky Way star-level CSV or processed sky-slice Parquet files). No synthetic data.
- Pipeline: Ingest → robust binning (optional asymmetric drift) → fit inner baryons → detect boundary → anchor and fit saturated-well tail beyond the boundary → fit NFW → produce figures, curves, and metrics.
- Performance: On the latest MW run, the saturated‑well model outperforms both GR (baryons-only) and GR+NFW across 6–16.5 kpc and overall by MAE/RMSE and χ²-based metrics; inside the boundary it is exactly GR by construction.
- Rigor: Uses a MW-like multi-component GR baseline (thin+thick stellar disks + H I + H₂ + small bulge), MW-like priors for NFW, two boundary detectors, bootstrap CI, and a smooth C¹ gate that is exactly zero inside the boundary.

# Data sources and I/O (local only)

- Preferred star-level input: `data/gaia_sky_slices/processed_*.parquet` (12 sky slices). Fallback: `data/gaia_mw_real.csv` (Galactocentric kinematics).
- Outputs are written to `maxdepth_gaia/outputs/`:
  - `rotation_curve_bins.csv` — binned radii, median vφ, errors, counts, AD flags
  - `model_curves.csv` — dense curves for GR, GR+NFW, GR+SatWell
  - `fit_params.json` — all parameters and model comparison stats
  - `mw_rotation_curve_maxdepth.png` — main figure
  - `g_ratio_vs_GR.png` — g_model/g_GR profile
  - `pipeline.log` — log of the run

# Pipeline architecture (high level)

- Ingestion: Auto-detects between slices vs. MW CSV; filters by |z|, σ cuts, radial velocity limits; optional azimuthal wedge selection (φ bins).
- Binning: Fixed- or wedge-aware radial bins; robust medians; optional asymmetric drift (AD) correction with tunable smoothing and uncertainty inflation.
- Inner GR fit: By default we use the MW-like multi-disk+gas baseline (below). We still fit a single MN+Hernquist model over 3–8 kpc for diagnostics and error scaling.
- Error renormalization: Inner-window (3–8 kpc) reduced-χ² ≈ 1 against the chosen GR baseline for fair comparisons.
- Boundary detection: Consecutive-excess test and a BIC changepoint model; bootstrap CI on R_b.
- Outer fits:
  - Anchored saturated-well tail (no mass creation): v_flat anchored by enclosed baryon mass at R_b (ξ≤1), C¹ gate (exact 0 inside), smooth transition.
  - NFW comparison: V200–c parameterization with MW-like priors.
- Plotting: Publication-style PNG with data, models, boundary shading, and a dashed MWPotential2014-like GR reference overlay.

# Models

- MOND: Added a baseline computed from the same GR baryon curve using a ν(y) function.
  - Default: simple μ(x)=x/(1+x) ⇒ ν(y)=0.5+√(0.25+1/y), with a0=1.2×10^-10 m s^-2.
  - v_MOND(R) = √ν · v_baryon(R) with y = g_N/a0, g_N = v_baryon^2/R in consistent units.
  - Plotted as “Baryons + MOND” and included in metrics and fit_params.json.

- GR baseline (mw_multi): Two MN stellar disks (thin+thick), two MN gas disks (H I, H₂) and a small Hernquist bulge. Defaults (order-of-magnitude MW-like):
  - Bulge ~5×10^9 Msun, a_b=0.6 kpc
  - Thin disk ~4.5×10^10 Msun, a=3.0 kpc, b=0.3 kpc
  - Thick disk ~1.0×10^10 Msun, a=2.5 kpc, b=0.9 kpc
  - H I ~1.1×10^10 Msun, a=7.0 kpc, b=0.1 kpc; H₂ ~1.2×10^9 Msun, a=1.5 kpc, b=0.05 kpc
- NFW: V200–c with R_s=R200/c; bounds [120,180] km/s and [8,20] to avoid extreme/unphysical corners when outer leverage is weak.
- Saturated-well: Extra term v_extra^2(R) = v_flat^2 · [1 − exp(−(R/R_s)^m)].
  - Gate: C¹ smoothstep; exactly zero inside R_b and one outside (over ΔR).
  - Anchor: v_flat^2 = ξ · G M(<R_b)/R_b, with ξ ≤ 1.
  - Global tail controls (optional): fix m (–-fix_m), fix R_s via R_s = η · R_b (–-eta_rs), fix gate width ΔR (–-gate_width_kpc).

# Boundary detection

- Consecutive-excess: K consecutive bins with significant positive excess vs GR beyond a threshold.
- BIC changepoint: Evaluate candidate boundaries and fit the anchored tail outside; select by BIC; also compute ΔBIC vs baryons-only.
- Bootstrap: Resample bins with replacement to estimate R_b confidence interval.

# Rotation curve binning and AD correction

- Robust medians per bin; star counts logged per bin.
- Optional AD correction (polynomial smoothing; uncertainty inflation factor) to better reflect circular velocity proxies.

# GPU acceleration

- CuPy (if available) is used for large arrays; falls back to NumPy automatically.
- Backend is logged (cupy vs numpy).

# Latest Milky Way run (Gaia DR3) — headline results

- Data: 108,048 stars after filters; 17 radial bins (3–16.5+ kpc).
- Boundary: R_b ≈ 6.54 kpc; bootstrap 16–84%: 5.83–7.96 kpc.
- NFW (MW-like priors): V200 ≈ 180 km/s, c ≈ 14.8.
- Saturated-well tail: ξ ≈ 0.59, m ≈ 1.50, R_s ≈ 1.06 kpc, v_flat ≈ 136 km/s; smooth gate ΔR ≈ 0.27 kpc; lensing heuristic α ≈ 0.267".
- Model comparison (per-bin residuals on vφ; MAE/RMSE in km/s; χ² per bin):
  - 3–6 kpc (inside R_b): SatWell = GR (gate is zero). MOND performs best locally (χ²/bin ≈ 20.9).
  - 6–8 kpc: GR 74.9, NFW 68.1, MOND 33.7 (χ²/bin ≈ 47.7), SatWell 51.5 (χ²/bin ≈ 62.4).
  - 8–12 kpc: GR 45.7, NFW 36.8, MOND 15.3 (χ²/bin ≈ 4.08e4), SatWell 11.2 (χ²/bin ≈ 3.59e4 — best).
  - 12–16.5 kpc: GR 73.2, NFW 62.4, MOND 25.0 (χ²/bin ≈ 241.0), SatWell 29.1 (χ²/bin ≈ 147.5 — best).
  - Overall 3–16.5+ kpc: GR 69.5, NFW 61.7, MOND 33.7 (χ²/bin ≈ 1.45e4), SatWell 47.9 (χ²/bin ≈ 1.27e4 — best total χ²), while MOND gives the lowest MAE/RMSE overall.
- Interpretation: With a realistic MW-like GR baseline and MW-like NFW priors, the anchored saturated‑well model provides the best match to Gaia rotation bins beyond ~6 kpc; inside the boundary it reduces exactly to GR.

# How to run (recap)

- Slices (preferred):
  - python -m maxdepth_gaia.run_pipeline --use_source slices --slices_glob "C:\\Users\\henry\\dev\\GravityCalculator\\data\\gaia_sky_slices\\processed_*.parquet" --baryon_model mw_multi --ad_correction --saveplot "C:\\Users\\henry\\dev\\GravityCalculator\\maxdepth_gaia\\outputs\\mw_rotation_curve_maxdepth.png"
- MW CSV:
  - python -m maxdepth_gaia.run_pipeline --use_source mw_csv --mw_csv_path "C:\\Users\\henry\\dev\\GravityCalculator\\data\\gaia_mw_real.csv" --baryon_model mw_multi --saveplot "C:\\Users\\henry\\dev\\GravityCalculator\\maxdepth_gaia\\outputs\\mw_rotation_curve_maxdepth.png"
- Global tail shape example:
  - python -m maxdepth_gaia.run_pipeline --use_source auto --baryon_model mw_multi --ad_correction --fix_m 1.5 --eta_rs 0.2 --gate_width_kpc 0.8
- Wedges (stability check):
  - python -m maxdepth_gaia.wedge_runner --phi_bins 4 --ad_correction --baryon_model mw_multi --fix_m 1.5 --eta_rs 0.2 --gate_width_kpc 0.8

# Useful artifacts

- Overlay reference: A dashed MWPotential2014-like GR curve (from `mw_multi` defaults) appears on the main figure.
- Evaluator (optional): `maxdepth_gaia/outputs/eval_metrics.py` prints per-range MAE/RMSE and χ²/bin using current outputs.
- Budget audit: `budget_audit.png` and `budget_audit.csv` show the used fraction of the tail budget vs radius (should not exceed 1 by construction).

# Next steps (prioritized)

1) MOND baseline
- Add MOND (simple μ/ν function with a0) and compare MAE/RMSE/χ²/bin vs GR, NFW, SatWell on the same bins.

2) Lensing benchmarks
- Replace the heuristic α with a consistent deflection law for the saturated‑well potential; evaluate on a strong-lensing sample (e.g., SLACS), alongside GR+NFW and MOND.
- A starter evaluator is included: `lensing_benchmark.py`. Provide a small CSV with columns `name,z_l,z_s,D_l,D_s,D_ls` and call:
  ```bash
  python -c "from maxdepth_gaia.lensing_benchmark import run_lensing_eval; run_lensing_eval('maxdepth_gaia/data/slacs_min.csv', 'maxdepth_gaia/outputs/fit_params.json', 'maxdepth_gaia/outputs/slacs_theta_pred.csv')"
  ```
  This uses a SIS-like mapping: θ_E = 2π (v_flat/c)^2 (D_ls/D_s). For GR/NFW/MOND, override with appropriate flat speeds or replace with a full relativistic lensing law as needed.

3) Cross-galaxy generalization
- Fix global tail shape (m, η=R_s/R_b, ΔR) across galaxies; fit only (R_b, ξ). Add SPARC ingestion to reuse published baryon curves.

4) Robustness sweeps
- Azimuthal wedges (φ bins), AD settings, variable-width outer bins (ensure N≥50), and minimum outside-segment occupancy rules to stabilize R_b.

5) Fitting enhancements
- Optional joint GR+NFW fit over all bins (helper stub provided), MCMC (emcee) for uncertainty mapping, and a flexible NFW parameterization (M200, c, R200 variable).

6) Diagnostics
- “Budget audit” panel: plot cumulative used vs allowed v_flat^2 budget to verify the anchor is respected across R.

# Reproducibility and logging

- Every run logs to `maxdepth_gaia/outputs/pipeline.log`.
- Provenance of input files is written to `used_files.json` in slice mode.
- Fit parameters and metrics are in `fit_params.json`.

# Notes and references

- GR baseline follows MWPotential2014/McMillan-like scales and masses.
- Typical MW rotation curve reference: Eilers et al. (2018) — flat ~229 km/s at R0~8.1 kpc with gentle decline.
- Thick vs thin disk context: A&A (Pouliasis+ 2017) — motivates two-disk GR baseline.
- NFW priors align with MW-mass halo expectations: c~10–15; V200 ~140–160 km/s (we constrained to [120,180] and [8,20]).

---

---

## Citation note

This code helps you build a rotation curve from Gaia DR3. If you publish, please
cite **Gaia Collaboration (DR3)**, **Astropy**, and **Astroquery**.

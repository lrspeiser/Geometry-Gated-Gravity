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
- `--boundary_method` tries both a consecutive‑excess significance test and a BIC changepoint to locate the onset of the tail.
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

---

## Citation note

This code helps you build a rotation curve from Gaia DR3. If you publish, please
cite **Gaia Collaboration (DR3)**, **Astropy**, and **Astroquery**.

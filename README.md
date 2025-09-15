# GravityCalculator — PLUS version

This repository contains an interactive toy galaxy rotation calculator with a PLUS script that adds 3D components, live rotation curve plotting, and CSV export.

Quick start
- macOS example using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python toy_galaxy_rotation_calculator_plus.py
```

Features in PLUS
- 3D particle sampling of disk, bulge, and gas; x–y projection shown.
- Component editor sliders: disk/bulge/gas masses, scale lengths (disk Rd, gas Rd), outer cutoffs (Rmax), bulge scale a, vertical thickness hz.
- Physics sliders: G scale, GR k (toy), Extra attenuation, Boost fraction (auto-boost), Softening.
- Presets: MW-like disk, Central-dominated, Dwarf disk, LMC-like dwarf, Ultra-diffuse disk, Compact nuclear disk.
- Rotation curve plot: click “Plot Rotation Curve” to open a second figure that updates live as you move sliders.
- CSV export: “Save Curve CSV” writes rotation_curve.csv with columns [R_kpc, v_newton_kms, v_gr_no_boost_kms, v_final_kms, G_scale, GR_k_toy, extra_atten, boost_fraction, softening_kpc].

Command-line options
- --backend MacOSX|TkAgg|QtAgg|Agg: force matplotlib backend (use Agg for headless save).
- --save path.png: save a PNG of the main figure and exit (headless mode).
- --config: initial preset selection.
- --seed, --spread, --outer, --soften, --gscale, --grk, --atten, --boost, --vobs: initial values for sliders.

Headless smoke test
- To verify it runs without an interactive backend:

```bash
python toy_galaxy_rotation_calculator_plus.py --backend Agg --save preview.png
ls -l preview.png
```

Qt desktop app (recommended UI)
- Install PyQt (PyQt6 preferred, PyQt5 fallback), then run the Qt app:

  1) source .venv/bin/activate
  2) pip install PyQt6 || pip install PyQt5
  3) python gravity_qt_app.py

- The Qt app embeds Matplotlib for the plot and uses native controls; it’s faster and less cluttered than Matplotlib widgets.

How to use the Qt app
- Preset: Choose a galaxy type. This sets the number of stars and a reasonable structure (disk radius, bulge fraction, mass, etc.). You can expand the Structure panel to tweak.
- G scale: Global multiplier on G (Newton’s constant) applied everywhere. Higher G raises speeds roughly as sqrt(G).
- Density k (≥0) and Density R (kpc): Adjusts effective G at the evaluation point based on local baryonic density.
  - Formula: G_eff = G × (1 + k × (rho_ref / rho_local − 1)), clamped to [0.1×, 10×].
  - rho_local is computed from mass inside a sphere of radius R around the point; rho_ref is a global average density.
  - Practical ranges: k = 0.0–1.5 (mild), up to ~3.0 (strong); R = 1–5 kpc typical.
  - Expected effect: Larger k and/or smaller local density ⇒ higher G_eff at the edge; boosts outer speeds and flattens the rotation curve.
- Well k (≥0) and Well R (kpc): Per-star “gravity well” modifier that scales each source body’s vector based on the local density around that source.
  - Concept: shallower wells (low local density) get a multiplier >1; deeper wells get ≈1.
  - Same clamp to [0.1×, 10×]. Practical ranges: k = 0.0–1.0 typical; R = 1–3 kpc typical.
  - Expected effect: Accentuates contributions from outskirts/low-density sources; can raise the test star speed and change the vector field pattern.
- Observed v (km/s): Reference velocity for the test star. The readout shows Final − Observed.
- Structure (expand to edit):
  - N stars: 8–200,000. More stars look smoother but are heavier to compute. 2,000–10,000 is a good balance.
  - R disk (kpc): Outer size of the luminous disk; typical 6–30.
  - Bulge a (kpc): Bulge Plummer scale; typical 0.2–1.0.
  - Thickness hz (kpc): Vertical thickness; typical 0.2–0.6.
  - Bulge frac: 0–1; 0.0 for pure disk, ~0.8 for bulge-dominated.
  - Total mass (1e10 Msun): Total luminous+gas mass distributed over all stars; typical 0.05–10.
- Buttons:
  - Build: Rebuilds the galaxy using the current Structure fields (farthest star is auto-selected as test star).
  - Force Vectors…: Opens a table of per-body acceleration vectors; you can export all to CSV.
  - Save Curve CSV: Exports a rotation curve (R, v_Newton, v_GR-like, v_Final) and current physics settings.

What you should see / expected results
- Farthest-star radius: Shown in the text panel. Speeds update when you change physics knobs (debounced for smoothness).
- G scale ↑ ⇒ all speeds grow ≈ sqrt(G_scale). Doubling G makes speeds ~1.41×.
- Density k ↑ (with reasonable R) ⇒ higher outer speeds; the rotation curve becomes flatter/less Keplerian.
- Well k ↑ ⇒ boosts contributions from low-density sources; similar qualitative effect but source-based rather than point-based.
- N stars ↑ ⇒ smoother forces and a cleaner rotation curve (less noise), but heavier computation.
- R disk ↑ ⇒ test star typically farther out; speeds respond to the mass distribution and G settings.

Valid ranges (clamped in the app)
- G scale: 0.1–5.0
- Density k: 0.0–5.0
- Density R: 0.05–20.0 kpc
- Well k: 0.0–5.0
- Well R: 0.05–20.0 kpc
- Observed v: 0–1000 km/s
- N stars: 8–200000
- R disk: 0.5–200 kpc
- Bulge a: 0.02–10 kpc
- Thickness hz: 0.01–3 kpc
- Bulge frac: 0.0–1.0
- Total mass: 0.01–200 (×1e10 Msun)

Tips for performance
- Start with N stars = 2000–4000; raise if you need smoother curves.
- Keep Density/Well radii in the 1–5 kpc range to avoid very wide neighbor searches.
- The vectors table can be heavy for very large N—use Top N to preview the dominant contributors and export full CSV when needed.

Troubleshooting
- If the app complains about QtAgg or a missing Qt binding:
  - pip install PyQt6 (preferred) or pip install PyQt5.
- If the window is large/small, resize the app; the canvas scales and the native toolbar helps with zoom/pan.

Notes
- Requires numpy and matplotlib. For the Qt app, install PyQt6 or PyQt5.
- No API keys or web services are used.

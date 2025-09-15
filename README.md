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
```bash
pip install PyQt6 || pip install PyQt5
python gravity_qt_app.py
```
- The Qt app embeds Matplotlib for the plot and uses native controls; it’s faster and less cluttered than Matplotlib widgets.

Notes
- Requires numpy and matplotlib. For the Qt app, install PyQt6 or PyQt5.
- No API keys or web services are used.

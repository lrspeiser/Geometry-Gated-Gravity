#!/usr/bin/env python3

"""
Toy Galaxy Rotation Calculator — PLUS version

Features:
- Fully 3D particle sampling for disk, bulge, and gas components
- Sliders for component masses and structural parameters
- Global physics sliders: G scale, GR k (toy), extra attenuation, boost fraction, softening
- Presets: MW-like disk, Central-dominated, Dwarf disk, LMC-like dwarf, Ultra-diffuse disk, Compact nuclear disk
- Separate live-updating Rotation Curve figure
- CSV export of rotation curve and physics settings via "Save Curve CSV" button
- CLI options for backend override and headless save (for CI or quick preview)

Notes:
- Requires numpy and matplotlib. See README.md for setup and usage details.
- No web services or API keys are used here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib

# ---------- Physical constants ----------
G_AST = 4.30091e-6   # kpc * (km/s)^2 / Msun
c_kms = 299_792.458  # km/s

# ---------- Sampling helpers ----------
def symmetrize_xy(pos: np.ndarray) -> np.ndarray:
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.vstack([
        np.column_stack([ x,  y,  z]),
        np.column_stack([ x, -y,  z]),
        np.column_stack([-x,  y,  z]),
        np.column_stack([-x, -y,  z]),
    ])


def mirror_z(pos: np.ndarray) -> np.ndarray:
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return np.vstack([
        np.column_stack([x, y,  z]),
        np.column_stack([x, y, -z]),
    ])


def sample_exponential_disk_3d(N_base: int, Rd_kpc: float, Rmax_kpc: float, hz_kpc: float, rng: np.random.Generator) -> np.ndarray:
    if N_base <= 0:
        return np.zeros((0, 3))
    R = np.zeros(N_base); k = 0
    while k < N_base:
        r = rng.exponential(Rd_kpc) + rng.exponential(Rd_kpc)
        if r <= Rmax_kpc:
            R[k] = r; k += 1
    theta = rng.uniform(0.0, 2.0*np.pi, size=N_base)
    x = R * np.cos(theta); y = R * np.sin(theta)
    z = rng.normal(0.0, max(1e-6, hz_kpc), size=N_base)
    pos = np.column_stack([x, y, z])
    return mirror_z(symmetrize_xy(pos))


def sample_plummer_bulge_3d(N_base: int, a_kpc: float, Rmax_kpc: float, hz_kpc: float, rng: np.random.Generator) -> np.ndarray:
    if N_base <= 0:
        return np.zeros((0, 3))
    R = np.zeros(N_base); k = 0
    while k < N_base:
        r = rng.uniform(0.0, Rmax_kpc)
        w = r / (r*r + a_kpc*a_kpc)**1.5
        w_ref = a_kpc / (a_kpc*a_kpc + a_kpc*a_kpc)**1.5 + 1e-12
        if rng.uniform(0.0, 1.0) < (w / (w_ref + 1e-18)):
            R[k] = r; k += 1
    theta = rng.uniform(0.0, 2.0*np.pi, size=N_base)
    x = R * np.cos(theta); y = R * np.sin(theta)
    z = rng.normal(0.0, max(1e-6, hz_kpc), size=N_base)
    pos = np.column_stack([x, y, z])
    return mirror_z(symmetrize_xy(pos))


# ---------- Galaxy builder ----------
def make_galaxy_components(
    seed: int,
    spread_scale: float,
    M_disk: float,
    M_bulge: float,
    M_gas: float,
    Rd_disk: float,
    Rmax_disk: float,
    a_bulge: float,
    Rd_gas: float,
    Rmax_gas: float,
    hz_disk: float,
    hz_bulge: float,
    hz_gas: float,
    N_disk_total: int = 2400,
    N_bulge_total: int = 800,
    N_gas_total: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_list, mass_list = [], []

    if M_disk > 0 and N_disk_total > 0:
        N_base = max(1, N_disk_total // 8)
        pos_d = sample_exponential_disk_3d(N_base, Rd_disk*spread_scale, Rmax_disk*spread_scale, hz_disk, rng)
        if len(pos_d):
            pos_list.append(pos_d)
            mass_list.append(np.full(len(pos_d), M_disk / len(pos_d)))

    if M_bulge > 0 and N_bulge_total > 0:
        N_base = max(1, N_bulge_total // 8)
        pos_b = sample_plummer_bulge_3d(N_base, a_bulge*spread_scale, (3.0*spread_scale), hz_bulge, rng)
        if len(pos_b):
            pos_list.append(pos_b)
            mass_list.append(np.full(len(pos_b), M_bulge / len(pos_b)))

    if M_gas > 0 and N_gas_total > 0:
        N_base = max(1, N_gas_total // 8)
        pos_g = sample_exponential_disk_3d(N_base, Rd_gas*spread_scale, Rmax_gas*spread_scale, hz_gas, rng)
        if len(pos_g):
            pos_list.append(pos_g)
            mass_list.append(np.full(len(pos_g), M_gas / len(pos_g)))

    if pos_list:
        pos = np.vstack(pos_list)
        masses = np.concatenate(mass_list)
    else:
        pos = np.zeros((0, 3))
        masses = np.zeros((0,))
    return pos, masses


# ---------- Physics ----------
def net_acceleration_at_point(point: np.ndarray, pos: np.ndarray, masses: np.ndarray, G_eff: float,
                              k_GR: float = 0.0, atten_extra: float = 0.0, soften_kpc: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if len(masses) == 0:
        return np.zeros(3), np.zeros(3)
    r_vec = pos - point
    r2 = np.sum(r_vec*r_vec, axis=1)
    eps2 = (float(soften_kpc)**2)
    inv_r3 = 1.0 / np.power(r2 + eps2, 1.5)
    newton_terms = (r_vec * (masses * inv_r3)[:, None]).sum(axis=0)
    a_newton_vec = G_eff * newton_terms
    r = np.sqrt(r2 + eps2)
    per_body_scale = (1.0 - k_GR * (G_eff / max(1e-30, G_AST)) * G_AST * masses / (r * c_kms**2))
    per_body_scale = np.clip(per_body_scale, 0.0, 1.0)
    per_body_scale *= (1.0 - np.clip(atten_extra, 0.0, 0.999))
    gr_terms = (r_vec * ((masses * inv_r3) * per_body_scale)[:, None]).sum(axis=0)
    a_gr_vec = G_eff * gr_terms
    return a_gr_vec, a_newton_vec


def circular_speed_from_accel(a_vec: np.ndarray, R_vec_to_COM: np.ndarray) -> float:
    R = np.linalg.norm(R_vec_to_COM)
    if R <= 0:
        return 0.0
    u_in = -R_vec_to_COM / R
    a_rad = float(np.dot(a_vec, u_in))
    a_rad = max(0.0, a_rad)
    return np.sqrt(R * a_rad)


# ---------- Presets ----------
def preset_params(name: str) -> dict:
    if name == "MW-like disk":
        return dict(M_disk=6.0e10, M_bulge=1.2e10, M_gas=0.8e10,
                    Rd_disk=3.0, Rmax_disk=20.0, a_bulge=0.6,
                    Rd_gas=5.0, Rmax_gas=25.0, hz_disk=0.3, hz_bulge=0.3, hz_gas=0.1)
    if name == "Central-dominated":
        return dict(M_disk=1.5e10, M_bulge=8.0e10, M_gas=0.5e10,
                    Rd_disk=4.0, Rmax_disk=15.0, a_bulge=0.8,
                    Rd_gas=5.0, Rmax_gas=18.0, hz_disk=0.4, hz_bulge=0.5, hz_gas=0.2)
    if name == "Dwarf disk":
        return dict(M_disk=0.3e10, M_bulge=0.0e10, M_gas=0.15e10,
                    Rd_disk=1.0, Rmax_disk=6.0, a_bulge=0.3,
                    Rd_gas=1.5, Rmax_gas=7.0, hz_disk=0.2, hz_bulge=0.2, hz_gas=0.1)
    if name == "LMC-like dwarf":
        return dict(M_disk=0.30e10, M_bulge=0.0e10, M_gas=0.10e10,
                    Rd_disk=1.5, Rmax_disk=9.0, a_bulge=0.2,
                    Rd_gas=2.0, Rmax_gas=10.0, hz_disk=0.25, hz_bulge=0.2, hz_gas=0.12)
    if name == "Ultra-diffuse disk":
        return dict(M_disk=0.05e10, M_bulge=0.0e10, M_gas=0.02e10,
                    Rd_disk=5.0, Rmax_disk=30.0, a_bulge=0.3,
                    Rd_gas=6.0, Rmax_gas=35.0, hz_disk=0.6, hz_bulge=0.3, hz_gas=0.3)
    if name == "Compact nuclear disk":
        return dict(M_disk=0.20e10, M_bulge=2.0e10, M_gas=0.02e10,
                    Rd_disk=0.3, Rmax_disk=2.0, a_bulge=0.2,
                    Rd_gas=0.5, Rmax_gas=2.5, hz_disk=0.15, hz_bulge=0.3, hz_gas=0.1)
    raise ValueError("Unknown preset")


def backend_is_interactive() -> bool:
    b = matplotlib.get_backend().lower()
    non_interactive = ('agg', 'pdf', 'ps', 'svg', 'cairo', 'template')
    return not any(b.startswith(x) for x in non_interactive)


def main():
    parser = argparse.ArgumentParser(description="Interactive toy galaxy rotation calculator — PLUS")
    parser.add_argument('--backend', type=str, default=None, help='Optional Matplotlib backend override (e.g., MacOSX, TkAgg, QtAgg, Agg).')
    parser.add_argument('--save', type=str, default=None, help='If set, save the main figure to this path and exit (headless mode).')

    config_names = ["MW-like disk", "Central-dominated", "Dwarf disk", "LMC-like dwarf", "Ultra-diffuse disk", "Compact nuclear disk"]
    parser.add_argument('--config', type=str, choices=config_names, default="MW-like disk", help='Initial configuration.')
    parser.add_argument('--seed', type=int, default=42, help='Initial RNG seed.')
    parser.add_argument('--spread', type=float, default=1.0, help='Initial spread scale.')
    parser.add_argument('--outer', type=float, default=0.95, help='Initial outer percentile (0.80–0.995).')
    parser.add_argument('--soften', type=float, default=0.2, help='Initial softening (kpc).')
    parser.add_argument('--gscale', type=float, default=1.0, help='Initial G scale.')
    parser.add_argument('--grk', type=float, default=0.0, help='Initial GR k (toy).')
    parser.add_argument('--atten', type=float, default=0.0, help='Initial extra attenuation (0–0.8).')
    parser.add_argument('--boost', type=float, default=1.0, help='Initial boost fraction.')
    parser.add_argument('--vobs', type=float, default=220.0, help='Initial observed v (km/s).')

    args = parser.parse_args()

    if args.backend:
        try:
            matplotlib.use(args.backend, force=True)
        except Exception as e:
            print(f"Warning: could not set backend {args.backend}: {e}")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, RadioButtons, Button

    # Wrapper to provide a slider-like interface over TextBox
    class TBWrapper:
        def __init__(self, tb, parse=float, clamp=None, default=None):
            self.tb = tb
            self.parse = parse
            self.clamp = clamp
            self.default = default
        @property
        def val(self):
            try:
                v = self.parse(self.tb.text)
            except Exception:
                v = self.default
            if self.clamp is not None and v is not None:
                lo, hi = self.clamp
                try:
                    v = max(lo, min(hi, v))
                except Exception:
                    v = self.default
            return v
        def set_val(self, v):
            self.tb.set_val(str(v))
        def on_changed(self, cb):
            # Mimic Slider.on_changed using TextBox.on_submit
            self.tb.on_submit(lambda _text: cb(self.val))

    # ----- Initialize galaxy -----
    pp = preset_params(args.config)
    pos, masses = make_galaxy_components(
        args.seed, args.spread,
        pp['M_disk'], pp['M_bulge'], pp['M_gas'],
        pp['Rd_disk'], pp['Rmax_disk'], pp['a_bulge'],
        pp['Rd_gas'], pp['Rmax_gas'], pp['hz_disk'], pp['hz_bulge'], pp['hz_gas']
    )
    if masses.sum() <= 0 or len(masses) == 0:
        COM = np.zeros(3)
        R_all = np.array([1.0])
    else:
        COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
        R_all = np.linalg.norm(pos[:, :2] - COM[:2], axis=1)
    R_edge = np.quantile(R_all, float(args.outer)) if len(R_all) else 1.0
    test_point = COM + np.array([R_edge, 0.0, 0.0])

    # ----- Figure and axes -----
    fig = plt.figure(figsize=(12, 9))
    # Narrow the main axes to make room for a table on the right
    ax = fig.add_axes([0.07, 0.32, 0.60, 0.66])
    ax.set_aspect('equal', 'box')
    galaxy_scatter = ax.scatter(pos[:, 0], pos[:, 1], s=5, alpha=0.6)
    test_star_plot, = ax.plot([test_point[0]], [test_point[1]], marker='*', markersize=10)
    com_plot, = ax.plot([COM[0]], [COM[1]], marker='x', markersize=8)
    ax.set_xlabel("x (kpc)"); ax.set_ylabel("y (kpc)")
    ax.set_title("Toy Galaxy Rotation Calculator — PLUS")
    text_box = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

    # Table showing Actual | GR Predicted | Simulation | Sim - Actual
    ax_table = fig.add_axes([0.70, 0.32, 0.25, 0.66])
    ax_table.axis('off')

    def update_table(v_obs, v_gr, v_final):
        ax_table.clear()
        ax_table.axis('off')
        colLabels = ["Actual", "GR Predicted", "Simulation", "Sim - Actual"]
        cellText = [[f"{v_obs:.2f}", f"{v_gr:.2f}", f"{v_final:.2f}", f"{(v_final - v_obs):+.2f}"]]
        tbl = ax_table.table(cellText=cellText, colLabels=colLabels, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.5)

    # Physics numeric inputs (tightened layout)
    ax_sG   = fig.add_axes([0.10, 0.27, 0.55, 0.024])
    ax_kGR  = fig.add_axes([0.10, 0.24, 0.55, 0.024])
    ax_att  = fig.add_axes([0.10, 0.21, 0.55, 0.024])
    ax_boost= fig.add_axes([0.10, 0.18, 0.55, 0.024])
    ax_vobs = fig.add_axes([0.10, 0.15, 0.55, 0.024])
    ax_soft = fig.add_axes([0.10, 0.12, 0.55, 0.024])

    tb_Gscale = TextBox(ax_sG,   'G scale',                 initial=str(float(args.gscale)))
    tb_kGR    = TextBox(ax_kGR,  'GR k (toy)',              initial=str(float(args.grk)))
    tb_att    = TextBox(ax_att,  'Extra atten (0-1)',       initial=str(float(args.atten)))
    tb_boost  = TextBox(ax_boost,'Boost fraction',          initial=str(float(args.boost)))
    tb_vobs   = TextBox(ax_vobs, 'Observed v (km/s)',       initial=str(float(args.vobs)))
    tb_soft   = TextBox(ax_soft, 'Softening (kpc)',         initial=str(float(args.soften)))
    # Hide advanced physics controls for a simpler UI
    ax_kGR.set_visible(False)
    ax_att.set_visible(False)
    ax_boost.set_visible(False)
    ax_soft.set_visible(False)

    s_Gscale = TBWrapper(tb_Gscale, parse=float, clamp=(0.1, 5.0),   default=float(args.gscale))
    s_kGR    = TBWrapper(tb_kGR,    parse=float, clamp=(0.0, 20000), default=float(args.grk))
    s_att    = TBWrapper(tb_att,    parse=float, clamp=(0.0, 0.8),   default=float(args.atten))
    s_boost  = TBWrapper(tb_boost,  parse=float, clamp=(0.0, 1.5),   default=float(args.boost))
    s_vobs   = TBWrapper(tb_vobs,   parse=float, clamp=(0.0, 450.0), default=float(args.vobs))
    s_soften = TBWrapper(tb_soft,   parse=float, clamp=(0.01, 1.5),  default=float(args.soften))

    # Structural numeric inputs (tightened layout)
    ax_spread = fig.add_axes([0.10, 0.09, 0.55, 0.024])
    ax_outer  = fig.add_axes([0.10, 0.06, 0.55, 0.024])
    ax_seed   = fig.add_axes([0.10, 0.03, 0.55, 0.024])
    tb_spread = TextBox(ax_spread, 'Spread ×',              initial=str(float(args.spread)))
    tb_outer  = TextBox(ax_outer,  'Outer percentile',      initial=str(float(args.outer)))
    tb_seed   = TextBox(ax_seed,   'Seed',                  initial=str(int(args.seed)))
    # Hide structural inputs from the UI; presets will control structure
    ax_spread.set_visible(False)
    ax_outer.set_visible(False)
    ax_seed.set_visible(False)

    s_spread = TBWrapper(tb_spread, parse=float, clamp=(0.5, 3.0),     default=float(args.spread))
    s_outer  = TBWrapper(tb_outer,  parse=float, clamp=(0.80, 0.995),  default=float(args.outer))
    s_seed   = TBWrapper(tb_seed,   parse=int,   clamp=(0, 9999),      default=int(args.seed))

    ax_radio = fig.add_axes([0.80, 0.08, 0.17, 0.20])
    radio = RadioButtons(ax_radio, config_names, active=config_names.index(args.config))

    # Component editor numeric inputs (tightened layout)
    ax_dm  = fig.add_axes([0.10,  0.00, 0.55, 0.024])
    ax_bm  = fig.add_axes([0.10, -0.03, 0.55, 0.024])
    ax_gm  = fig.add_axes([0.10, -0.06, 0.55, 0.024])
    ax_rd  = fig.add_axes([0.10, -0.09, 0.55, 0.024])
    ax_rmax= fig.add_axes([0.10, -0.12, 0.55, 0.024])
    ax_ab  = fig.add_axes([0.10, -0.15, 0.55, 0.024])
    ax_rdg = fig.add_axes([0.10, -0.18, 0.55, 0.024])
    ax_rmg = fig.add_axes([0.10, -0.21, 0.55, 0.024])
    ax_hzd = fig.add_axes([0.10, -0.24, 0.55, 0.024])
    ax_hzb = fig.add_axes([0.10, -0.27, 0.55, 0.024])
    ax_hzg = fig.add_axes([0.10, -0.30, 0.55, 0.024])

    tb_Mdisk = TextBox(ax_dm,   'Disk mass (1e10 Msun)',   initial=str(pp['M_disk']/1e10))
    tb_Mbulge= TextBox(ax_bm,   'Bulge mass (1e10 Msun)',  initial=str(pp['M_bulge']/1e10))
    tb_Mgas  = TextBox(ax_gm,   'Gas mass (1e10 Msun)',    initial=str(pp['M_gas']/1e10))
    tb_Rd    = TextBox(ax_rd,   'Disk Rd (kpc)',           initial=str(pp['Rd_disk']))
    tb_Rmax  = TextBox(ax_rmax, 'Disk Rmax (kpc)',         initial=str(pp['Rmax_disk']))
    tb_ab    = TextBox(ax_ab,   'Bulge a (kpc)',           initial=str(pp['a_bulge']))
    tb_Rdg   = TextBox(ax_rdg,  'Gas Rd (kpc)',            initial=str(pp['Rd_gas']))
    tb_Rmg   = TextBox(ax_rmg,  'Gas Rmax (kpc)',          initial=str(pp['Rmax_gas']))
    tb_hzd   = TextBox(ax_hzd,  'Disk hz (kpc)',           initial=str(pp['hz_disk']))
    tb_hzb   = TextBox(ax_hzb,  'Bulge hz (kpc)',          initial=str(pp['hz_bulge']))
    tb_hzg   = TextBox(ax_hzg,  'Gas hz (kpc)',            initial=str(pp['hz_gas']))
    # Hide component editor inputs; galaxy types will set these
    for ax_comp in [ax_dm, ax_bm, ax_gm, ax_rd, ax_rmax, ax_ab, ax_rdg, ax_rmg, ax_hzd, ax_hzb, ax_hzg]:
        ax_comp.set_visible(False)

    s_Mdisk = TBWrapper(tb_Mdisk, parse=float, clamp=(0.0, 8.0),   default=pp['M_disk']/1e10)
    s_Mbulge= TBWrapper(tb_Mbulge,parse=float, clamp=(0.0, 8.0),   default=pp['M_bulge']/1e10)
    s_Mgas  = TBWrapper(tb_Mgas,  parse=float, clamp=(0.0, 2.0),   default=pp['M_gas']/1e10)
    s_Rd    = TBWrapper(tb_Rd,    parse=float, clamp=(0.2, 10.0),  default=pp['Rd_disk'])
    s_Rmax  = TBWrapper(tb_Rmax,  parse=float, clamp=(2.0, 40.0),  default=pp['Rmax_disk'])
    s_ab    = TBWrapper(tb_ab,    parse=float, clamp=(0.05, 5.0),  default=pp['a_bulge'])
    s_Rdg   = TBWrapper(tb_Rdg,   parse=float, clamp=(0.2, 12.0),  default=pp['Rd_gas'])
    s_Rmg   = TBWrapper(tb_Rmg,   parse=float, clamp=(2.0, 45.0),  default=pp['Rmax_gas'])
    s_hzd   = TBWrapper(tb_hzd,   parse=float, clamp=(0.01, 1.5),  default=pp['hz_disk'])
    s_hzb   = TBWrapper(tb_hzb,   parse=float, clamp=(0.01, 1.5),  default=pp['hz_bulge'])
    s_hzg   = TBWrapper(tb_hzg,   parse=float, clamp=(0.01, 1.5),  default=pp['hz_gas'])

    # Defaults state for reset button (updated on preset change for component fields)

    ax_btn_curve = fig.add_axes([0.80, 0.03, 0.17, 0.035]); btn_curve = Button(ax_btn_curve, 'Plot Rotation Curve')
    ax_btn_save  = fig.add_axes([0.80, -0.02, 0.17, 0.035]); btn_save  = Button(ax_btn_save,  'Save Curve CSV')
    # Open vectors table window
    ax_btn_vectors = fig.add_axes([0.80, 0.56, 0.17, 0.035]); btn_vectors = Button(ax_btn_vectors, 'Force Vectors')
    # Reset to defaults button
    ax_btn_reset = fig.add_axes([0.80, 0.28, 0.17, 0.035]); btn_reset = Button(ax_btn_reset, 'Reset Defaults')
    
    fig_rot = None; rot_lines = {}; fig_vec = None; vec_state = {}
    def reset_defaults(_=None):
        s_Gscale.set_val(defaults['Gscale'])
        s_vobs.set_val(defaults['vobs'])
        s_DensK.set_val(defaults['DensK'])
        s_DensR.set_val(defaults['DensR'])
        s_WellK.set_val(defaults['WellK'])
        s_WellR.set_val(defaults['WellR'])
        s_seed.set_val(defaults['seed'])
        s_Nstars.set_val(defaults['Nstars'])
        s_Rdisk.set_val(defaults['Rdisk'])
        s_BulgeA.set_val(defaults['BulgeA'])
        s_Hz.set_val(defaults['Hz'])
        s_BulgeFrac.set_val(defaults['BulgeFrac'])
        s_Mtot1e10.set_val(defaults['Mtot1e10'])
        build_simple_galaxy()
        refresh_main()

    btn_reset.on_clicked(reset_defaults)

    # ----- Simple controls (right column) -----
    # Defaults for simple mode
    simple_defaults = dict(Nstars=4000, Rdisk=20.0, BulgeA=0.6, Hz=0.3, BulgeFrac=0.2, Mtot1e10=8.0, DensK=0.0, DensR=2.0, WellK=0.0, WellR=2.0)
    ax_sN  = fig.add_axes([0.80, 0.93, 0.17, 0.035]); tb_sN  = TextBox(ax_sN,  'N stars',        initial=str(simple_defaults['Nstars']))
    ax_sR  = fig.add_axes([0.80, 0.89, 0.17, 0.035]); tb_sR  = TextBox(ax_sR,  'R disk (kpc)',   initial=str(simple_defaults['Rdisk']))
    ax_sBa = fig.add_axes([0.80, 0.85, 0.17, 0.035]); tb_sBa = TextBox(ax_sBa, 'Bulge a (kpc)',  initial=str(simple_defaults['BulgeA']))
    ax_sHz = fig.add_axes([0.80, 0.81, 0.17, 0.035]); tb_sHz = TextBox(ax_sHz, 'Thickness hz',   initial=str(simple_defaults['Hz']))
    ax_sBf = fig.add_axes([0.80, 0.77, 0.17, 0.035]); tb_sBf = TextBox(ax_sBf, 'Bulge frac 0-1', initial=str(simple_defaults['BulgeFrac']))
    ax_sMt = fig.add_axes([0.80, 0.73, 0.17, 0.035]); tb_sMt = TextBox(ax_sMt, 'Total mass (1e10)', initial=str(simple_defaults['Mtot1e10']))
    ax_sDk = fig.add_axes([0.80, 0.69, 0.17, 0.035]); tb_sDk = TextBox(ax_sDk, 'Density k',      initial=str(simple_defaults['DensK']))
    ax_sDr = fig.add_axes([0.80, 0.65, 0.17, 0.035]); tb_sDr = TextBox(ax_sDr, 'Density R (kpc)',initial=str(simple_defaults['DensR']))
    ax_sWk = fig.add_axes([0.80, 0.62, 0.17, 0.035]); tb_sWk = TextBox(ax_sWk, 'Well k',         initial=str(simple_defaults['WellK']))
    ax_sWr = fig.add_axes([0.80, 0.58, 0.17, 0.035]); tb_sWr = TextBox(ax_sWr, 'Well R (kpc)',   initial=str(simple_defaults['WellR']))
    ax_sBuild = fig.add_axes([0.80, 0.61, 0.17, 0.035]); btn_sBuild = Button(ax_sBuild, 'Build Simple Galaxy')
    # Hide the explicit build button; presets will rebuild automatically
    ax_sBuild.set_visible(False)

    s_Nstars   = TBWrapper(tb_sN,  parse=int,   clamp=(8, 100000), default=simple_defaults['Nstars'])
    s_Rdisk    = TBWrapper(tb_sR,  parse=float, clamp=(0.5, 100.0), default=simple_defaults['Rdisk'])
    s_BulgeA   = TBWrapper(tb_sBa, parse=float, clamp=(0.02, 10.0), default=simple_defaults['BulgeA'])
    s_Hz       = TBWrapper(tb_sHz, parse=float, clamp=(0.01, 3.0),  default=simple_defaults['Hz'])
    s_BulgeFrac= TBWrapper(tb_sBf, parse=float, clamp=(0.0, 1.0),   default=simple_defaults['BulgeFrac'])
    s_Mtot1e10 = TBWrapper(tb_sMt, parse=float, clamp=(0.01, 200.0),default=simple_defaults['Mtot1e10'])
    s_DensK    = TBWrapper(tb_sDk, parse=float, clamp=(0.0, 5.0),   default=simple_defaults['DensK'])
    s_DensR    = TBWrapper(tb_sDr, parse=float, clamp=(0.05, 10.0), default=simple_defaults['DensR'])
    s_WellK    = TBWrapper(tb_sWk, parse=float, clamp=(0.0, 5.0),   default=simple_defaults['WellK'])
    s_WellR    = TBWrapper(tb_sWr, parse=float, clamp=(0.05, 10.0), default=simple_defaults['WellR'])

    # Defaults map used by Reset button
    defaults = {
        'Gscale': float(args.gscale), 'vobs': float(args.vobs),
        'DensK': simple_defaults['DensK'], 'DensR': simple_defaults['DensR'],
        'WellK': simple_defaults['WellK'], 'WellR': simple_defaults['WellR'],
        'seed': int(args.seed),
        'Nstars': simple_defaults['Nstars'], 'Rdisk': simple_defaults['Rdisk'], 'BulgeA': simple_defaults['BulgeA'], 'Hz': simple_defaults['Hz'], 'BulgeFrac': simple_defaults['BulgeFrac'], 'Mtot1e10': simple_defaults['Mtot1e10'],
    }

    def build_simple_galaxy(_evt=None):
        nonlocal pos, masses, COM, R_all, R_edge, test_point
        N_total = int(s_Nstars.val)
        R_disk = float(s_Rdisk.val)
        a_bulge = float(s_BulgeA.val)
        hz = float(s_Hz.val)
        f_bulge = float(s_BulgeFrac.val)
        N_bulge = max(0, min(N_total, int(round(N_total * f_bulge))))
        N_disk  = max(0, N_total - N_bulge)
        rng_seed = int(s_seed.val)
        rng = np.random.default_rng(rng_seed)
        pos_list = []; mass_list = []
        if N_disk > 0:
            N_base_d = max(1, N_disk // 8)
            Rd = max(0.05, R_disk / 3.0)
            pos_d = sample_exponential_disk_3d(N_base_d, Rd, R_disk, hz, rng)
            if len(pos_d): pos_list.append(pos_d)
        if N_bulge > 0:
            N_base_b = max(1, N_bulge // 8)
            Rmax_b = min(R_disk, max(2.0, 3.0*a_bulge))
            pos_b = sample_plummer_bulge_3d(N_base_b, a_bulge, Rmax_b, hz, rng)
            if len(pos_b): pos_list.append(pos_b)
        if pos_list:
            pos = np.vstack(pos_list)
            Mtot = float(s_Mtot1e10.val) * 1e10
            masses = np.full(len(pos), Mtot / len(pos))
        else:
            pos = np.zeros((0,3)); masses = np.zeros((0,))
        if masses.sum() > 0 and len(masses) > 0:
            COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
            R_all = np.linalg.norm(pos - COM, axis=1)
            test_point = pos[int(np.argmax(R_all))]
            R_edge = float(np.max(R_all))
        else:
            COM = np.zeros(3); R_all = np.array([1.0]); test_point = np.zeros(3); R_edge = 1.0
        # Update visuals without triggering advanced rebuild
        if len(masses) > 0:
            galaxy_scatter.set_offsets(pos[:, :2])
            test_star_plot.set_data([test_point[0]], [test_point[1]])
            com_plot.set_data([COM[0]], [COM[1]])
            R_plot = np.max(np.linalg.norm(pos[:, :2] - COM[:2], axis=1)) * 1.15 + 1e-3
            ax.set_xlim(COM[0] - R_plot, COM[0] + R_plot)
            ax.set_ylim(COM[1] - R_plot, COM[1] + R_plot)
        update_readout(); fig.canvas.draw_idle(); refresh_rotation_curve()

    btn_sBuild.on_clicked(build_simple_galaxy)

    # ---- Finder logic to match Simulation to Actual (Observed v) ----
    def _eval_error_for(wrapper, candidate, is_structural):
        """Return v_final - v_obs at candidate value for the given parameter.
        Does minimal recomputation; no drawing.
        """
        prev = wrapper.val
        # set
        wrapper.set_val(candidate)
        if is_structural:
            rebuild_galaxy()
        # compute
        v_newton, v_gr_noboost, v_final, drop_frac, boost_factor = compute_speeds_at_point()
        err = float(v_final - float(s_vobs.val))
        # restore
        wrapper.set_val(prev)
        if is_structural:
            rebuild_galaxy()
        return err

    def _solve_1d(wrapper, is_structural):
        clamp = wrapper.clamp if wrapper.clamp is not None else (None, None)
        lo, hi = clamp
        if lo is None or hi is None:
            # If no clamp, try a reasonable numeric window
            lo, hi = -1e3, 1e3
        is_int = (wrapper.parse is int)
        # helper to coerce to valid domain
        def coerce(x):
            if is_int:
                x = int(round(x))
            x = max(lo, min(hi, x))
            return x
        # Bisection if sign change exists, else coarse search and refine
        e_lo = _eval_error_for(wrapper, coerce(lo), is_structural)
        e_hi = _eval_error_for(wrapper, coerce(hi), is_structural)
        if np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo*e_hi < 0:
            a, b = coerce(lo), coerce(hi)
            for _ in range(40):
                m = coerce(0.5*(a+b))
                e_m = _eval_error_for(wrapper, m, is_structural)
                if abs(e_m) < 0.05:  # within 0.05 km/s
                    return m
                if e_lo * e_m <= 0:
                    b, e_hi = m, e_m
                else:
                    a, e_lo = m, e_m
            return coerce(0.5*(a+b))
        # Coarse grid search
        best_x, best_e = None, float('inf')
        N = 15
        for i in range(N+1):
            x = coerce(lo + (hi-lo)*i/N)
            e = _eval_error_for(wrapper, x, is_structural)
            if abs(e) < best_e:
                best_x, best_e = x, abs(e)
        # Local refine around best
        span = max(1 if is_int else 0.0, 0.1*(hi-lo))
        a, b = coerce(best_x - span), coerce(best_x + span)
        e_lo = _eval_error_for(wrapper, a, is_structural)
        e_hi = _eval_error_for(wrapper, b, is_structural)
        if np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo*e_hi < 0:
            # refine by bisection
            for _ in range(30):
                m = coerce(0.5*(a+b))
                e_m = _eval_error_for(wrapper, m, is_structural)
                if abs(e_m) < 0.05:
                    return m
                if e_lo * e_m <= 0:
                    b, e_hi = m, e_m
                else:
                    a, e_lo = m, e_m
            return coerce(0.5*(a+b))
        return best_x

    def _add_finder(next_to_ax, wrapper, label, is_structural):
        # place a small button to the right of the given TextBox axis
        bb = next_to_ax.get_position()
        btn_ax = fig.add_axes([bb.x1 + 0.01, bb.y0, 0.05, bb.height])
        btn = Button(btn_ax, 'Find')
        def _on_click(_event=None):
            sol = _solve_1d(wrapper, is_structural)
            if sol is not None:
                wrapper.set_val(sol)
                refresh_main()
        btn.on_clicked(_on_click)
        return btn

    # Finder buttons: keep only for global G and density/well knobs
    _add_finder(ax_sG,   s_Gscale, 'Find', False)
    _add_finder(ax_sDk,  s_DensK,  'Find', False)
    _add_finder(ax_sWk,  s_WellK,  'Find', False)

    fig_rot = None; rot_lines = {}; fig_vec = None; vec_state = {}

    def rebuild_galaxy():
        nonlocal pos, masses, COM, R_all, R_edge, test_point
        params = dict(
            M_disk=float(s_Mdisk.val)*1e10, M_bulge=float(s_Mbulge.val)*1e10, M_gas=float(s_Mgas.val)*1e10,
            Rd_disk=float(s_Rd.val), Rmax_disk=float(s_Rmax.val), a_bulge=float(s_ab.val),
            Rd_gas=float(s_Rdg.val), Rmax_gas=float(s_Rmg.val),
            hz_disk=float(s_hzd.val), hz_bulge=float(s_hzb.val), hz_gas=float(s_hzg.val))
        pos, masses = make_galaxy_components(int(s_seed.val), float(s_spread.val), **params)
        if masses.sum() <= 0 or len(masses) == 0:
            COM = np.zeros(3); R_all = np.array([1.0]); test_point = np.zeros(3)
        else:
            COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
            R_all = np.linalg.norm(pos - COM, axis=1)
            test_point = pos[int(np.argmax(R_all))]
        R_edge = np.max(R_all) if len(R_all) else 1.0

    def compute_G_eff(p):
        # Base G scale
        G0 = G_AST * float(s_Gscale.val)
        # Density-based boost: higher when local density is lower than a global reference
        try:
            k = float(s_DensK.val)
            Rn = max(1e-6, float(s_DensR.val))
        except Exception:
            k = 0.0; Rn = 1.0
        if k <= 0 or len(masses) == 0:
            return G0
        # Local density around point p
        d2 = np.sum((pos - p)**2, axis=1)
        mask = d2 <= (Rn*Rn)
        m_local = float(masses[mask].sum())
        vol = (4.0/3.0) * np.pi * (Rn**3)
        rho_local = m_local / max(1e-30, vol)
        # Global reference density using farthest-star radius as a characteristic scale
        if len(masses) > 0:
            Rchar = float(np.max(np.linalg.norm(pos - COM, axis=1)))
            Vchar = (4.0/3.0) * np.pi * max(1e-30, Rchar**3)
            rho_ref = float(masses.sum()) / Vchar
        else:
            rho_ref = rho_local
        boost = 0.0
        if rho_ref > 0:
            boost = k * max(0.0, (rho_ref / max(rho_local, 1e-30)) - 1.0)
        mult = max(0.1, min(10.0, 1.0 + boost))
        return G0 * mult

    def compute_well_scales():
        # Per-source gravity well scaling based on local shallowness
        try:
            k = float(s_WellK.val)
            Rw = max(1e-6, float(s_WellR.val))
        except Exception:
            k = 0.0; Rw = 1.0
        if k <= 0 or len(masses) == 0:
            return np.ones(len(masses))
        # Reference density as before
        Rchar = float(np.max(np.linalg.norm(pos - COM, axis=1))) if len(masses) else 1.0
        Vchar = (4.0/3.0) * np.pi * max(1e-30, Rchar**3)
        rho_ref = float(masses.sum()) / Vchar if Vchar > 0 else 0.0
        scales = np.ones(len(masses))
        # Compute local densities per source (approximate)
        # If N is modest, do exact; else sample for speed
        N = len(masses)
        if N <= 4000:
            # distance matrix may be large but acceptable
            # Compute squared distances in chunks for memory safety
            batch = 1000
            for i0 in range(0, N, batch):
                i1 = min(N, i0 + batch)
                d2 = np.sum((pos[i0:i1, None, :] - pos[None, :, :])**2, axis=2)
                mask = d2 <= (Rw*Rw)
                m_local = mask @ masses
                vol = (4.0/3.0) * np.pi * (Rw**3)
                rho_local = m_local / max(1e-30, vol)
                boost = np.zeros_like(rho_local)
                if rho_ref > 0:
                    boost = k * np.maximum(0.0, (rho_ref / np.maximum(rho_local, 1e-30)) - 1.0)
                scales[i0:i1] = np.clip(1.0 + boost, 0.1, 10.0)
        else:
            # Approximate by sampling neighbors
            idx = np.arange(N)
            for i in range(N):
                # sample 300 random indices
                samp = np.random.default_rng(int(s_seed.val)).choice(idx, size=min(300, N), replace=False)
                d2 = np.sum((pos[samp] - pos[i])**2, axis=1)
                m_local = masses[samp][d2 <= (Rw*Rw)].sum() * (N / max(1, len(samp)))
                vol = (4.0/3.0) * np.pi * (Rw**3)
                rho_local = m_local / max(1e-30, vol)
                boost = k * max(0.0, (rho_ref / max(rho_local, 1e-30)) - 1.0) if rho_ref > 0 else 0.0
                scales[i] = max(0.1, min(10.0, 1.0 + boost))
        return scales

    def compute_speeds_at_point():
        # Per-body vectors with density-at-point and well scaling
        G_point = compute_G_eff(test_point)
        # Base per-body construction (no GR attenuation or extra) computed directly
        if len(masses) == 0:
            return 0.0, 0.0, 0.0, 0.0, 1.0
        r_vec = pos - test_point
        r2 = np.sum(r_vec*r_vec, axis=1)
        eps2 = (float(s_soften.val)**2)
        inv_r3 = 1.0 / np.power(r2 + eps2, 1.5)
        base = (masses * inv_r3)
        well_sc = compute_well_scales()
        # Newtonian and GR vectors (with any GR attenuation kept at zero by default)
        newton_vecs = (G_point * well_sc)[:, None] * (r_vec * base[:, None])
        # Keep GR k and extra atten at defaults (hidden UI), but compute path anyway for clarity
        kGR = 0.0
        atten = 0.0
        r = np.sqrt(r2 + eps2)
        per_body_scale = (1.0 - kGR * (G_point / max(1e-30, G_AST)) * G_AST * masses / (r * c_kms**2))
        per_body_scale = np.clip(per_body_scale, 0.0, 1.0) * (1.0 - np.clip(atten, 0.0, 0.999))
        gr_vecs = (G_point * well_sc)[:, None] * (r_vec * ((base * per_body_scale)[:, None]))
        a_newton_vec = newton_vecs.sum(axis=0)
        a_gr_vec = gr_vecs.sum(axis=0)
        # No auto-boost by default
        boost_factor = 1.0
        a_final_vec = a_gr_vec * boost_factor
        v_newton = circular_speed_from_accel(a_newton_vec, test_point - COM)
        v_gr_noboost = circular_speed_from_accel(a_gr_vec, test_point - COM)
        v_final = circular_speed_from_accel(a_final_vec, test_point - COM)
        drop_frac = 0.0
        return v_newton, v_gr_noboost, v_final, drop_frac, boost_factor

    def compute_per_body_vectors():
        G_eff_base = compute_G_eff(test_point)
        if len(masses) == 0:
            return {
                'newton_vecs': np.zeros((0,3)),
                'gr_vecs': np.zeros((0,3)),
                'final_vecs': np.zeros((0,3)),
                'r_vec': np.zeros((0,3)),
                'r': np.zeros((0,)),
                'masses': np.zeros((0,)),
                'boost_factor': 1.0,
            }
        r_vec = pos - test_point
        r2 = np.sum(r_vec*r_vec, axis=1)
        eps2 = (float(s_soften.val)**2)
        inv_r3 = 1.0 / np.power(r2 + eps2, 1.5)
        base = (masses * inv_r3)
        well_sc = compute_well_scales()
        newton_vecs = (G_eff_base * well_sc)[:, None] * (r_vec * base[:, None])
        r = np.sqrt(r2 + eps2)
        per_body_scale = (1.0 - float(s_kGR.val) * (G_eff_base / max(1e-30, G_AST)) * G_AST * masses / (r * c_kms**2))
        per_body_scale = np.clip(per_body_scale, 0.0, 1.0)
        per_body_scale *= (1.0 - np.clip(float(s_att.val), 0.0, 0.999))
        gr_vecs = (G_eff_base * well_sc)[:, None] * (r_vec * ((base * per_body_scale)[:, None]))
        a0 = np.linalg.norm(newton_vecs.sum(axis=0))
        agr = np.linalg.norm(gr_vecs.sum(axis=0))
        drop_frac = 0.0
        if a0 > 0:
            drop_frac = max(0.0, 1.0 - (agr/a0))
        f_boost = float(s_boost.val); boost_factor = 1.0
        if f_boost > 0 and drop_frac > 0 and (1.0 - f_boost*drop_frac) > 1e-9:
            boost_factor = 1.0 / (1.0 - f_boost*drop_frac)
        final_vecs = gr_vecs * boost_factor
        return {
            'newton_vecs': newton_vecs,
            'gr_vecs': gr_vecs,
            'final_vecs': final_vecs,
            'r_vec': r_vec,
            'r': r,
            'masses': masses.copy(),
            'boost_factor': boost_factor,
        }

    def rotation_curve(nR: int = 50):
        if len(masses) == 0:
            return np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1)
        Rmax = float(np.max(np.linalg.norm(pos - COM, axis=1)))
        R_vals = np.linspace(max(0.2, 0.02*Rmax), Rmax, nR)
        p_edge = COM + np.array([Rmax, 0, 0])
        G_eff_edge = compute_G_eff(p_edge)
        a_gr_edge, a_newt_edge = net_acceleration_at_point(
            p_edge, pos, masses, G_eff_edge, k_GR=0.0, atten_extra=0.0, soften_kpc=float(s_soften.val))
        drop_edge = 0.0
        if np.linalg.norm(a_newt_edge) > 0:
            drop_edge = max(0.0, 1.0 - np.linalg.norm(a_gr_edge)/np.linalg.norm(a_newt_edge))
        f_boost = float(s_boost.val); boost_factor = 1.0
        if f_boost > 0 and drop_edge > 0 and (1.0 - f_boost*drop_edge) > 1e-9:
            boost_factor = 1.0 / (1.0 - f_boost*drop_edge)
        vN, vGR, vF = [], [], []
        for R in R_vals:
            p = COM + np.array([R, 0, 0])
            G_eff_p = compute_G_eff(p)
            # Per-body with well scaling
            r_vec = pos - p
            r2 = np.sum(r_vec*r_vec, axis=1)
            inv_r3 = 1.0 / np.power(r2 + float(s_soften.val)**2, 1.5)
            base = (masses * inv_r3)
            well_sc = compute_well_scales()
            # Newtonian and GR-like (no extra attenuation)
            newton_vecs = (G_eff_p * well_sc)[:, None] * (r_vec * base[:, None])
            a_newton_vec = newton_vecs.sum(axis=0)
            gr_vecs = (G_eff_p * well_sc)[:, None] * (r_vec * (base[:, None]))
            a_gr_vec = gr_vecs.sum(axis=0)
            vN.append(circular_speed_from_accel(a_newton_vec, p - COM))
            vGR.append(circular_speed_from_accel(a_gr_vec, p - COM))
            vF.append(circular_speed_from_accel(a_gr_vec * boost_factor, p - COM))
        return R_vals, np.array(vN), np.array(vGR), np.array(vF)

    def update_readout():
        v_newton, v_gr_noboost, v_final, drop_frac, boost_factor = compute_speeds_at_point()
        v_obs = float(s_vobs.val); dv = v_final - v_obs
        if len(masses) > 0:
            R_plot = np.max(np.linalg.norm(pos[:, :2] - COM[:2], axis=1)) * 1.15 + 1e-3
        else:
            R_plot = 1.0
        ax.set_xlim(COM[0] - R_plot, COM[0] + R_plot)
        ax.set_ylim(COM[1] - R_plot, COM[1] + R_plot)
        total_mass = masses.sum()
        far_r = float(np.linalg.norm(test_point[:2] - COM[:2]))
        lines = [
            f"Preset: {radio.value_selected} | N stars: {len(masses)} | Total mass: {total_mass:,.2e} Msun",
            f"Farthest-star radius: R = {far_r:.2f} kpc",
            f"G scale: {s_Gscale.val:.2f} | Density k: {s_DensK.val:.2f} @ R={s_DensR.val:.2f} kpc | Well k: {s_WellK.val:.2f} @ R={s_WellR.val:.2f} kpc",
            f"Speeds (km/s): Newtonian={v_newton:.2f}  GR-like={v_gr_noboost:.2f}  Final={v_final:.2f}",
            f"Observed target: {v_obs:.1f} km/s | Final − Observed = {dv:+.2f} km/s"
        ]
        text_box.set_text('\n'.join(lines))
        # Update side table for quick readability
        update_table(v_obs, v_gr_noboost, v_final)

    def refresh_main(_=None):
        rebuild_galaxy()
        if len(masses) > 0:
            galaxy_scatter.set_offsets(pos[:, :2])
        else:
            galaxy_scatter.set_offsets(np.empty((0, 2)))
        test_star_plot.set_data([test_point[0]], [test_point[1]])
        com_plot.set_data([COM[0]], [COM[1]])
        update_readout()
        fig.canvas.draw_idle()
        refresh_rotation_curve()

    def apply_preset(name: str):
        # Simple galaxy defaults per type
        if name == "MW-like disk":
            p = dict(Nstars=8000, Rdisk=20.0, BulgeA=0.6, Hz=0.3, BulgeFrac=0.2, Mtot1e10=8.0)
        elif name == "Central-dominated":
            p = dict(Nstars=6000, Rdisk=15.0, BulgeA=0.8, Hz=0.5, BulgeFrac=0.8, Mtot1e10=9.5)
        elif name == "Dwarf disk":
            p = dict(Nstars=3000, Rdisk=6.0,  BulgeA=0.3, Hz=0.2, BulgeFrac=0.0, Mtot1e10=0.45)
        elif name == "LMC-like dwarf":
            p = dict(Nstars=2500, Rdisk=9.0,  BulgeA=0.2, Hz=0.25, BulgeFrac=0.0, Mtot1e10=0.40)
        elif name == "Ultra-diffuse disk":
            p = dict(Nstars=2000, Rdisk=30.0, BulgeA=0.3, Hz=0.6, BulgeFrac=0.0, Mtot1e10=0.07)
        elif name == "Compact nuclear disk":
            p = dict(Nstars=5000, Rdisk=2.0,  BulgeA=0.2, Hz=0.3, BulgeFrac=0.9, Mtot1e10=2.2)
        else:
            p = dict(Nstars=s_Nstars.val, Rdisk=s_Rdisk.val, BulgeA=s_BulgeA.val, Hz=s_Hz.val, BulgeFrac=s_BulgeFrac.val, Mtot1e10=s_Mtot1e10.val)
        s_Nstars.set_val(p['Nstars'])
        s_Rdisk.set_val(p['Rdisk'])
        s_BulgeA.set_val(p['BulgeA'])
        s_Hz.set_val(p['Hz'])
        s_BulgeFrac.set_val(p['BulgeFrac'])
        s_Mtot1e10.set_val(p['Mtot1e10'])
        # Update defaults
        defaults.update(p)
        build_simple_galaxy()

    def on_preset_clicked(label: str):
        apply_preset(label)

    def show_rotation_curve(_=None):
        nonlocal fig_rot, rot_lines
        if fig_rot is None:
            fig_rot = plt.figure(figsize=(7, 5))
            ax2 = fig_rot.add_axes([0.12, 0.12, 0.85, 0.85])
            ax2.set_xlabel("R (kpc)"); ax2.set_ylabel("v_circ (km/s)")
            ax2.set_title("Rotation Curve")
            R, vN, vGR, vF = rotation_curve()
            lnN, = ax2.plot(R, vN, lw=1.5, label="Newtonian")
            lnG, = ax2.plot(R, vGR, lw=1.5, label="GR,no boost")
            lnF, = ax2.plot(R, vF, lw=2.0, label="Final (boosted)")
            ax2.legend(loc="best")
            rot_lines = {'ax': ax2, 'N': lnN, 'GR': lnG, 'F': lnF}
            fig_rot.canvas.draw_idle()
        else:
            refresh_rotation_curve()
        plt.show()

    def refresh_rotation_curve():
        if rot_lines:
            R, vN, vGR, vF = rotation_curve()
            rot_lines['N'].set_data(R, vN)
            rot_lines['GR'].set_data(R, vGR)
            rot_lines['F'].set_data(R, vF)
            ax2 = rot_lines['ax']
            if len(R) > 0:
                ax2.set_xlim(R.min(), R.max()*1.02)
                vmax = max(1.0, vN.max() if len(vN) else 1.0, vGR.max() if len(vGR) else 1.0, vF.max() if len(vF) else 1.0)
                ax2.set_ylim(0.0, vmax*1.1)
            fig_rot.canvas.draw_idle()

    # ----- Force vectors window -----
    def show_vectors_table(_=None):
        nonlocal fig_vec, vec_state
        data = compute_per_body_vectors()
        mags = np.linalg.norm(data['final_vecs'], axis=1)
        order = np.argsort(mags)[::-1]
        vec_state = {
            'order': order,
            'data': data,
            'topN': 25,
        }
        def build_table(topN: int):
            d = vec_state['data']; ordidx = vec_state['order']
            topN = int(max(1, min(len(ordidx), topN)))
            idx = ordidx[:topN]
            # Build rows
            rows = []
            for j, i in enumerate(idx):
                dx, dy, dz = d['r_vec'][i]
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                ax, ay, az = d['final_vecs'][i]
                rows.append([
                    str(int(i)), f"{dx:.3f}", f"{dy:.3f}", f"{dz:.3f}", f"{r:.3f}", f"{d['masses'][i]:.3e}",
                    f"{ax:.3e}", f"{ay:.3e}", f"{az:.3e}", f"{np.sqrt(ax*ax+ay*ay+az*az):.3e}"
                ])
            # Summary (vector sum)
            sum_vec = d['final_vecs'].sum(axis=0)
            sum_mag = float(np.linalg.norm(sum_vec))
            return rows, sum_vec, sum_mag
        if fig_vec is None:
            fig_vec = plt.figure(figsize=(11, 7))
            axT = fig_vec.add_axes([0.05, 0.15, 0.90, 0.80])
            axT.axis('off')
            axN = fig_vec.add_axes([0.05, 0.05, 0.12, 0.06])
            tb_top = TextBox(axN, 'Top N', initial=str(vec_state['topN']))
            ax_save = fig_vec.add_axes([0.20, 0.05, 0.18, 0.06]); btn_save_vec = Button(ax_save, 'Save Vectors CSV')
            ax_close= fig_vec.add_axes([0.40, 0.05, 0.12, 0.06]); btn_close = Button(ax_close, 'Close')
            colLabels = ["i", "dx", "dy", "dz", "r_kpc", "mass_Msun", "ax", "ay", "az", "|a|"]
            def render():
                axT.clear(); axT.axis('off')
                rows, sum_vec, sum_mag = build_table(vec_state['topN'])
                tbl = axT.table(cellText=rows, colLabels=colLabels, loc='center')
                tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.2)
                # Footer text with vector sum
                axT.text(0.5, -0.08, f"Sum(final per-body) = ({sum_vec[0]:.3e}, {sum_vec[1]:.3e}, {sum_vec[2]:.3e}), |sum|={sum_mag:.3e}",
                         transform=axT.transAxes, ha='center', va='top')
                fig_vec.canvas.draw_idle()
            def on_top_submit(_text):
                try:
                    vec_state['topN'] = int(max(1, min(len(vec_state['order']), float(tb_top.text))))
                except Exception:
                    vec_state['topN'] = 25
                render()
            def on_save(_evt=None):
                out = Path('force_vectors.csv')
                import csv
                d = vec_state['data']
                with out.open('w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["i","pos_x","pos_y","pos_z","dx","dy","dz","r_kpc","mass_Msun",
                                "newton_ax","newton_ay","newton_az","gr_ax","gr_ay","gr_az","final_ax","final_ay","final_az","final_mag"])
                    for i in range(len(d['masses'])):
                        dx, dy, dz = d['r_vec'][i]
                        r = d['r'][i]
                        nx, ny, nz = d['newton_vecs'][i]
                        gx, gy, gz = d['gr_vecs'][i]
                        fx, fy, fz = d['final_vecs'][i]
                        w.writerow([int(i), float(pos[i,0]), float(pos[i,1]), float(pos[i,2]),
                                    float(dx), float(dy), float(dz), float(r), float(d['masses'][i]),
                                    float(nx), float(ny), float(nz), float(gx), float(gy), float(gz), float(fx), float(fy), float(fz),
                                    float(np.sqrt(fx*fx+fy*fy+fz*fz))])
                print(f"Saved force vectors to {out.resolve()}")
            def on_close(_evt=None):
                nonlocal fig_vec
                plt.close(fig_vec)
                # reset to allow reopening
                fig_vec = None
            tb_top.on_submit(on_top_submit)
            btn_save_vec.on_clicked(on_save)
            btn_close.on_clicked(on_close)
            render()
        else:
            # If already open, just rerender with current state
            fig_vec.canvas.manager.set_window_title('Force Vectors') if hasattr(fig_vec.canvas.manager, 'set_window_title') else None
            vec_state['data'] = compute_per_body_vectors()
            vec_state['order'] = np.argsort(np.linalg.norm(vec_state['data']['final_vecs'], axis=1))[::-1]
            # Rebuild content by re-calling show
            plt.figure(fig_vec.number)
        plt.show()

    def save_curve_csv(_=None):
        R, vN, vGR, vF = rotation_curve()
        out = Path('rotation_curve.csv')
        import csv
        with out.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["R_kpc", "v_newton_kms", "v_gr_no_boost_kms", "v_final_kms",
                        "G_scale", "GR_k_toy", "extra_atten", "boost_fraction", "softening_kpc"])
            for i in range(len(R)):
                w.writerow([float(R[i]), float(vN[i]), float(vGR[i]), float(vF[i]),
                            float(s_Gscale.val), float(s_kGR.val), float(s_att.val),
                            float(s_boost.val), float(s_soften.val)])
        print(f"Saved rotation curve to {out.resolve()}")

    # Wire up callbacks
    for s in [s_Gscale, s_kGR, s_att, s_boost, s_vobs, s_soften,
              s_spread, s_outer, s_seed,
              s_Mdisk, s_Mbulge, s_Mgas, s_Rd, s_Rmax, s_ab, s_Rdg, s_Rmg, s_hzd, s_hzb, s_hzg]:
        s.on_changed(refresh_main)
    radio.on_clicked(on_preset_clicked)
    btn_curve.on_clicked(show_rotation_curve)
    btn_save.on_clicked(save_curve_csv)
    btn_vectors.on_clicked(show_vectors_table)

    # Initial draw
    refresh_main()

    # Headless save or interactive show
    if args.save:
        update_readout()
        out_path = os.path.abspath(args.save)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved main figure to {out_path}")
        return

    if not backend_is_interactive():
        out_path = os.path.abspath('rotation_preview.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Non-interactive backend detected; saved main figure to {out_path}")
        return

    plt.show()


if __name__ == "__main__":
    main()

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
    defaults = {
        'Gscale': float(args.gscale), 'kGR': float(args.grk), 'atten': float(args.atten), 'boost': float(args.boost), 'vobs': float(args.vobs), 'soften': float(args.soften),
        'spread': float(args.spread), 'outer': float(args.outer), 'seed': int(args.seed),
        'Mdisk': pp['M_disk']/1e10, 'Mbulge': pp['M_bulge']/1e10, 'Mgas': pp['M_gas']/1e10,
        'Rd': pp['Rd_disk'], 'Rmax': pp['Rmax_disk'], 'ab': pp['a_bulge'], 'Rdg': pp['Rd_gas'], 'Rmg': pp['Rmax_gas'], 'hzd': pp['hz_disk'], 'hzb': pp['hz_bulge'], 'hzg': pp['hz_gas'],
    }

    ax_btn_curve = fig.add_axes([0.80, 0.03, 0.17, 0.035]); btn_curve = Button(ax_btn_curve, 'Plot Rotation Curve')
    ax_btn_save  = fig.add_axes([0.80, -0.02, 0.17, 0.035]); btn_save  = Button(ax_btn_save,  'Save Curve CSV')
    ax_btn_reset = fig.add_axes([0.80, 0.28, 0.17, 0.035]); btn_reset = Button(ax_btn_reset, 'Reset Defaults')

    def reset_defaults(_=None):
        s_Gscale.set_val(defaults['Gscale'])
        s_kGR.set_val(defaults['kGR'])
        s_att.set_val(defaults['atten'])
        s_boost.set_val(defaults['boost'])
        s_vobs.set_val(defaults['vobs'])
        s_soften.set_val(defaults['soften'])
        s_spread.set_val(defaults['spread'])
        s_outer.set_val(defaults['outer'])
        s_seed.set_val(defaults['seed'])
        s_Mdisk.set_val(defaults['Mdisk'])
        s_Mbulge.set_val(defaults['Mbulge'])
        s_Mgas.set_val(defaults['Mgas'])
        s_Rd.set_val(defaults['Rd'])
        s_Rmax.set_val(defaults['Rmax'])
        s_ab.set_val(defaults['ab'])
        s_Rdg.set_val(defaults['Rdg'])
        s_Rmg.set_val(defaults['Rmg'])
        s_hzd.set_val(defaults['hzd'])
        s_hzb.set_val(defaults['hzb'])
        s_hzg.set_val(defaults['hzg'])
        refresh_main()

    btn_reset.on_clicked(reset_defaults)

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

    # Finder buttons for physics (skip Observed v; it's the target itself)
    _add_finder(ax_sG,   s_Gscale, 'Find', False)
    _add_finder(ax_kGR,  s_kGR,    'Find', False)
    _add_finder(ax_att,  s_att,    'Find', False)
    _add_finder(ax_boost,s_boost,  'Find', False)
    _add_finder(ax_soft, s_soften, 'Find', False)

    # Finder buttons for structural
    _add_finder(ax_spread, s_spread, 'Find', True)
    _add_finder(ax_outer,  s_outer,  'Find', True)
    _add_finder(ax_seed,   s_seed,   'Find', True)

    # Finder buttons for components
    _add_finder(ax_dm,   s_Mdisk, 'Find', True)
    _add_finder(ax_bm,   s_Mbulge,'Find', True)
    _add_finder(ax_gm,   s_Mgas,  'Find', True)
    _add_finder(ax_rd,   s_Rd,    'Find', True)
    _add_finder(ax_rmax, s_Rmax,  'Find', True)
    _add_finder(ax_ab,   s_ab,    'Find', True)
    _add_finder(ax_rdg,  s_Rdg,   'Find', True)
    _add_finder(ax_rmg,  s_Rmg,   'Find', True)
    _add_finder(ax_hzd,  s_hzd,   'Find', True)
    _add_finder(ax_hzb,  s_hzb,   'Find', True)
    _add_finder(ax_hzg,  s_hzg,   'Find', True)

    fig_rot = None; rot_lines = {}

    def rebuild_galaxy():
        nonlocal pos, masses, COM, R_all, R_edge, test_point
        params = dict(
            M_disk=float(s_Mdisk.val)*1e10, M_bulge=float(s_Mbulge.val)*1e10, M_gas=float(s_Mgas.val)*1e10,
            Rd_disk=float(s_Rd.val), Rmax_disk=float(s_Rmax.val), a_bulge=float(s_ab.val),
            Rd_gas=float(s_Rdg.val), Rmax_gas=float(s_Rmg.val),
            hz_disk=float(s_hzd.val), hz_bulge=float(s_hzb.val), hz_gas=float(s_hzg.val))
        pos, masses = make_galaxy_components(int(s_seed.val), float(s_spread.val), **params)
        if masses.sum() <= 0 or len(masses) == 0:
            COM = np.zeros(3); R_all = np.array([1.0])
        else:
            COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
            R_all = np.linalg.norm(pos[:, :2] - COM[:2], axis=1)
        R_edge = np.quantile(R_all, float(s_outer.val)) if len(R_all) else 1.0
        test_point = COM + np.array([R_edge, 0.0, 0.0])

    def compute_speeds_at_point():
        G_eff_base = G_AST * float(s_Gscale.val)
        a_gr_vec, a_newton_vec = net_acceleration_at_point(
            test_point, pos, masses, G_eff_base, k_GR=float(s_kGR.val), atten_extra=float(s_att.val), soften_kpc=float(s_soften.val))
        a0 = np.linalg.norm(a_newton_vec); agr = np.linalg.norm(a_gr_vec)
        drop_frac = 0.0
        if a0 > 0:
            drop_frac = max(0.0, 1.0 - (agr/a0))
        f_boost = float(s_boost.val); boost_factor = 1.0
        if f_boost > 0 and drop_frac > 0 and (1.0 - f_boost*drop_frac) > 1e-9:
            boost_factor = 1.0 / (1.0 - f_boost*drop_frac)
        a_final_vec = a_gr_vec * boost_factor
        v_newton = circular_speed_from_accel(a_newton_vec, test_point - COM)
        v_gr_noboost = circular_speed_from_accel(a_gr_vec, test_point - COM)
        v_final = circular_speed_from_accel(a_final_vec, test_point - COM)
        return v_newton, v_gr_noboost, v_final, drop_frac, boost_factor

    def rotation_curve(nR: int = 50):
        if len(masses) == 0:
            return np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1)
        Rmax = np.quantile(np.linalg.norm(pos[:, :2] - COM[:2], axis=1), float(s_outer.val))
        R_vals = np.linspace(max(0.2, 0.02*Rmax), Rmax, nR)
        G_eff_base = G_AST * float(s_Gscale.val)
        a_gr_edge, a_newt_edge = net_acceleration_at_point(
            COM + np.array([Rmax, 0, 0]), pos, masses, G_eff_base, k_GR=float(s_kGR.val), atten_extra=float(s_att.val), soften_kpc=float(s_soften.val))
        drop_edge = 0.0
        if np.linalg.norm(a_newt_edge) > 0:
            drop_edge = max(0.0, 1.0 - np.linalg.norm(a_gr_edge)/np.linalg.norm(a_newt_edge))
        f_boost = float(s_boost.val); boost_factor = 1.0
        if f_boost > 0 and drop_edge > 0 and (1.0 - f_boost*drop_edge) > 1e-9:
            boost_factor = 1.0 / (1.0 - f_boost*drop_edge)
        vN, vGR, vF = [], [], []
        for R in R_vals:
            p = COM + np.array([R, 0, 0])
            a_gr_vec, a_newton_vec = net_acceleration_at_point(
                p, pos, masses, G_eff_base, k_GR=float(s_kGR.val), atten_extra=float(s_att.val), soften_kpc=float(s_soften.val))
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
        lines = [
            f"Preset: {radio.value_selected} | N lumps: {len(masses)} | Total luminous+gas mass: {total_mass:,.2e} Msun",
            f"Outer-edge radius (pctl {s_outer.val:.3f}): R = {np.linalg.norm(test_point[:2] - COM[:2]):.2f} kpc",
            f"G scale: {s_Gscale.val:.2f} | GR k (toy): {s_kGR.val:.0f} | Extra atten: {s_att.val:.2f} | Softening: {s_soften.val:.2f} kpc",
            f"Drop from attenuation at edge: {100*drop_frac:.4f}% | Auto-boost factor: ×{boost_factor:.5f}",
            f"Speeds (km/s): Newtonian={v_newton:.2f}  GR,no boost={v_gr_noboost:.2f}  Final (with boost)={v_final:.2f}",
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
        params = preset_params(name)
        s_Mdisk.set_val(params['M_disk']/1e10)
        s_Mbulge.set_val(params['M_bulge']/1e10)
        s_Mgas.set_val(params['M_gas']/1e10)
        s_Rd.set_val(params['Rd_disk'])
        s_Rmax.set_val(params['Rmax_disk'])
        s_ab.set_val(params['a_bulge'])
        s_Rdg.set_val(params['Rd_gas'])
        s_Rmg.set_val(params['Rmax_gas'])
        s_hzd.set_val(params['hz_disk'])
        s_hzb.set_val(params['hz_bulge'])
        s_hzg.set_val(params['hz_gas'])
        # Update defaults for component fields to align with current preset
        defaults.update({
            'Mdisk': params['M_disk']/1e10,
            'Mbulge': params['M_bulge']/1e10,
            'Mgas': params['M_gas']/1e10,
            'Rd': params['Rd_disk'],
            'Rmax': params['Rmax_disk'],
            'ab': params['a_bulge'],
            'Rdg': params['Rd_gas'],
            'Rmg': params['Rmax_gas'],
            'hzd': params['hz_disk'],
            'hzb': params['hz_bulge'],
            'hzg': params['hz_gas'],
        })
        refresh_main()

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

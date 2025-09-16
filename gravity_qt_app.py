#!/usr/bin/env python3
"""
Gravity UI (Qt) — A cleaner, faster frontend for the Toy Galaxy Rotation Calculator

Why Qt?
- Native widgets (spin boxes, combo boxes) perform better than Matplotlib's widget toolkit
- Robust event loop and timers for responsive, debounced updates
- Embedded Matplotlib canvas with toolbar for panning/zooming

Requires: numpy, matplotlib, PyQt5 (or PyQt6 if available)
"""
from __future__ import annotations

import sys
import math
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib
# Use QtAgg backend explicitly
matplotlib.use("QtAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Wedge


# Try PyQt6 first, fall back to PyQt5
try:
    from PyQt6.QtCore import Qt, QTimer, QThread
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
        QSplitter, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
        QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog,
        QSizePolicy, QInputDialog, QListWidget, QListWidgetItem, QRadioButton,
        QProgressBar, QDialogButtonBox, QAbstractItemView)
    QT_IS_6 = True
except Exception:
    from PyQt5.QtCore import Qt, QTimer, QThread
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
        QSplitter, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
        QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog,
        QSizePolicy, QInputDialog, QListWidget, QListWidgetItem, QRadioButton,
        QProgressBar, QDialogButtonBox, QAbstractItemView
    )
    QT_IS_6 = False

# ----- Small UI helpers (after Qt widgets are available) -----
try:
    from PyQt6.QtCore import pyqtSignal as Signal
except Exception:
    from PyQt5.QtCore import pyqtSignal as Signal

class ClickLabel(QLabel):
    clicked = Signal()
    def __init__(self, text:str="", parent=None):
        super().__init__(text, parent)
        try:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        except Exception:
            self.setCursor(Qt.PointingHandCursor)
    def mousePressEvent(self, e):
        try:
            self.clicked.emit()
        except Exception:
            pass
        return super().mousePressEvent(e)


def make_tip(text: str) -> QLabel:
    tip = QLabel(text)
    tip.setWordWrap(True)
    tip.setVisible(False)
    tip.setStyleSheet(
        "background:#f0f7ff;border:1px solid #c9e1ff;border-radius:4px;padding:6px;color:#234;"
    )
    return tip

# ===== Core physics and sampling (from PLUS version, trimmed/adapted) =====
G_AST = 4.30091e-6   # kpc * (km/s)^2 / Msun
c_kms = 299_792.458  # km/s


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


def make_simple_galaxy(seed: int, N_total: int, R_disk: float, a_bulge: float, hz: float, f_bulge: float, Mtot: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N_bulge = max(0, min(N_total, int(round(N_total * f_bulge))))
    N_disk = max(0, N_total - N_bulge)
    pos_list = []
    if N_disk > 0:
        N_base_d = max(1, N_disk // 8)
        Rd = max(0.05, R_disk / 3.0)
        pos_d = sample_exponential_disk_3d(N_base_d, Rd, R_disk, hz, rng)
        if len(pos_d):
            pos_list.append(pos_d)
    if N_bulge > 0:
        N_base_b = max(1, N_bulge // 8)
        Rmax_b = min(R_disk, max(2.0, 3.0 * a_bulge))
        pos_b = sample_plummer_bulge_3d(N_base_b, a_bulge, Rmax_b, hz, rng)
        if len(pos_b):
            pos_list.append(pos_b)
    if pos_list:
        pos = np.vstack(pos_list)
        masses = np.full(len(pos), Mtot / len(pos))
    else:
        pos = np.zeros((0, 3)); masses = np.zeros((0,))
    return pos, masses


def circular_speed_from_accel(a_vec: np.ndarray, R_vec_to_COM: np.ndarray) -> float:
    R = np.linalg.norm(R_vec_to_COM)
    if R <= 0:
        return 0.0
    u_in = -R_vec_to_COM / R
    a_rad = float(np.dot(a_vec, u_in))
    a_rad = max(0.0, a_rad)
    return math.sqrt(R * a_rad)


@dataclass
class DensityKnobs:
    g_scale: float = 1.0
    dens_k: float = 0.0
    dens_R: float = 2.0
    dens_alpha: float = 1.0  # power-law exponent; if <=0, fall back to linear dens_k mode
    dens_thresh_frac: float = 1.0  # apply enhancement only if rho_local < dens_thresh_frac * rho_ref
    well_k: float = 0.0
    well_R: float = 2.0
    boundary_R: float = 0.0  # apply per-source enhancement only for sources with R>=boundary_R (simple mode)


class GravityModel:
    def __init__(self):
        self.pos = np.zeros((0, 3))
        self.masses = np.zeros((0,))
        self.COM = np.zeros(3)
        self.test_point = np.zeros(3)
        self.test_idx = None
        self.cache_well = {"version": -1, "k": None, "R": None, "scales": None, "base": None}
        self.version = 0

    def rebuild(self, pos: np.ndarray, masses: np.ndarray):
        self.pos = pos; self.masses = masses
        if len(masses) == 0 or masses.sum() <= 0:
            self.COM = np.zeros(3)
            self.test_point = np.zeros(3)
            self.test_idx = None
        else:
            self.COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
            R_all = np.linalg.norm(pos - self.COM, axis=1)
            self.test_idx = int(np.argmax(R_all))
            self.test_point = pos[self.test_idx]
        self.version += 1
        self.cache_well.update({"version": -1, "base": None, "scales": None, "k": None, "R": None})

    def compute_G_eff(self, p: np.ndarray, knobs: DensityKnobs) -> float:
        G0 = G_AST * float(knobs.g_scale)
        Rn = max(1e-6, float(knobs.dens_R))
        if len(self.masses) == 0:
            return G0
        d2 = np.sum((self.pos - p) ** 2, axis=1)
        m_local = float(self.masses[d2 <= (Rn * Rn)].sum())
        vol = (4.0 / 3.0) * math.pi * (Rn ** 3)
        rho_local = m_local / max(1e-30, vol)
        Rchar = float(np.max(np.linalg.norm(self.pos - self.COM, axis=1))) if len(self.masses) else 1.0
        Vchar = (4.0 / 3.0) * math.pi * max(1e-30, Rchar ** 3)
        rho_ref = float(self.masses.sum()) / Vchar if Vchar > 0 else 0.0
        # Apply enhancement only if local density below threshold fraction of rho_ref
        apply_enh = (rho_ref > 0.0) and (rho_local < float(knobs.dens_thresh_frac) * rho_ref)
        alpha = float(knobs.dens_alpha)
        if apply_enh and alpha > 0.0:
            ratio = rho_ref / max(rho_local, 1e-30)
            mult = float(max(0.0, ratio ** alpha))
            return G0 * mult
        # Fallback linear mode if alpha <= 0
        k = float(knobs.dens_k)
        if apply_enh and k > 0.0 and rho_ref > 0.0:
            boost = k * max(0.0, (rho_ref / max(rho_local, 1e-30)) - 1.0)
            mult = max(0.0, 1.0 + boost)
            return G0 * mult
        return G0

    def well_scales(self, knobs: DensityKnobs) -> np.ndarray:
        k = float(knobs.well_k); Rw = max(1e-6, float(knobs.well_R))
        alpha = float(knobs.dens_alpha)
        N = len(self.masses)
        if N == 0:
            return np.ones(0)
        c = self.cache_well
        # Recompute k-independent base factor if cache invalid for current galaxy state or Rw
        needs_base = (c.get("base") is None) or (c.get("version") != self.version) or (c.get("R") != Rw)
        if needs_base:
            Rchar = float(np.max(np.linalg.norm(self.pos - self.COM, axis=1))) if N else 1.0
            Vchar = (4.0 / 3.0) * math.pi * max(1e-30, Rchar ** 3)
            rho_ref = float(self.masses.sum()) / Vchar if Vchar > 0 else 0.0
            base = np.zeros(N, dtype=float)
            batch = 1000
            for i0 in range(0, N, batch):
                i1 = min(N, i0 + batch)
                d2 = np.sum((self.pos[i0:i1, None, :] - self.pos[None, :, :]) ** 2, axis=2)
                m_local = (d2 <= (Rw * Rw)) @ self.masses
                vol = (4.0 / 3.0) * math.pi * (Rw ** 3)
                rho_local = m_local / max(1e-30, vol)
                base[i0:i1] = np.maximum(0.0, (rho_ref / np.maximum(rho_local, 1e-30)) - 1.0)
            c.update({"version": self.version, "R": Rw, "base": base})
        else:
            base = c["base"]
        # Start with ones
        scales = np.ones(N, dtype=float)
        # Apply per-source boundary-based enhancement using alpha if configured
        if alpha > 0.0 and float(getattr(knobs, 'boundary_R', 0.0)) > 0.0:
            # ratio = 1 + base
            ratio = 1.0 + base
            Rsrc = np.linalg.norm(self.pos - self.COM, axis=1)
            mask = Rsrc >= float(knobs.boundary_R)
            scales[mask] *= np.power(np.maximum(0.0, ratio[mask]), alpha)
        # Apply legacy well_k linear scaling if requested (advanced)
        if k > 0.0:
            scales *= (1.0 + k * base)
        c.update({"k": k, "scales": scales})
        return scales

    def per_body_vectors(self, p: np.ndarray, knobs: DensityKnobs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.masses) == 0:
            z3 = np.zeros((0, 3)); return z3, z3, z3
        G_eff = self.compute_G_eff(p, knobs)
        r_vec = self.pos - p
        r2 = np.sum(r_vec * r_vec, axis=1)
        inv_r3 = 1.0 / np.power(r2 + 1e-6, 1.5)
        base = (self.masses * inv_r3)
        well = self.well_scales(knobs)
        newton = (G_eff * well)[:, None] * (r_vec * base[:, None])
        gr_like = newton.copy()  # we keep GR atten off for this UI
        final = gr_like  # no boost by default
        return newton, gr_like, final

    def speeds_at_test(self, knobs: DensityKnobs) -> Tuple[float, float, float]:
        if len(self.masses) == 0:
            return 0.0, 0.0, 0.0
        newton, gr_like, final = self.per_body_vectors(self.test_point, knobs)
        aN = newton.sum(axis=0)
        aG = gr_like.sum(axis=0)
        aF = final.sum(axis=0)
        vN = circular_speed_from_accel(aN, self.test_point - self.COM)
        vG = circular_speed_from_accel(aG, self.test_point - self.COM)
        vF = circular_speed_from_accel(aF, self.test_point - self.COM)
        return vN, vG, vF


def percent_closeness(v_sim: float, v_target: float) -> float:
    if v_target <= 0:
        return 100.0 if abs(v_sim) < 1e-9 else 0.0
    return max(0.0, 100.0 * (1.0 - abs(v_sim - v_target) / v_target))


def required_g_for_test(model: GravityModel, knobs: DensityKnobs, v_target: float) -> float:
    # Compute the g_scale required to reach v_target at the test star, holding other knobs fixed.
    # Since v ~ sqrt(g) for fixed density modifiers, g_req = g_curr * (v_target / v_curr)^2
    if v_target <= 0:
        return float('nan')
    _, _, v_curr = model.speeds_at_test(knobs)
    if v_curr <= 1e-12:
        return float('inf')
    return float(knobs.g_scale) * float(v_target / v_curr) ** 2

# ===== Qt UI =====
PRESET_OPTIONS = [
    "MW-like disk", "Central-dominated", "Dwarf disk", "LMC-like dwarf", "Ultra-diffuse disk", "Compact nuclear disk"
]

PRESET_SIMPLE = {
    "MW-like disk":         dict(Nstars=8000, Rdisk=20.0, BulgeA=0.6, Hz=0.3, BulgeFrac=0.2, Mtot1e10=8.0),
    "Central-dominated":    dict(Nstars=6000, Rdisk=15.0, BulgeA=0.8, Hz=0.5, BulgeFrac=0.8, Mtot1e10=9.5),
    "Dwarf disk":           dict(Nstars=3000, Rdisk=6.0,  BulgeA=0.3, Hz=0.2, BulgeFrac=0.0, Mtot1e10=0.45),
    "LMC-like dwarf":       dict(Nstars=2500, Rdisk=9.0,  BulgeA=0.2, Hz=0.25, BulgeFrac=0.0, Mtot1e10=0.40),
    "Ultra-diffuse disk":   dict(Nstars=2000, Rdisk=30.0, BulgeA=0.3, Hz=0.6,  BulgeFrac=0.0, Mtot1e10=0.07),
    "Compact nuclear disk": dict(Nstars=5000, Rdisk=2.0,  BulgeA=0.2, Hz=0.3,  BulgeFrac=0.9, Mtot1e10=2.2),
}


class VectorsDialog(QDialog):
    def __init__(self, model: GravityModel, knobs: DensityKnobs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Force Vectors")
        self.model = model
        self.knobs = knobs
        self.resize(900, 600)

        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        self.topN = QSpinBox(); self.topN.setRange(1, max(1, len(self.model.masses)))
        self.topN.setValue(min(25, max(1, len(self.model.masses))))
        btnSave = QPushButton("Save CSV")
        ctrl.addWidget(QLabel("Top N:")); ctrl.addWidget(self.topN); ctrl.addStretch(1); ctrl.addWidget(btnSave)
        layout.addLayout(ctrl)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["i","dx","dy","dz","r_kpc","mass","ax","ay","az","|a|"])
        layout.addWidget(self.table)

        self.topN.valueChanged.connect(self.refresh)
        btnSave.clicked.connect(self.save_csv)
        self.refresh()

    def refresh(self):
        newton, gr_like, final = self.model.per_body_vectors(self.model.test_point, self.knobs)
        mags = np.linalg.norm(final, axis=1)
        order = np.argsort(mags)[::-1]
        N = min(self.topN.value(), len(order))
        idx = order[:N]
        self.table.setRowCount(N)
        for row, i in enumerate(idx):
            dx, dy, dz = self.model.pos[i] - self.model.test_point
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            ax, ay, az = final[i]
            values = [i, dx, dy, dz, r, self.model.masses[i], ax, ay, az, math.sqrt(ax*ax+ay*ay+az*az)]
            for col, val in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(f"{val:.6g}" if col>0 else str(val)))

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save force_vectors.csv", str(Path.cwd()/"force_vectors.csv"))
        if not path:
            return
        newton, gr_like, final = self.model.per_body_vectors(self.model.test_point, self.knobs)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i","pos_x","pos_y","pos_z","dx","dy","dz","r_kpc","mass_Msun",
                        "final_ax","final_ay","final_az","final_mag"])
            for i in range(len(self.model.masses)):
                dx, dy, dz = self.model.pos[i] - self.model.test_point
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                fx, fy, fz = final[i]
                w.writerow([i, self.model.pos[i,0], self.model.pos[i,1], self.model.pos[i,2], dx, dy, dz, r,
                            self.model.masses[i], fx, fy, fz, math.sqrt(fx*fx+fy*fy+fz*fz)])


class CompareWorker(QThread):
    rowReady = Signal(dict)
    progress = Signal(int, int, str)  # done, total, preset name
    finished = Signal(list)

    def __init__(self, presets, knobs: DensityKnobs, objective: str, vobs: float, obs_curve=None, N_override: int = 0,
                 resolve: bool = True, solve_dk: bool = True, solve_g: bool = True, solve_wk: bool = True,
                 boundary_mw_kpc: float = 8.0, use_mw_frac: bool = True, parent=None):
        super().__init__(parent)
        self.presets = presets
        self.knobs = knobs
        self.objective = objective
        self.vobs = float(vobs)
        self.obs_curve = obs_curve
        self.N_override = int(N_override or 0)
        self.resolve = bool(resolve)
        self.solve_dk = bool(solve_dk)
        self.solve_g = bool(solve_g)
        self.solve_wk = bool(solve_wk)
        self.rows = []
        self.boundary_mw_kpc = float(boundary_mw_kpc)
        self.use_mw_frac = bool(use_mw_frac)

    def run(self):
        import time
        t0 = time.perf_counter()
        total = len(self.presets)
        for i, name in enumerate(self.presets, 1):
            row = self._process_one(name)
            self.rows.append(row)
            self.rowReady.emit(row)
            self.progress.emit(i, total, name)
        self.finished.emit(self.rows)

    def _build_from_preset(self, name: str):
        p = PRESET_SIMPLE.get(name, PRESET_SIMPLE[PRESET_OPTIONS[0]])
        Nstars = self.N_override if self.N_override > 0 else int(p['Nstars'])
        pos, masses = make_simple_galaxy(
            seed=42,
            N_total=Nstars,
            R_disk=float(p['Rdisk']),
            a_bulge=float(p['BulgeA']),
            hz=float(p['Hz']),
            f_bulge=float(p['BulgeFrac']),
            Mtot=float(p['Mtot1e10'])*1e10,
        )
        m = GravityModel(); m.rebuild(pos, masses)
        return m

    def _curve_vtriple_for_model(self, model: GravityModel, knobs: DensityKnobs, nR: int = 60):
        if len(model.masses) == 0:
            z = np.zeros(1); return np.array([0.0]), z, z, z
        Rmax = float(np.max(np.linalg.norm(model.pos - model.COM, axis=1)))
        R_vals = np.linspace(max(0.2, 0.02*Rmax), Rmax, nR)
        vN, vG, vF = [], [], []
        for R in R_vals:
            p = model.COM + np.array([R,0,0])
            newton, gr_like, final = model.per_body_vectors(p, knobs)
            vN.append(circular_speed_from_accel(newton.sum(axis=0), p - model.COM))
            vG.append(circular_speed_from_accel(gr_like.sum(axis=0), p - model.COM))
            vF.append(circular_speed_from_accel(final.sum(axis=0), p - model.COM))
        return np.array(R_vals), np.array(vN), np.array(vG), np.array(vF)

    def _golden_section(self, f, lo, hi, iters=28, tol=1e-4):
        phi = (1 + 5 ** 0.5) / 2
        invphi = 1 / phi
        a, b = float(lo), float(hi)
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = f(c)
        fd = f(d)
        for _ in range(iters):
            if abs(b - a) < tol * (abs(c) + abs(d)) + 1e-12:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = f(d)
        x = (a + b) / 2
        fx = f(x)
        return x, fx

    def _boundary_for_preset(self, preset_name: str) -> float:
        try:
            mw_Rdisk = float(PRESET_SIMPLE["MW-like disk"]["Rdisk"])  # kpc
            this_Rdisk = float(PRESET_SIMPLE.get(preset_name, {}).get("Rdisk", mw_Rdisk))
            if self.use_mw_frac and mw_Rdisk > 0:
                frac = self.boundary_mw_kpc / mw_Rdisk
                return frac * this_Rdisk
            else:
                return self.boundary_mw_kpc
        except Exception:
            return self.boundary_mw_kpc

    def _err_for_knobs(self, model: GravityModel, knobs: DensityKnobs):
        if self.objective.startswith('curve'):
            R_vals, vN_arr, vG_arr, vF_arr = self._curve_vtriple_for_model(model, knobs, nR=60)
            if self.obs_curve is not None:
                R_obs, v_obs = self.obs_curve
                import numpy as _np
                v_obs_grid = _np.interp(R_vals, R_obs, v_obs, left=v_obs[0], right=v_obs[-1])
            else:
                v_obs_grid = np.full_like(vF_arr, self.vobs, dtype=float)
            return float(np.sum((vF_arr - v_obs_grid)**2))
        else:
            _, _, vF = model.speeds_at_test(knobs)
            return float((vF - self.vobs) ** 2)

    def _process_one(self, name: str):
        import time
        t1 = time.perf_counter()
        model = self._build_from_preset(name)
        # Apply boundary for this preset
        k2 = DensityKnobs(
            g_scale=self.knobs.g_scale,
            dens_k=self.knobs.dens_k, dens_R=self.knobs.dens_R,
            dens_alpha=self.knobs.dens_alpha, dens_thresh_frac=self.knobs.dens_thresh_frac,
            well_k=self.knobs.well_k, well_R=self.knobs.well_R,
            boundary_R=self._boundary_for_preset(name),
        )
        err_fixed = self._err_for_knobs(model, k2)
        # Compute required G and closeness at test star for fixed knobs (test-star objective semantics)
        _, _, vF_fixed = model.speeds_at_test(self.knobs)
        g_req_fixed = required_g_for_test(model, self.knobs, self.vobs)
        close_fixed = percent_closeness(vF_fixed, self.vobs)
        row = {
            "preset": name,
            "N": int(len(model.masses)),
            "err_fixed": float(err_fixed),
            "close_fixed_pct": float(close_fixed),
            "g": float(self.knobs.g_scale),
            "dens_k": float(self.knobs.dens_k),
            "dens_R": float(self.knobs.dens_R),
            "alpha": float(self.knobs.dens_alpha),
            "boundary_R": float(k2.boundary_R),
            "well_k": float(self.knobs.well_k),
            "well_R": float(self.knobs.well_R),
            "g_req_fixed": float(g_req_fixed),
        }
        if self.resolve:
            g_lo, g_hi = 0.5, 2.0
            dk_lo, dk_hi = 0.0, 3.0
            wk_lo, wk_hi = 0.0, 1.0
            g = float(self.knobs.g_scale); dk = float(self.knobs.dens_k); wk = float(self.knobs.well_k)
            for _ in range(2):
                if self.solve_dk:
                    f1 = lambda x: self._err_for_knobs(model, DensityKnobs(g_scale=g, dens_k=max(dk_lo, min(dk_hi, x)), dens_R=self.knobs.dens_R, dens_alpha=self.knobs.dens_alpha, dens_thresh_frac=self.knobs.dens_thresh_frac, well_k=wk, well_R=self.knobs.well_R))
                    dk, _ = self._golden_section(f1, dk_lo, dk_hi)
                if self.solve_g:
                    f2 = lambda x: self._err_for_knobs(model, DensityKnobs(g_scale=max(g_lo, min(g_hi, x)), dens_k=dk, dens_R=self.knobs.dens_R, dens_alpha=self.knobs.dens_alpha, dens_thresh_frac=self.knobs.dens_thresh_frac, well_k=wk, well_R=self.knobs.well_R))
                    g, _ = self._golden_section(f2, g_lo, g_hi)
                if self.solve_wk:
                    f3 = lambda x: self._err_for_knobs(model, DensityKnobs(g_scale=g, dens_k=dk, dens_R=self.knobs.dens_R, dens_alpha=self.knobs.dens_alpha, dens_thresh_frac=self.knobs.dens_thresh_frac, well_k=max(wk_lo, min(wk_hi, x)), well_R=self.knobs.well_R))
                    wk, _ = self._golden_section(f3, wk_lo, wk_hi)
            solved_knobs = DensityKnobs(g_scale=float(g), dens_k=float(dk), dens_R=float(self.knobs.dens_R), dens_alpha=self.knobs.dens_alpha, dens_thresh_frac=self.knobs.dens_thresh_frac, well_k=float(wk), well_R=float(self.knobs.well_R), boundary_R=self._boundary_for_preset(name))
            err_solved = float(self._err_for_knobs(model, solved_knobs))
            _, _, vF_solved = model.speeds_at_test(solved_knobs)
            g_req_solved = required_g_for_test(model, solved_knobs, self.vobs)
            close_solved = percent_closeness(vF_solved, self.vobs)
            row.update({
                "g_solved": solved_knobs.g_scale,
                "dens_k_solved": solved_knobs.dens_k,
                "well_k_solved": solved_knobs.well_k,
                "err_solved": err_solved,
                "g_req_solved": float(g_req_solved),
                "close_solved_pct": float(close_solved),
            })
        t2 = time.perf_counter()
        row["time_s"] = float(t2 - t1)
        return row


class CompareAcrossPresetsDialog(QDialog):
    def __init__(self, parent: 'MainWindow'):
        super().__init__(parent)
        self.setWindowTitle("Compare Across Presets")
        self.parent_ref = parent
        self.resize(900, 600)
        self.rows = []

        vbox = QVBoxLayout(self)
        ctrl = QGridLayout()

        # Preset list
        self.listPresets = QListWidget()
        self.listPresets.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for name in PRESET_OPTIONS:
            item = QListWidgetItem(name)
            self.listPresets.addItem(item)
            item.setSelected(True)
        ctrl.addWidget(QLabel("Presets"), 0, 0)
        ctrl.addWidget(self.listPresets, 1, 0, 4, 1)

        # Options panel
        self.radioTest = QRadioButton("Objective: match test star")
        self.radioCurve = QRadioButton("Objective: match curve")
        self.radioTest.setChecked(True)
        self.chkResolve = QCheckBox("Re-solve per preset starting from current knobs")
        self.chkResolve.setChecked(True)
        self.spinN = QSpinBox(); self.spinN.setRange(100, 200000); self.spinN.setValue(3000)
        ctrl.addWidget(self.radioTest, 0, 1)
        ctrl.addWidget(self.radioCurve, 1, 1)
        ctrl.addWidget(self.chkResolve, 2, 1)
        ctrl.addWidget(QLabel("N stars override"), 3, 1)
        ctrl.addWidget(self.spinN, 3, 2)

        vbox.addLayout(ctrl)

        # Table
        headers = [
            "Preset","N","Err (fixed)","Err (solved)","% close (fixed)","% close (solved)",
            "G","dens_k","dens_R","alpha","boundary_R","well_k","well_R","g_req (fixed)","g_req (solved)","time_s"
        ]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        vbox.addWidget(self.table)

        # Progress + buttons
        bot = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        self.btnRun = QPushButton("Run")
        self.btnSaveCSV = QPushButton("Save CSV…")
        self.btnApply = QPushButton("Apply knobs from selected")
        self.btnClose = QPushButton("Close")
        bot.addWidget(self.progress, 1)
        bot.addWidget(self.btnRun)
        bot.addWidget(self.btnSaveCSV)
        bot.addWidget(self.btnApply)
        bot.addWidget(self.btnClose)
        vbox.addLayout(bot)

        self.btnRun.clicked.connect(self.start_run)
        self.btnSaveCSV.clicked.connect(self.save_csv)
        self.btnApply.clicked.connect(self.apply_selected)
        self.btnClose.clicked.connect(self.accept)

    def start_run(self):
        names = [i.text() for i in self.listPresets.selectedItems()]
        if not names:
            return
        obj = 'test' if self.radioTest.isChecked() else 'curve'
        N_override = int(self.spinN.value())
        kn = DensityKnobs(
            g_scale=float(self.parent_ref.spinG.value()),
            dens_k=float(self.parent_ref.spinDk.value()), dens_R=float(self.parent_ref.spinDr.value()),
            dens_alpha=float(self.parent_ref.spinDa.value()), dens_thresh_frac=float(self.parent_ref.spinDth.value()),
            well_k=float(self.parent_ref.spinWk.value()), well_R=float(self.parent_ref.spinWr.value()),
        )
        obs_curve = self.parent_ref.obsCurve if (obj=='curve') else None
        self.worker = CompareWorker(names, kn, obj, float(self.parent_ref.spinVobs.value()), obs_curve=obs_curve,
                                    N_override=N_override, resolve=self.chkResolve.isChecked(),
                                    boundary_mw_kpc=float(self.parent_ref.spinBoundaryMW.value()), use_mw_frac=bool(self.parent_ref.chkBoundaryUseMWFrac.isChecked()))
        self.worker.rowReady.connect(self._on_row_ready)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.rows = []
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.btnRun.setEnabled(False)
        self.worker.start()

    def _on_row_ready(self, row: dict):
        self.rows.append(row)
        r = self.table.rowCount(); self.table.insertRow(r)
        def get(k, default=""):
            return row[k] if k in row else default
        values = [
            row["preset"], get("N", ""), f"{row['err_fixed']:.4g}",
            (f"{row['err_solved']:.4g}" if 'err_solved' in row else ""),
            (f"{row['close_fixed_pct']:.1f}%" if 'close_fixed_pct' in row else ""),
            (f"{row['close_solved_pct']:.1f}%" if 'close_solved_pct' in row else ""),
            f"{row['g']:.4f}", f"{row['dens_k']:.4f}", f"{row['dens_R']:.3f}", f"{row.get('alpha','')}", f"{row.get('boundary_R','')}", f"{row['well_k']:.4f}", f"{row['well_R']:.3f}",
            (f"{row['g_req_fixed']:.4f}" if 'g_req_fixed' in row and np.isfinite(row['g_req_fixed']) else "inf"),
            (f"{row['g_req_solved']:.4f}" if 'g_req_solved' in row and np.isfinite(row.get('g_req_solved', float('nan'))) else ""),
            f"{row['time_s']:.2f}"
        ]
        for c, val in enumerate(values):
            self.table.setItem(r, c, QTableWidgetItem(str(val)))

    def _on_progress(self, done: int, total: int, name: str):
        pct = int(100 * done / max(1, total))
        self.progress.setValue(pct)

    def _on_finished(self, rows: list):
        self.btnRun.setEnabled(True)

    def save_csv(self):
        if not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save compare.csv", str(Path.cwd()/"compare_presets.csv"))
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            keys = [
                "preset","N","err_fixed","err_solved","close_fixed_pct","close_solved_pct",
                "g","dens_k","dens_R","alpha","boundary_R","well_k","well_R","g_req_fixed","g_req_solved","time_s"
            ]
            w.writerow(keys)
            for row in self.rows:
                w.writerow([row.get(k, "") for k in keys])

    def apply_selected(self):
        r = self.table.currentRow()
        if r < 0:
            return
        # Prefer solved knobs if present, else fixed
        row = self.rows[r]
        g = float(row.get('g_solved', row.get('g', 1.0)))
        dk = float(row.get('dens_k_solved', row.get('dens_k', 0.0)))
        wk = float(row.get('well_k_solved', row.get('well_k', 0.0)))
        dens_R = float(row.get('dens_R', self.parent_ref.spinDr.value()))
        well_R = float(row.get('well_R', self.parent_ref.spinWr.value()))
        self.parent_ref.apply_knobs(DensityKnobs(g_scale=g, dens_k=dk, dens_R=dens_R, well_k=wk, well_R=well_R))
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gravity Calculator — Qt Frontend")
        self.resize(1200, 800)

        # Core state
        self.model = GravityModel()
        self.knobs = DensityKnobs()
        # Initialize animation/curve-related placeholders early to avoid attribute errors during startup
        self.base_pos = np.zeros((0,3))
        self.base_angles = np.zeros(0)
        self.base_rxy = np.zeros(0)
        self.vf_stars = np.zeros(0)
        self.vg_stars = np.zeros(0)
        self.curveR = np.zeros(0)
        self.curveVF = np.zeros(0)
        self.curveVG = np.zeros(0)
        self.moving_mask = np.zeros(0, dtype=bool)

        # Central splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); right = QWidget()
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([800, 400])
        self.setCentralWidget(splitter)

        # Left: Matplotlib canvas + toolbar + info label
        leftLayout = QVBoxLayout(left)
        self.fig = Figure(figsize=(6,5), tight_layout=False)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.infoLabel = QLabel("")
        self.infoLabel.setWordWrap(True)
        leftLayout.addWidget(self.toolbar)
        leftLayout.addWidget(self.canvas, 1)
        leftLayout.addWidget(self.infoLabel)

        # Right: controls
        rightLayout = QVBoxLayout(right)
        form = QFormLayout()

        # Helper: add clickable rows; tips appear in a dedicated panel below
        def add_row_with_tip(layout: QFormLayout, label_text: str, widget: QWidget, tip_text: str):
            lbl = ClickLabel(label_text + "  ⓘ")
            layout.addRow(lbl, widget)
            lbl.clicked.connect(lambda _=None, k=label_text, t=tip_text: self.toggle_tip(k, t))
            return lbl

        # Preset
        self.preset = QComboBox(); self.preset.addItems(PRESET_OPTIONS)
        add_row_with_tip(form, "Preset", self.preset,
            "Selects a galaxy template. It sets N stars, disk size, bulge fraction/scale, and total mass. Changing presets rebuilds the galaxy.")

        # Core knobs
        self.spinG = QDoubleSpinBox(); self.spinG.setRange(0.1, 5.0); self.spinG.setDecimals(3); self.spinG.setValue(1.0)
        self.spinDk = QDoubleSpinBox(); self.spinDk.setRange(0.0, 5.0); self.spinDk.setDecimals(3); self.spinDk.setValue(0.0)
        self.spinDr = QDoubleSpinBox(); self.spinDr.setRange(0.05, 50.0); self.spinDr.setDecimals(3); self.spinDr.setValue(2.0)
        self.chkDrFrac = QCheckBox("Use dens R as fraction of Rchar")
        self.spinDrFrac = QDoubleSpinBox(); self.spinDrFrac.setRange(0.01, 1.0); self.spinDrFrac.setDecimals(3); self.spinDrFrac.setValue(0.15)
        self.spinDa = QDoubleSpinBox(); self.spinDa.setRange(0.0, 3.0); self.spinDa.setDecimals(3); self.spinDa.setValue(1.0)
        self.spinDth = QDoubleSpinBox(); self.spinDth.setRange(0.0, 5.0); self.spinDth.setDecimals(3); self.spinDth.setValue(1.0)
        # Absolute density threshold (Msun/kpc^3) synced with fraction
        self.spinDAbs = QDoubleSpinBox(); self.spinDAbs.setRange(0.0, 1e12); self.spinDAbs.setDecimals(3); self.spinDAbs.setValue(0.0)
        self.spinWk = QDoubleSpinBox(); self.spinWk.setRange(0.0, 5.0); self.spinWk.setDecimals(3); self.spinWk.setValue(0.0)
        self.spinWr = QDoubleSpinBox(); self.spinWr.setRange(0.05, 20.0); self.spinWr.setDecimals(3); self.spinWr.setValue(2.0)
        self.spinVobs = QDoubleSpinBox(); self.spinVobs.setRange(0.0, 1000.0); self.spinVobs.setDecimals(1); self.spinVobs.setValue(220.0)
        add_row_with_tip(form, "G scale", self.spinG,
            "Global multiplier on Newton's constant G. Speeds scale roughly as sqrt(G). Increase to raise all speeds.")
        add_row_with_tip(form, "Density k", self.spinDk,
            "Legacy linear mode: if local density at the test star is low vs global, G_eff multiplies by (1 + k*(rho_ref/rho_local - 1)).")
        add_row_with_tip(form, "Density R (kpc)", self.spinDr,
            "Neighborhood size for computing local density at the evaluation point. 1–5 kpc typical. Smaller R makes the effect more local.")
        add_row_with_tip(form, "Use dens R as frac of Rchar", self.chkDrFrac,
            "If enabled, dens_R = frac × Rchar; otherwise, use absolute kpc.")
        add_row_with_tip(form, "dens R frac of Rchar", self.spinDrFrac,
            "When enabled, sets dens_R = frac × Rchar.")
        add_row_with_tip(form, "Density alpha", self.spinDa,
            "Power-law enhancement only at the evaluation point when local density is low: multiplier = (rho_ref/rho_local)^alpha. Set 0 to use legacy k mode.")
        add_row_with_tip(form, "Low dens thresh × rho_ref", self.spinDth,
            "Only apply density enhancement when rho_local < (threshold × rho_ref). For Earth-like regions, set threshold ≤ 1.0 so dense regions get baseline G only.")
        add_row_with_tip(form, "Low dens abs (Msun/kpc^3)", self.spinDAbs,
            "Absolute density threshold; synced with fraction via rho_ref.")
        add_row_with_tip(form, "Well k", self.spinWk,
            "Per-source modifier: stars in shallower local density get a >1× scale on their vector. Higher k boosts low-density contributors.")
        add_row_with_tip(form, "Well R (kpc)", self.spinWr,
            "Neighborhood size around each source to judge 'well' shallowness. 1–3 kpc typical.")
        add_row_with_tip(form, "Observed v (km/s)", self.spinVobs,
            "Target velocity for comparison only. It does not change physics; we show Final − Observed.")

        # Structure (collapsible)
        grpStruct = QGroupBox("Structure (preset-controlled)")
        grpStruct.setCheckable(True); grpStruct.setChecked(False)
        f2 = QFormLayout()
        self.sN = QSpinBox(); self.sN.setRange(8, 200000); self.sN.setValue(4000)
        self.sR = QDoubleSpinBox(); self.sR.setRange(0.5, 200.0); self.sR.setValue(20.0)
        self.sBa = QDoubleSpinBox(); self.sBa.setRange(0.02, 10.0); self.sBa.setValue(0.6)
        self.sHz = QDoubleSpinBox(); self.sHz.setRange(0.01, 3.0); self.sHz.setValue(0.3)
        self.sBf = QDoubleSpinBox(); self.sBf.setRange(0.0, 1.0); self.sBf.setSingleStep(0.05); self.sBf.setValue(0.2)
        self.sMt = QDoubleSpinBox(); self.sMt.setRange(0.01, 200.0); self.sMt.setValue(8.0)
        add_row_with_tip(f2, "N stars", self.sN, "Number of particles. More = smoother but slower. 2k–10k is a good balance.")
        add_row_with_tip(f2, "R disk (kpc)", self.sR, "Outer disk radius; sets how far luminous matter extends.")
        add_row_with_tip(f2, "Bulge a (kpc)", self.sBa, "Bulge Plummer scale; smaller = more concentrated center.")
        add_row_with_tip(f2, "Thickness hz", self.sHz, "Vertical thickness; larger spreads stars in z.")
        add_row_with_tip(f2, "Bulge frac", self.sBf, "Fraction of stars in the bulge. 0 = pure disk; ~0.8 = bulge-dominated.")
        add_row_with_tip(f2, "Total mass (1e10 Msun)", self.sMt, "Total luminous+gas mass distributed across all stars.")
        grpStruct.setLayout(f2)

        # Simple boundary group
        self.grpBoundary = QGroupBox("Boundary and enhancement (simple)")
        layB = QFormLayout(self.grpBoundary)
        self.spinBoundaryMW = QDoubleSpinBox(); self.spinBoundaryMW.setRange(0.0, 200.0); self.spinBoundaryMW.setDecimals(2); self.spinBoundaryMW.setValue(8.0)
        self.chkBoundaryUseMWFrac = QCheckBox("Apply MW fraction to all presets")
        self.chkBoundaryUseMWFrac.setChecked(True)
        self.spinAlphaSimple = QDoubleSpinBox(); self.spinAlphaSimple.setRange(0.0, 3.0); self.spinAlphaSimple.setDecimals(3); self.spinAlphaSimple.setValue(self.spinDa.value())
        layB.addRow("MW boundary R (kpc)", self.spinBoundaryMW)
        layB.addRow(self.chkBoundaryUseMWFrac)
        layB.addRow("Enhancement exponent α", self.spinAlphaSimple)
        rightLayout.addWidget(self.grpBoundary)

        rightLayout.addLayout(form)
        rightLayout.addWidget(grpStruct)

        # Buttons
        btnRow = QHBoxLayout()
        self.btnBuild = QPushButton("Build")
        self.btnVectors = QPushButton("Force Vectors…")
        self.btnSaveCSV = QPushButton("Save Curve CSV")
        btnRow.addWidget(self.btnBuild); btnRow.addWidget(self.btnVectors); btnRow.addWidget(self.btnSaveCSV)
        rightLayout.addLayout(btnRow)

        # Solver group
        self.grpSolve = QGroupBox("Solver")
        layS = QFormLayout(self.grpSolve)
        self.solveObjective = QComboBox(); self.solveObjective.addItems(["Match test star (Final → Observed)"])  # placeholder for future curve fit
        self.chkSolveDk = QCheckBox("Solve Density k"); self.chkSolveDk.setChecked(True)
        self.chkSolveG  = QCheckBox("Solve G scale"); self.chkSolveG.setChecked(False)
        self.chkSolveWk = QCheckBox("Solve Well k"); self.chkSolveWk.setChecked(False)
        self.btnSolve = QPushButton("Solve")
        self.btnLoadObs = QPushButton("Load Obs Curve…")
        layS.addRow("Objective", self.solveObjective)
        layS.addRow(self.chkSolveDk)
        layS.addRow(self.chkSolveG)
        layS.addRow(self.chkSolveWk)
        layS.addRow(self.btnSolve)
        layS.addRow(self.btnLoadObs)
        # Extra actions: save/load and compare
        self.btnSaveSolution = QPushButton("Save Solution…")
        self.btnLoadSolution = QPushButton("Load Solution…")
        self.btnCompare = QPushButton("Compare Across Presets…")
        layS.addRow(self.btnSaveSolution)
        layS.addRow(self.btnLoadSolution)
        layS.addRow(self.btnCompare)
        rightLayout.addWidget(self.grpSolve)
        self.obsCurve = None  # (R_obs, v_obs)

        # Dedicated Tips panel
        self.tipsBox = QGroupBox("Tips")
        self.tipsBox.setVisible(False)
        tipsLayout = QVBoxLayout(self.tipsBox)
        try:
            from PyQt6.QtWidgets import QTextEdit
        except Exception:
            from PyQt5.QtWidgets import QTextEdit
        self.tipsText = QTextEdit()
        self.tipsText.setReadOnly(True)
        self.tipsText.setMinimumHeight(180)
        self.tipsText.setStyleSheet("background:#f8fbff;border:1px solid #c9e1ff;border-radius:4px;padding:8px;color:#123;")
        tipsLayout.addWidget(self.tipsText)
        rightLayout.addWidget(self.tipsBox)
        # Ring overlay controls
        self.grpRings = QGroupBox("Ring density overlay")
        self.grpRings.setCheckable(True); self.grpRings.setChecked(False)
        layR = QFormLayout(self.grpRings)
        self.chkShowRings = QCheckBox("Show low-density rings")
        self.spinRBins = QSpinBox(); self.spinRBins.setRange(5, 200); self.spinRBins.setValue(40)
        self.spinRThresh = QDoubleSpinBox(); self.spinRThresh.setRange(0.0, 5.0); self.spinRThresh.setDecimals(3); self.spinRThresh.setValue(1.0)
        self.spinRAbs = QDoubleSpinBox(); self.spinRAbs.setRange(0.0, 1e12); self.spinRAbs.setDecimals(3); self.spinRAbs.setValue(0.0)
        self.spinRHz = QDoubleSpinBox(); self.spinRHz.setRange(0.001, 50.0); self.spinRHz.setDecimals(3); self.spinRHz.setValue( max(0.002, 2.0*self.spinDr.value()) )
        layR.addRow(self.chkShowRings)
        layR.addRow("Radial bins", self.spinRBins)
        layR.addRow("Low dens thresh × rho_ref", self.spinRThresh)
        layR.addRow("Low dens abs (Msun/kpc^3)", self.spinRAbs)
        layR.addRow("Vertical thickness (kpc)", self.spinRHz)
        rightLayout.addWidget(self.grpRings)
        rightLayout.addStretch(1)

        # Matplotlib artists
        self.scatter = self.ax.scatter([], [], s=5, alpha=0.6)
        self.star, = self.ax.plot([], [], marker='*', ms=10, color='gold')
        self.com, = self.ax.plot([], [], marker='x', ms=8)
        # Patch indicating low-density enhancement region around test star
        self.lowDensPatch = Circle((0,0), radius=1.0, facecolor=(0.2,0.5,1.0,0.08), edgecolor=None)
        self.lowDensPatch.set_visible(False)
        self.ax.add_patch(self.lowDensPatch)
        self.ax.set_xlabel("x (kpc)"); self.ax.set_ylabel("y (kpc)")
        self.ax.set_title("Gravity Calculator — Qt")
        # Ring density overlay (annuli)
        self.ringPatches = []

        # Events (debounced)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh_physics)

        self.preset.currentTextChanged.connect(self.on_preset)
        self.btnBuild.clicked.connect(self.build_from_fields)
        self.btnVectors.clicked.connect(self.open_vectors)
        self.btnSaveCSV.clicked.connect(self.save_curve_csv)
        self.btnSolve.clicked.connect(self.solve_params)
        self.btnLoadObs.clicked.connect(self.load_obs_curve)
        # Simple boundary wiring: keep alpha in sync
        self.spinAlphaSimple.valueChanged.connect(lambda v: self.spinDa.setValue(float(v)))
        # Ring overlay signals
        self.grpRings.toggled.connect(lambda _v: self.refresh_ring_overlay())
        self.chkShowRings.toggled.connect(lambda _v: self.refresh_ring_overlay())
        self.spinRBins.valueChanged.connect(lambda _v: self.refresh_ring_overlay())
        self.spinRThresh.valueChanged.connect(lambda _v: self.refresh_ring_overlay())
        self.spinRAbs.valueChanged.connect(lambda _v: self.refresh_ring_overlay(abs_changed=True))
        self.spinRHz.valueChanged.connect(lambda _v: self.refresh_ring_overlay())
        # Initialize sync guard before first sync
        self._sync_guard = False
        # Initial threshold sync
        self.sync_density_thresholds()
        self.btnSaveSolution.clicked.connect(self.save_solution_profile)
        self.btnLoadSolution.clicked.connect(self.load_solution_profile)
        self.btnCompare.clicked.connect(self.open_compare_dialog)
        for w in [self.spinG, self.spinDk, self.spinDr, self.chkDrFrac, self.spinDrFrac, self.spinDa, self.spinDth, self.spinDAbs, self.spinWk, self.spinWr, self.spinVobs, self.spinBoundaryMW, self.chkBoundaryUseMWFrac, self.spinAlphaSimple]:
            try:
                w.valueChanged.connect(self.schedule_refresh)
            except Exception:
                # QCheckBox uses toggled
                w.toggled.connect(self.schedule_refresh)
        # Sync fraction/absolute thresholds based on rho_ref
        self._sync_guard = False
        self.spinDth.valueChanged.connect(lambda _v: self.sync_density_thresholds())
        self.spinDAbs.valueChanged.connect(lambda _v: self.sync_density_thresholds(from_abs=True))
        for w in [self.sN, self.sR, self.sBa, self.sHz, self.sBf, self.sMt]:
            w.valueChanged.connect(self.schedule_rebuild)
        grpStruct.toggled.connect(lambda _checked: None)  # keep state; no action needed

        # Initialize
        self.on_preset(self.preset.currentText())

        # Animation state
        self.animTimer = QTimer(self); self.animTimer.setInterval(40)
        self.animTimer.timeout.connect(self.step_animation)
        self.grpAnim = QGroupBox("Animation")
        layA = QFormLayout(self.grpAnim)
        self.chkAnimate = QCheckBox("Animate stars in xy plane")
        self.spinAnim = QDoubleSpinBox(); self.spinAnim.setRange(0.0, 5.0); self.spinAnim.setDecimals(2); self.spinAnim.setValue(1.00)
        layA.addRow(self.chkAnimate)
        layA.addRow("Speed scale", self.spinAnim)
        # insert below solver
        rightLayout.insertWidget(rightLayout.indexOf(self.grpSolve)+1, self.grpAnim)
        self.chkAnimate.toggled.connect(self.toggle_animation)
        # GR tolerance for classification
        self.spinTol = QDoubleSpinBox(); self.spinTol.setRange(0.1, 50.0); self.spinTol.setDecimals(1); self.spinTol.setValue(10.0)
        layA.addRow("GR tol %", self.spinTol)
        self.spinTol.valueChanged.connect(lambda _v: self.update_colors())

    # ----- Scheduling helpers -----
    def schedule_refresh(self):
        self.timer.start()

    def schedule_rebuild(self):
        self.build_from_fields()

    # ----- Tips helpers -----
    def show_tip(self, key: str, text: str):
        self.current_tip_key = key
        self.tipsText.setPlainText(text)
        self.tipsBox.setVisible(True)

    def toggle_tip(self, key: str, text: str):
        if getattr(self, 'current_tip_key', None) == key and self.tipsBox.isVisible():
            self.tipsBox.setVisible(False)
        else:
            self.show_tip(key, text)

    # ----- Solver -----
    def curve_vtriple(self, knobs: DensityKnobs, nR: int = 60, R_vals=None):
        if len(self.model.masses) == 0:
            z = np.zeros(1); return np.array([0.0]), z, z, z
        if R_vals is None:
            Rmax = float(np.max(np.linalg.norm(self.model.pos - self.model.COM, axis=1)))
            R_vals = np.linspace(max(0.2, 0.02*Rmax), Rmax, nR)
        vN, vG, vF = [], [], []
        for R in R_vals:
            p = self.model.COM + np.array([R,0,0])
            newton, gr_like, final = self.model.per_body_vectors(p, knobs)
            vN.append(circular_speed_from_accel(newton.sum(axis=0), p - self.model.COM))
            vG.append(circular_speed_from_accel(gr_like.sum(axis=0), p - self.model.COM))
            vF.append(circular_speed_from_accel(final.sum(axis=0), p - self.model.COM))
        return np.array(R_vals), np.array(vN), np.array(vG), np.array(vF)

    def load_obs_curve(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load observed curve CSV", str(Path.cwd()))
        if not path:
            return
        R_obs = []
        v_obs = []
        import csv as _csv
        with open(path, 'r') as f:
            rdr = _csv.reader(f)
            head = next(rdr, None)
            for row in rdr:
                if not row or len(row) < 2:
                    continue
                try:
                    R_obs.append(float(row[0])); v_obs.append(float(row[1]))
                except Exception:
                    continue
        if len(R_obs) == 0:
            self.show_tip("Solver", "Could not parse any (R, v) rows from the CSV. Expect two columns: R_kpc, v_kms.")
            return
        import numpy as _np
        self.obsCurve = ( _np.array(R_obs, dtype=float), _np.array(v_obs, dtype=float) )
        self.show_tip("Solver", f"Loaded observed curve with {len(R_obs)} points from {Path(path).name}.")

    def golden_section(self, f, lo, hi, iters=28, tol=1e-4):
        phi = (1 + 5 ** 0.5) / 2
        invphi = 1 / phi
        invphi2 = (1 - invphi)
        a, b = float(lo), float(hi)
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = f(c)
        fd = f(d)
        for _ in range(iters):
            if abs(b - a) < tol * (abs(c) + abs(d)) + 1e-12:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = f(d)
        x = (a + b) / 2
        fx = f(x)
        return x, fx

    def solve_params(self):
        # Objective: minimize squared error at test star
        if len(self.model.masses) == 0:
            self.show_tip("Solver", "No stars available to solve against. Build a galaxy first.")
            return
        vobs = float(self.spinVobs.value())
        # Starting vals
        g0  = float(self.spinG.value())
        dk0 = float(self.spinDk.value())
        wk0 = float(self.spinWk.value())
        # Bounds
        g_lo, g_hi = 0.5, 2.0
        dk_lo, dk_hi = 0.0, 3.0
        wk_lo, wk_hi = 0.0, 1.0

        mode = self.solveObjective.currentText()

        def obj_point(g, dk, wk):
            knobs = DensityKnobs(g_scale=g, dens_k=dk, dens_R=float(self.spinDr.value()), well_k=wk, well_R=float(self.spinWr.value()))
            _, _, vF = self.model.speeds_at_test(knobs)
            return (vF - vobs) ** 2

        def obj_curve(g, dk, wk):
            knobs = DensityKnobs(g_scale=g, dens_k=dk, dens_R=float(self.spinDr.value()), well_k=wk, well_R=float(self.spinWr.value()))
            R_vals, vN_arr, vG_arr, vF_arr = self.curve_vtriple(knobs, nR=60)
            if self.obsCurve is not None and mode.endswith('CSV'):
                R_obs, v_obs = self.obsCurve
                # interpolate obs to our grid
                import numpy as _np
                v_obs_grid = _np.interp(R_vals, R_obs, v_obs, left=v_obs[0], right=v_obs[-1])
            else:
                v_obs_grid = np.full_like(vF_arr, vobs, dtype=float)
            err = float(np.sum((vF_arr - v_obs_grid)**2))
            return err

        def obj(g,dk,wk):
            if mode.startswith('Match test'):
                return obj_point(g,dk,wk)
            else:
                return obj_curve(g,dk,wk)

        g, dk, wk = g0, dk0, wk0
        # Coordinate descent: a few passes
        for _ in range(2):
            if self.chkSolveDk.isChecked():
                def f1(x): return obj(g, max(dk_lo, min(dk_hi, x)), wk)
                dk, _ = self.golden_section(f1, dk_lo, dk_hi)
            if self.chkSolveG.isChecked():
                def f2(x): return obj(max(g_lo, min(g_hi, x)), dk, wk)
                g, _ = self.golden_section(f2, g_lo, g_hi)
            if self.chkSolveWk.isChecked():
                def f3(x): return obj(g, dk, max(wk_lo, min(wk_hi, x)))
                wk, _ = self.golden_section(f3, wk_lo, wk_hi)

        # Apply results
        if self.chkSolveDk.isChecked():
            self.spinDk.setValue(float(dk))
        if self.chkSolveG.isChecked():
            self.spinG.setValue(float(g))
        if self.chkSolveWk.isChecked():
            self.spinWk.setValue(float(wk))
        # Refresh
        self.refresh_physics()
        self.show_tip("Solver", f"Solved values:\nG scale={g:.4f}\nDensity k={dk:.4f}\nWell k={wk:.4f}\nObjective (squared error)={(obj(g, dk, wk)):.4f}")

    # ----- Actions -----
    def on_preset(self, name: str):
        p = PRESET_SIMPLE.get(name, PRESET_SIMPLE[PRESET_OPTIONS[0]])
        self.sN.setValue(int(p['Nstars']))
        self.sR.setValue(float(p['Rdisk']))
        self.sBa.setValue(float(p['BulgeA']))
        self.sHz.setValue(float(p['Hz']))
        self.sBf.setValue(float(p['BulgeFrac']))
        self.sMt.setValue(float(p['Mtot1e10']))
        self.build_from_fields()

    def build_from_fields(self):
        pos, masses = make_simple_galaxy(
            seed=42,
            N_total=int(self.sN.value()),
            R_disk=float(self.sR.value()),
            a_bulge=float(self.sBa.value()),
            hz=float(self.sHz.value()),
            f_bulge=float(self.sBf.value()),
            Mtot=float(self.sMt.value())*1e10,
        )
        self.model.rebuild(pos, masses)
        self.refresh_scene()
        # Build animation state first so base_rxy is defined before physics refresh uses it
        self.build_animation_state()
        self.refresh_physics()

    def refresh_scene(self):
        if len(self.model.masses) == 0:
            self.scatter.set_offsets(np.empty((0,2)))
            self.star.set_data([], [])
            self.com.set_data([], [])
            self.canvas.draw_idle(); return
        self.scatter.set_offsets(self.model.pos[:, :2])
        self.star.set_data([self.model.test_point[0]], [self.model.test_point[1]])
        self.com.set_data([self.model.COM[0]], [self.model.COM[1]])
        R_plot = np.max(np.linalg.norm(self.model.pos[:, :2] - self.model.COM[:2], axis=1)) * 1.15 + 1e-3
        self.ax.set_xlim(self.model.COM[0] - R_plot, self.model.COM[0] + R_plot)
        self.ax.set_ylim(self.model.COM[1] - R_plot, self.model.COM[1] + R_plot)
        self.canvas.draw_idle()
        # Update ring overlay after scene limits change
        self.refresh_ring_overlay()

    def build_animation_state(self):
        if len(self.model.masses) == 0:
            self.base_pos = np.zeros((0,3)); self.base_angles = np.zeros(0); self.base_rxy = np.zeros(0); self.vf_stars = np.zeros(0)
            return
        self.base_pos = self.model.pos.copy()
        self.base_angles = np.arctan2(self.base_pos[:,1]-self.model.COM[1], self.base_pos[:,0]-self.model.COM[0])
        self.base_rxy = np.linalg.norm(self.base_pos[:,:2]-self.model.COM[:2], axis=1)
        # recompute speeds used for animation and percentile
        self.compute_star_speeds_vf()

    def compute_star_speeds_vf(self):
        if len(self.model.masses) == 0:
            self.vf_stars = np.zeros(0); self.vg_stars = np.zeros(0); return
        # Build triple curves for interpolation (Newtonian, GR-like, Final)
        R_vals, vN_arr, vG_arr, vF_arr = self.curve_vtriple(self.knobs, nR=60)
        self.curveR = R_vals; self.curveVF = vF_arr; self.curveVG = vG_arr
        rxy = self.base_rxy
        if len(vF_arr)==0:
            self.vf_stars = np.zeros_like(rxy); self.vg_stars = np.zeros_like(rxy); self.moving_mask = np.zeros_like(rxy, dtype=bool); return
        vf = np.interp(rxy, R_vals, vF_arr, left=float(vF_arr[0]), right=float(vF_arr[-1]))
        vg = np.interp(rxy, R_vals, vG_arr, left=float(vG_arr[0]), right=float(vG_arr[-1]))
        self.vf_stars = vf
        self.vg_stars = vg
        self.moving_mask = vf > 0.0
        self.update_colors()

    def toggle_animation(self, on: bool):
        if on and len(getattr(self, 'base_rxy', []))>0:
            self.animTimer.start()
        else:
            self.animTimer.stop()
        # ensure colors are up-to-date
        self.update_colors()

    def update_colors(self):
        if len(getattr(self, 'vf_stars', []))==0 or len(getattr(self, 'vg_stars', []))==0:
            return
        # spinTol may not be constructed yet during initial startup; default to 10%
        spinTol_widget = getattr(self, 'spinTol', None)
        tol = (float(spinTol_widget.value()) / 100.0) if (spinTol_widget is not None) else 0.10
        vg = np.maximum(self.vg_stars, 1e-6)
        rel = np.abs(self.vf_stars - vg) / vg
        fit_mask = rel <= tol
        moving = (self.vf_stars > 0.0)
        try:
            green = mcolors.to_rgba('#2ca02c', 1.0)
            red = mcolors.to_rgba('#d62728', 1.0)
            color_arr = np.zeros((len(self.vf_stars), 4), dtype=float)  # default transparent (0,0,0,0)
            color_arr[moving & fit_mask] = green
            color_arr[moving & ~fit_mask] = red
            self.scatter.set_facecolors(color_arr)
        except Exception:
            pass
        self.canvas.draw_idle()

    def step_animation(self):
        if not self.chkAnimate.isChecked() or len(self.vf_stars)==0:
            return
        rxy = np.maximum(self.base_rxy, 1e-3)
        v_anim = np.maximum(self.vf_stars, 0.0)
        dtheta = (v_anim / rxy) * (0.001 * float(self.spinAnim.value()))
        self.base_angles = (self.base_angles + dtheta) % (2*np.pi)
        x = self.model.COM[0] + self.base_rxy * np.cos(self.base_angles)
        y = self.model.COM[1] + self.base_rxy * np.sin(self.base_angles)
        self.scatter.set_offsets(np.column_stack([x,y]))
        # Move highlighted test star
        idx = self.model.test_idx
        if idx is not None and 0 <= idx < len(x):
            self.star.set_data([x[idx]],[y[idx]])
        self.canvas.draw_idle()

    def refresh_physics(self):
        # Compute effective dens_R
        dens_R_eff = float(self.spinDr.value())
        try:
            if self.chkDrFrac.isChecked() and len(self.model.masses)>0:
                Rchar = float(np.max(np.linalg.norm(self.model.pos - self.model.COM, axis=1)))
                dens_R_eff = float(self.spinDrFrac.value()) * max(1e-6, Rchar)
        except Exception:
            pass
        # Compute boundary R for current preset based on MW fraction rule if enabled
        def boundary_for_preset(preset_name: str) -> float:
            try:
                mw_Rdisk = float(PRESET_SIMPLE["MW-like disk"]["Rdisk"])  # kpc
                this_Rdisk = float(PRESET_SIMPLE.get(preset_name, {}).get("Rdisk", mw_Rdisk))
                mw_boundary = float(self.spinBoundaryMW.value())
                if self.chkBoundaryUseMWFrac.isChecked() and mw_Rdisk > 0:
                    frac = mw_boundary / mw_Rdisk
                    return frac * this_Rdisk
                else:
                    return mw_boundary
            except Exception:
                return float(self.spinBoundaryMW.value())
        boundary_R_now = boundary_for_preset(self.preset.currentText())
        self.knobs = DensityKnobs(
            g_scale=float(self.spinG.value()),
            dens_k=float(self.spinDk.value()), dens_R=dens_R_eff,
            dens_alpha=float(self.spinDa.value()), dens_thresh_frac=float(self.spinDth.value()),
            well_k=float(self.spinWk.value()), well_R=float(self.spinWr.value()),
            boundary_R=float(boundary_R_now),
        )
        vN, vG, vF = self.model.speeds_at_test(self.knobs)
        # Update low-density enhancement patch visibility/location
        try:
            Rn = float(self.knobs.dens_R)
            if len(self.model.masses) > 0 and Rn > 0:
                p = self.model.test_point
                d2 = np.sum((self.model.pos - p) ** 2, axis=1)
                m_local = float(self.model.masses[d2 <= (Rn*Rn)].sum())
                vol = (4.0/3.0) * math.pi * (Rn**3)
                rho_local = m_local / max(1e-30, vol)
                Rchar = float(np.max(np.linalg.norm(self.model.pos - self.model.COM, axis=1))) if len(self.model.masses) else 1.0
                Vchar = (4.0/3.0) * math.pi * max(1e-30, Rchar**3)
                rho_ref = float(self.model.masses.sum()) / Vchar if Vchar>0 else 0.0
                lowD = (rho_ref > 0.0) and (rho_local < self.knobs.dens_thresh_frac * rho_ref)
                self.lowDensPatch.center = (p[0], p[1])
                self.lowDensPatch.set_radius(Rn)
                self.lowDensPatch.set_visible(lowD)
            else:
                self.lowDensPatch.set_visible(False)
        except Exception:
            self.lowDensPatch.set_visible(False)
        vobs = float(self.spinVobs.value())
        dv = vF - vobs
        match_pct = percent_closeness(vF, vobs)
        # update curve cache and star speeds (for percentile & animation)
        self.compute_star_speeds_vf()
        # Update ring overlay when physics refreshed (rho_ref depends on mass distribution extents)
        self.refresh_ring_overlay()
        total_mass = float(self.model.masses.sum())
        far_R = float(np.linalg.norm(self.model.test_point[:2] - self.model.COM[:2])) if len(self.model.masses) else 0.0
        pct = 0.0
        if len(getattr(self, 'vf_stars', []))>0:
            import numpy as _np
            pct = 100.0 * (_np.sum(self.vf_stars <= vF) / max(1, len(self.vf_stars)))
        ring_extra = ""
        if getattr(self, 'grpRings', None) and self.grpRings.isChecked() and getattr(self, 'chkShowRings', None) and self.chkShowRings.isChecked():
            try:
                total_bins = int(self.spinRBins.value())
                shaded = int(getattr(self, '_ring_shaded', 0))
                rho_ref_last = float(getattr(self, '_rho_ref_last', 0.0))
                rabs = float(self.spinRAbs.value())
                rfrac = float(self.spinRThresh.value())
                ring_extra = f"\nRing overlay: shaded {shaded}/{total_bins} | rho_ref={rho_ref_last:.3e} Msun/kpc^3 | thresh_abs={rabs:.3e} ({rfrac:.2f}×rho_ref)"
            except Exception:
                ring_extra = ""
        self.infoLabel.setText(
            f"Preset: {self.preset.currentText()} | N stars: {len(self.model.masses)} | Total mass: {total_mass:,.2e} Msun\n"
            f"Farthest-star radius: R = {far_R:.2f} kpc\n"
            f"G={self.knobs.g_scale:.2f} | α={self.knobs.dens_alpha:.2f} | dens_R={self.knobs.dens_R:.2f} kpc | boundary_R={self.knobs.boundary_R:.2f} kpc | thresh×rho_ref={self.knobs.dens_thresh_frac:.2f} | "
            f"Well k={self.knobs.well_k:.2f} @ R={self.knobs.well_R:.2f}\n"
            f"Speeds (km/s): Newtonian={vN:.2f}  GR-like={vG:.2f}  Final={vF:.2f} | Observed={vobs:.1f}  Δ={dv:+.2f} km/s\n"
            f"Match to observed at test star: {match_pct:.1f}%{ring_extra}\n"
            f"Test star speed percentile among stars: {pct:.1f}% (green=GR fit, red=outliers, gold=test star)"
        )

    def open_vectors(self):
        dlg = VectorsDialog(self.model, self.knobs, self)
        dlg.exec()

    def sync_density_thresholds(self, from_abs: bool = False):
        # synchronize fraction and absolute thresholds using current rho_ref
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            rho_ref = getattr(self, '_rho_ref_last', None)
            if rho_ref is None or rho_ref <= 0:
                # attempt to compute rho_ref quickly
                if len(self.model.masses) > 0:
                    Rchar = float(np.max(np.linalg.norm(self.model.pos - self.model.COM, axis=1)))
                    Vchar = (4.0/3.0) * math.pi * max(1e-30, Rchar**3)
                    rho_ref = float(self.model.masses.sum()) / Vchar if Vchar > 0 else 0.0
                else:
                    rho_ref = 0.0
            if rho_ref > 0:
                if from_abs:
                    # Update fraction from absolute for both eval and ring controls
                    try:
                        self.spinDth.setValue(float(self.spinDAbs.value())/rho_ref)
                    except Exception:
                        pass
                    try:
                        self.spinRThresh.setValue(float(self.spinRAbs.value())/rho_ref)
                    except Exception:
                        pass
                else:
                    # Update absolute from fraction for both eval and ring controls
                    try:
                        self.spinDAbs.setValue(float(self.spinDth.value())*rho_ref)
                    except Exception:
                        pass
                    try:
                        self.spinRAbs.setValue(float(self.spinRThresh.value())*rho_ref)
                    except Exception:
                        pass
        finally:
            self._sync_guard = False

    def refresh_ring_overlay(self, abs_changed: bool = False):
        # Clear existing ring patches
        try:
            for p in self.ringPatches:
                p.remove()
        except Exception:
            pass
        self.ringPatches = []
        if not self.grpRings.isChecked() or not self.chkShowRings.isChecked():
            self.canvas.draw_idle(); return
        if len(self.model.masses) == 0:
            self.canvas.draw_idle(); return
        import numpy as _np
        # Compute rho_ref from 3D volume as in compute_G_eff
        Rchar = float(_np.max(_np.linalg.norm(self.model.pos - self.model.COM, axis=1))) if len(self.model.masses) else 1.0
        Vchar = (4.0/3.0) * math.pi * max(1e-30, Rchar**3)
        rho_ref = float(self.model.masses.sum()) / Vchar if Vchar > 0 else 0.0
        self._rho_ref_last = rho_ref
        # Sync fraction/absolute controls based on which changed
        self.sync_density_thresholds(from_abs=abs_changed)
        # Compute surface density per ring (annuli) using 2D projected radii
        bins = int(self.spinRBins.value())
        Rmax = float(_np.max(_np.linalg.norm(self.model.pos[:, :2] - self.model.COM[:2], axis=1)))
        if not (Rmax > 0 and bins > 0):
            self.canvas.draw_idle(); return
        edges = _np.linspace(0.0, Rmax, bins+1)
        Rxy = _np.linalg.norm(self.model.pos[:, :2] - self.model.COM[:2], axis=1)
        mass = self.model.masses
        for i in range(bins):
            r0, r1 = float(edges[i]), float(edges[i+1])
            mask = (Rxy >= r0) & (Rxy < r1)
            m = float(mass[mask].sum())
            area = math.pi * (r1*r1 - r0*r0)
            sigma = m / max(1e-30, area)  # Msun / kpc^2
            # Convert to an equivalent 3D-like density by dividing by an effective height (use Hz from preset approximation or 2*dens_R)
            hz_eff = max(1e-3, float(self.spinRHz.value()))
            rho_equiv = sigma / (2.0*hz_eff)
            thresh_abs = float(self.spinRAbs.value()) if rho_ref > 0 else 0.0
            if rho_ref > 0 and (rho_equiv < (float(self.spinRThresh.value()) * rho_ref) or (thresh_abs > 0.0 and rho_equiv < thresh_abs)):
                # Draw a translucent annulus for this ring
                patch = Wedge(center=(self.model.COM[0], self.model.COM[1]), r=r1, theta1=0, theta2=360, width=(r1-r0),
                              facecolor=(0.5,0.5,0.5,0.15), edgecolor=(0.5,0.5,0.5,0.35))
                self.ax.add_patch(patch)
                self.ringPatches.append(patch)
        self.canvas.draw_idle()

    def save_curve_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save rotation_curve.csv", str(Path.cwd()/"rotation_curve.csv"))
        if not path:
            return
        if len(self.model.masses) == 0:
            return
        R_vals, vN_arr, vG_arr, vF_arr = self.curve_vtriple(self.knobs, nR=60)
        rows = []
        for i, R in enumerate(R_vals):
            rows.append([R, float(vN_arr[i]), float(vG_arr[i]), float(vF_arr[i]), self.knobs.g_scale, self.knobs.dens_k, self.knobs.dens_R, self.knobs.well_k, self.knobs.well_R])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["R_kpc","v_newton_kms","v_gr_like_kms","v_final_kms","G_scale","dens_k","dens_R","well_k","well_R"])
            for r in rows:
                w.writerow([float(x) for x in r])

    # ----- Solution profiles -----
    def _solutions_path(self) -> Path:
        return Path.cwd()/"solutions.json"

    def save_solution_profile(self):
        # Prompt for profile name
        default_name = f"{self.preset.currentText()}"
        name, ok = QInputDialog.getText(self, "Save Solution", "Profile name:", text=default_name)
        if not ok or not str(name).strip():
            return
        name = str(name).strip()
        entry = dict(
            G=float(self.spinG.value()), Dk=float(self.spinDk.value()), Dr=float(self.spinDr.value()),
            Da=float(self.spinDa.value()), Dth=float(self.spinDth.value()),
            Wk=float(self.spinWk.value()), Wr=float(self.spinWr.value()), Vobs=float(self.spinVobs.value()),
            Preset=str(self.preset.currentText())
        )
        path = self._solutions_path()
        data = {}
        if path.exists():
            try:
                data = json.load(open(path, 'r'))
            except Exception:
                data = {}
        data[name] = entry
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.show_tip("Solutions", f"Saved solution '{name}' to {path.name}.")

    def load_solution_profile(self):
        path = self._solutions_path()
        if not path.exists():
            self.show_tip("Solutions", "No solutions.json found in current directory.")
            return
        try:
            data = json.load(open(path, 'r'))
        except Exception:
            self.show_tip("Solutions", "Could not parse solutions.json")
            return
        if not data:
            self.show_tip("Solutions", "solutions.json has no entries.")
            return
        names = list(data.keys())
        name, ok = QInputDialog.getItem(self, "Load Solution", "Choose profile:", names, 0, False)
        if not ok:
            return
        ent = data.get(name)
        if not ent:
            return
        self.spinG.setValue(float(ent.get('G', self.spinG.value())))
        self.spinDk.setValue(float(ent.get('Dk', self.spinDk.value())))
        self.spinDr.setValue(float(ent.get('Dr', self.spinDr.value())))
        self.spinDa.setValue(float(ent.get('Da', self.spinDa.value())))
        self.spinDth.setValue(float(ent.get('Dth', self.spinDth.value())))
        self.spinWk.setValue(float(ent.get('Wk', self.spinWk.value())))
        self.spinWr.setValue(float(ent.get('Wr', self.spinWr.value())))
        vobs = float(ent.get('Vobs', self.spinVobs.value()))
        self.spinVobs.setValue(vobs)
        self.refresh_physics()
        self.show_tip("Solutions", f"Loaded solution '{name}'.")

    def open_compare_dialog(self):
        dlg = CompareAcrossPresetsDialog(self)
        dlg.exec()

    def apply_knobs(self, knobs: DensityKnobs):
        self.spinG.setValue(float(knobs.g_scale))
        self.spinDk.setValue(float(knobs.dens_k))
        self.spinDr.setValue(float(knobs.dens_R))
        self.spinWk.setValue(float(knobs.well_k))
        self.spinWr.setValue(float(knobs.well_R))
        self.refresh_physics()


def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

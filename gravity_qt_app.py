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


# Try PyQt6 first, fall back to PyQt5
try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
        QSplitter, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
        QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog,
        QSizePolicy)
    QT_IS_6 = True
except Exception:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
        QSplitter, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
        QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog,
        QSizePolicy
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
    well_k: float = 0.0
    well_R: float = 2.0


class GravityModel:
    def __init__(self):
        self.pos = np.zeros((0, 3))
        self.masses = np.zeros((0,))
        self.COM = np.zeros(3)
        self.test_point = np.zeros(3)
        self.test_idx = None
        self.cache_well = {"version": -1, "k": None, "R": None, "scales": None}
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
        self.cache_well["version"] = -1

    def compute_G_eff(self, p: np.ndarray, knobs: DensityKnobs) -> float:
        G0 = G_AST * float(knobs.g_scale)
        k = float(knobs.dens_k); Rn = max(1e-6, float(knobs.dens_R))
        if k <= 0 or len(self.masses) == 0:
            return G0
        d2 = np.sum((self.pos - p) ** 2, axis=1)
        m_local = float(self.masses[d2 <= (Rn * Rn)].sum())
        vol = (4.0 / 3.0) * math.pi * (Rn ** 3)
        rho_local = m_local / max(1e-30, vol)
        if len(self.masses) > 0:
            Rchar = float(np.max(np.linalg.norm(self.pos - self.COM, axis=1)))
            Vchar = (4.0 / 3.0) * math.pi * max(1e-30, Rchar ** 3)
            rho_ref = float(self.masses.sum()) / Vchar
        else:
            rho_ref = rho_local
        boost = k * max(0.0, (rho_ref / max(rho_local, 1e-30)) - 1.0) if rho_ref > 0 else 0.0
        mult = max(0.1, min(10.0, 1.0 + boost))
        return G0 * mult

    def well_scales(self, knobs: DensityKnobs) -> np.ndarray:
        k = float(knobs.well_k); Rw = max(1e-6, float(knobs.well_R))
        if len(self.masses) == 0 or k <= 0:
            return np.ones(len(self.masses))
        c = self.cache_well
        if c["scales"] is not None and c["version"] == self.version and c["k"] == k and c["R"] == Rw:
            return c["scales"]
        Rchar = float(np.max(np.linalg.norm(self.pos - self.COM, axis=1))) if len(self.masses) else 1.0
        Vchar = (4.0 / 3.0) * math.pi * max(1e-30, Rchar ** 3)
        rho_ref = float(self.masses.sum()) / Vchar if Vchar > 0 else 0.0
        N = len(self.masses)
        scales = np.ones(N)
        batch = 1000
        for i0 in range(0, N, batch):
            i1 = min(N, i0 + batch)
            d2 = np.sum((self.pos[i0:i1, None, :] - self.pos[None, :, :]) ** 2, axis=2)
            m_local = (d2 <= (Rw * Rw)) @ self.masses
            vol = (4.0 / 3.0) * math.pi * (Rw ** 3)
            rho_local = m_local / max(1e-30, vol)
            boost = k * np.maximum(0.0, (rho_ref / np.maximum(rho_local, 1e-30)) - 1.0)
            scales[i0:i1] = np.clip(1.0 + boost, 0.1, 10.0)
        self.cache_well.update({"version": self.version, "k": k, "R": Rw, "scales": scales})
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gravity Calculator — Qt Frontend")
        self.resize(1200, 800)

        # Core state
        self.model = GravityModel()
        self.knobs = DensityKnobs()

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
        self.spinDr = QDoubleSpinBox(); self.spinDr.setRange(0.05, 20.0); self.spinDr.setDecimals(3); self.spinDr.setValue(2.0)
        self.spinWk = QDoubleSpinBox(); self.spinWk.setRange(0.0, 5.0); self.spinWk.setDecimals(3); self.spinWk.setValue(0.0)
        self.spinWr = QDoubleSpinBox(); self.spinWr.setRange(0.05, 20.0); self.spinWr.setDecimals(3); self.spinWr.setValue(2.0)
        self.spinVobs = QDoubleSpinBox(); self.spinVobs.setRange(0.0, 1000.0); self.spinVobs.setDecimals(1); self.spinVobs.setValue(220.0)
        add_row_with_tip(form, "G scale", self.spinG,
            "Global multiplier on Newton's constant G. Speeds scale roughly as sqrt(G). Increase to raise all speeds.")
        add_row_with_tip(form, "Density k", self.spinDk,
            "Point-based modifier: if local density at the test star is low versus global, G_eff increases by k*(rho_ref/rho_local - 1). Higher k flattens the outer curve.")
        add_row_with_tip(form, "Density R (kpc)", self.spinDr,
            "Neighborhood size for computing local density at the evaluation point. 1–5 kpc typical. Smaller R makes the effect more local.")
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
        rightLayout.addStretch(1)

        # Matplotlib artists
        self.scatter = self.ax.scatter([], [], s=5, alpha=0.6)
        self.star, = self.ax.plot([], [], marker='*', ms=10)
        self.com, = self.ax.plot([], [], marker='x', ms=8)
        self.ax.set_xlabel("x (kpc)"); self.ax.set_ylabel("y (kpc)")
        self.ax.set_title("Gravity Calculator — Qt")

        # Events (debounced)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh_physics)

        self.preset.currentTextChanged.connect(self.on_preset)
        self.btnBuild.clicked.connect(self.build_from_fields)
        self.btnVectors.clicked.connect(self.open_vectors)
        self.btnSaveCSV.clicked.connect(self.save_curve_csv)
        self.btnSolve.clicked.connect(self.solve_params)
        self.btnLoadObs.clicked.connect(self.load_obs_curve)
        for w in [self.spinG, self.spinDk, self.spinDr, self.spinWk, self.spinWr, self.spinVobs]:
            w.valueChanged.connect(self.schedule_refresh)
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
            R_vals, vF_arr = self.curve_vfinal(knobs, nR=60)
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
        self.refresh_physics()
        self.build_animation_state()

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
            self.vf_stars = np.zeros(0); return
        R_vals, vF = self.curve_vfinal(self.knobs, nR=60)
        self.curveR = R_vals; self.curveVF = vF
        rxy = self.base_rxy
        if len(vF)==0:
            self.vf_stars = np.zeros_like(rxy); return
        vf = np.interp(rxy, R_vals, vF, left=float(vF[0]), right=float(vF[-1]))
        self.vf_stars = vf

    def toggle_animation(self, on: bool):
        if on and len(getattr(self, 'base_rxy', []))>0:
            self.animTimer.start()
        else:
            self.animTimer.stop()

    def step_animation(self):
        if not self.chkAnimate.isChecked() or len(self.vf_stars)==0:
            return
        dtheta = (self.vf_stars / np.maximum(self.base_rxy, 1e-3)) * (0.001 * float(self.spinAnim.value()))
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
        self.knobs = DensityKnobs(
            g_scale=float(self.spinG.value()),
            dens_k=float(self.spinDk.value()), dens_R=float(self.spinDr.value()),
            well_k=float(self.spinWk.value()), well_R=float(self.spinWr.value()),
        )
        vN, vG, vF = self.model.speeds_at_test(self.knobs)
        vobs = float(self.spinVobs.value())
        dv = vF - vobs
        # update curve cache and star speeds (for percentile & animation)
        self.compute_star_speeds_vf()
        total_mass = float(self.model.masses.sum())
        far_R = float(np.linalg.norm(self.model.test_point[:2] - self.model.COM[:2])) if len(self.model.masses) else 0.0
        pct = 0.0
        if len(getattr(self, 'vf_stars', []))>0:
            import numpy as _np
            pct = 100.0 * (_np.sum(self.vf_stars <= vF) / max(1, len(self.vf_stars)))
        self.infoLabel.setText(
            f"Preset: {self.preset.currentText()} | N stars: {len(self.model.masses)} | Total mass: {total_mass:,.2e} Msun\n"
            f"Farthest-star radius: R = {far_R:.2f} kpc\n"
            f"G={self.knobs.g_scale:.2f} | Density k={self.knobs.dens_k:.2f} @ R={self.knobs.dens_R:.2f} | "
            f"Well k={self.knobs.well_k:.2f} @ R={self.knobs.well_R:.2f}\n"
            f"Speeds (km/s): Newtonian={vN:.2f}  GR-like={vG:.2f}  Final={vF:.2f} | Observed={vobs:.1f}  Δ={dv:+.2f} km/s\n"
            f"Test star speed percentile among stars: {pct:.1f}%"
        )

    def open_vectors(self):
        dlg = VectorsDialog(self.model, self.knobs, self)
        dlg.exec()

    def save_curve_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save rotation_curve.csv", str(Path.cwd()/"rotation_curve.csv"))
        if not path:
            return
        if len(self.model.masses) == 0:
            return
        R_vals, vF = self.curve_vfinal(self.knobs, nR=60)
        rows = []
        for i, R in enumerate(R_vals):
            # Recompute N and G for completeness (cheap to keep code clarity)
            p = self.model.COM + np.array([R,0,0])
            newton, gr_like, final = self.model.per_body_vectors(p, self.knobs)
            vN = circular_speed_from_accel(newton.sum(axis=0), p - self.model.COM)
            vG = circular_speed_from_accel(gr_like.sum(axis=0), p - self.model.COM)
            rows.append([R, vN, vG, vF[i], self.knobs.g_scale, self.knobs.dens_k, self.knobs.dens_R, self.knobs.well_k, self.knobs.well_R])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["R_kpc","v_newton_kms","v_gr_like_kms","v_final_kms","G_scale","dens_k","dens_R","well_k","well_R"])
            for r in rows:
                w.writerow([float(x) for x in r])


def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

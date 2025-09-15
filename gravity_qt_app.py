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
        QSplitter, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
        QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog,
        QSizePolicy
    )
    QT_IS_6 = True
except Exception:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
        QSplitter, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
        QGroupBox, QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem, QDialog,
        QSizePolicy
    )
    QT_IS_6 = False

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
        self.cache_well = {"version": -1, "k": None, "R": None, "scales": None}
        self.version = 0

    def rebuild(self, pos: np.ndarray, masses: np.ndarray):
        self.pos = pos; self.masses = masses
        if len(masses) == 0 or masses.sum() <= 0:
            self.COM = np.zeros(3)
            self.test_point = np.zeros(3)
        else:
            self.COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
            R_all = np.linalg.norm(pos - self.COM, axis=1)
            self.test_point = pos[int(np.argmax(R_all))]
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

        # Preset
        self.preset = QComboBox(); self.preset.addItems(PRESET_OPTIONS)
        form.addRow("Preset", self.preset)

        # Core knobs
        self.spinG = QDoubleSpinBox(); self.spinG.setRange(0.1, 5.0); self.spinG.setDecimals(3); self.spinG.setValue(1.0)
        self.spinDk = QDoubleSpinBox(); self.spinDk.setRange(0.0, 5.0); self.spinDk.setDecimals(3); self.spinDk.setValue(0.0)
        self.spinDr = QDoubleSpinBox(); self.spinDr.setRange(0.05, 20.0); self.spinDr.setDecimals(3); self.spinDr.setValue(2.0)
        self.spinWk = QDoubleSpinBox(); self.spinWk.setRange(0.0, 5.0); self.spinWk.setDecimals(3); self.spinWk.setValue(0.0)
        self.spinWr = QDoubleSpinBox(); self.spinWr.setRange(0.05, 20.0); self.spinWr.setDecimals(3); self.spinWr.setValue(2.0)
        self.spinVobs = QDoubleSpinBox(); self.spinVobs.setRange(0.0, 1000.0); self.spinVobs.setDecimals(1); self.spinVobs.setValue(220.0)
        form.addRow("G scale", self.spinG)
        form.addRow("Density k", self.spinDk)
        form.addRow("Density R (kpc)", self.spinDr)
        form.addRow("Well k", self.spinWk)
        form.addRow("Well R (kpc)", self.spinWr)
        form.addRow("Observed v (km/s)", self.spinVobs)

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
        f2.addRow("N stars", self.sN)
        f2.addRow("R disk (kpc)", self.sR)
        f2.addRow("Bulge a (kpc)", self.sBa)
        f2.addRow("Thickness hz", self.sHz)
        f2.addRow("Bulge frac", self.sBf)
        f2.addRow("Total mass (1e10 Msun)", self.sMt)
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
        for w in [self.spinG, self.spinDk, self.spinDr, self.spinWk, self.spinWr, self.spinVobs]:
            w.valueChanged.connect(self.schedule_refresh)
        for w in [self.sN, self.sR, self.sBa, self.sHz, self.sBf, self.sMt]:
            w.valueChanged.connect(self.schedule_rebuild)
        grpStruct.toggled.connect(lambda _checked: None)  # keep state; no action needed

        # Initialize
        self.on_preset(self.preset.currentText())

    # ----- Scheduling helpers -----
    def schedule_refresh(self):
        self.timer.start()

    def schedule_rebuild(self):
        self.build_from_fields()

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

    def refresh_physics(self):
        self.knobs = DensityKnobs(
            g_scale=float(self.spinG.value()),
            dens_k=float(self.spinDk.value()), dens_R=float(self.spinDr.value()),
            well_k=float(self.spinWk.value()), well_R=float(self.spinWr.value()),
        )
        vN, vG, vF = self.model.speeds_at_test(self.knobs)
        vobs = float(self.spinVobs.value())
        dv = vF - vobs
        total_mass = float(self.model.masses.sum())
        far_R = float(np.linalg.norm(self.model.test_point[:2] - self.model.COM[:2])) if len(self.model.masses) else 0.0
        self.infoLabel.setText(
            f"Preset: {self.preset.currentText()} | N stars: {len(self.model.masses)} | Total mass: {total_mass:,.2e} Msun\n"
            f"Farthest-star radius: R = {far_R:.2f} kpc\n"
            f"G={self.knobs.g_scale:.2f} | Density k={self.knobs.dens_k:.2f} @ R={self.knobs.dens_R:.2f} | "
            f"Well k={self.knobs.well_k:.2f} @ R={self.knobs.well_R:.2f}\n"
            f"Speeds (km/s): Newtonian={vN:.2f}  GR-like={vG:.2f}  Final={vF:.2f} | "
            f"Observed={vobs:.1f}  Δ={dv:+.2f} km/s"
        )

    def open_vectors(self):
        dlg = VectorsDialog(self.model, self.knobs, self)
        dlg.exec()

    def save_curve_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save rotation_curve.csv", str(Path.cwd()/"rotation_curve.csv"))
        if not path:
            return
        # Build simple 1D curve out to edge
        if len(self.model.masses) == 0:
            return
        Rmax = float(np.max(np.linalg.norm(self.model.pos - self.model.COM, axis=1)))
        R_vals = np.linspace(max(0.2, 0.02*Rmax), Rmax, 60)
        rows = []
        for R in R_vals:
            p = self.model.COM + np.array([R,0,0])
            newton, gr_like, final = self.model.per_body_vectors(p, self.knobs)
            vN = circular_speed_from_accel(newton.sum(axis=0), p - self.model.COM)
            vG = circular_speed_from_accel(gr_like.sum(axis=0), p - self.model.COM)
            vF = circular_speed_from_accel(final.sum(axis=0), p - self.model.COM)
            rows.append([R, vN, vG, vF, self.knobs.g_scale, self.knobs.dens_k, self.knobs.dens_R, self.knobs.well_k, self.knobs.well_R])
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

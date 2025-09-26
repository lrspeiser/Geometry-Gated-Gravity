# -*- coding: utf-8 -*-
"""
root-m/pde/handoff.py

Unified handoff utilities for multi-order gravity pipeline:
- O1: Newtonian (always on)
- O2: Second-Order (SOG or G³ tail) — engages in low-Σ / low-g environments via w2 gate
- O3: Third-Order lensing-only booster — engages in extended, cluster-like, low-curvature regimes and for tiny test mass

This module does not implement new physics; it orchestrates gating/weights
for the already-implemented components:
- second_order.S_env (local gate for O2)
- concepts.cluster_lensing.g3_cluster_tests.o3_lensing.nonlocal_env_boost (non-local env gate for O3)

Usage outline:
- Compute g1 and a candidate g2 (e.g., SOG or G³ tail) on target radii
- Compute w2 = S_env(Sigma_loc, g1; thresholds)
- Dynamics: g_dyn = g1 + w2 * g2
- Lensing-only: g_lens = g_dyn * B_env(R,Sigma) * B_test(m_test)

All arrays are numpy 1D on a consistent radial grid in kpc.
"""
from __future__ import annotations
import numpy as np
import importlib.util
from pathlib import Path

# Optional dynamic imports for consistency with existing modules
_sog_path = Path(__file__).resolve().parent / 'second_order.py'
_o3_path = Path(__file__).resolve().parents[2] / 'concepts' / 'cluster_lensing' / 'g3_cluster_tests' / 'o3_lensing.py'

second_order = None
if _sog_path.exists():
    _spec = importlib.util.spec_from_file_location('second_order', str(_sog_path))
    second_order = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(second_order)

o3_lensing = None
if _o3_path.exists():
    _spec_o3 = importlib.util.spec_from_file_location('o3_lensing', str(_o3_path))
    o3_lensing = importlib.util.module_from_spec(_spec_o3)
    _spec_o3.loader.exec_module(o3_lensing)


def w2_gate(Sigma_loc_Msun_pc2: np.ndarray,
            g1_kms2_per_kpc: np.ndarray,
            gate_params: dict) -> np.ndarray:
    """Second-order (O2) handoff gate in [0,1].
    Wraps second_order.S_env if available, else uses the same rational form.
    """
    Sigma = np.asarray(Sigma_loc_Msun_pc2, float)
    g1 = np.asarray(g1_kms2_per_kpc, float)
    if second_order is not None:
        return second_order.S_env(Sigma, g1,
                                  float(gate_params.get('Sigma_star', 100.0)),
                                  float(gate_params.get('g_star', 1200.0)),
                                  float(gate_params.get('aSigma', 2.0)),
                                  float(gate_params.get('ag', 2.0)))
    # fallback identical to S_env definition
    s1 = 1.0 / (1.0 + (np.maximum(Sigma, 0.0) / max(gate_params.get('Sigma_star', 100.0), 1e-12))**float(gate_params.get('aSigma', 2.0)))
    s2 = 1.0 / (1.0 + (np.maximum(g1, 0.0) / max(gate_params.get('g_star', 1200.0), 1e-12))**float(gate_params.get('ag', 2.0)))
    return s1 * s2


def dynamics_with_handoff(g1_kms2_per_kpc: np.ndarray,
                          g2_kms2_per_kpc: np.ndarray,
                          w2: np.ndarray) -> np.ndarray:
    """Compute dynamics field with O1→O2 handoff.
    g_dyn = g1 + w2 * g2
    """
    g1 = np.asarray(g1_kms2_per_kpc, float)
    g2 = np.asarray(g2_kms2_per_kpc, float)
    return g1 + np.asarray(w2, float) * g2


def lensing_with_o3(g_dyn_kms2_per_kpc: np.ndarray,
                     R_kpc: np.ndarray,
                     Sigma_Msun_pc2: np.ndarray,
                     o3_params: dict,
                     m_test_Msun: float = 0.0) -> np.ndarray:
    """Apply O3 lensing-only booster as the last stage of the pipeline.
    Wraps o3_lensing.apply_o3_lensing.
    """
    if o3_lensing is None:
        # No O3 available; pass through
        return np.asarray(g_dyn_kms2_per_kpc, float)
    return o3_lensing.apply_o3_lensing(g_dyn_kms2_per_kpc, R_kpc, Sigma_Msun_pc2, o3_params, m_test_Msun=m_test_Msun)

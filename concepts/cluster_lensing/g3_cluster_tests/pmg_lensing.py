# g3_cluster_tests/pmg_lensing.py
import numpy as np

def curvature_proxy(logr, logrho):
    d1 = np.gradient(logrho, logr)
    d2 = np.gradient(d1, logr)
    kappa = np.maximum(d2, 0.0)
    return kappa / (1.0 + np.abs(d1))

def env_gate(r_kpc, Sigma_Msun_pc2, params, kappa_norm=None):
    # Low-Σ gate: S ~ 1 at low Σ, decays at high Σ (achieves screening automatically)
    Sigma0     = params.get('Sigma0', 10.0)
    beta       = params.get('beta', 0.6)
    S_sigma    = 1.0 / (1.0 + (np.maximum(Sigma_Msun_pc2,0.0) / Sigma0)**beta)
    # Radial gate ensuring integrability and finiteness (x/(1+x))
    Rboost     = params.get('Rboost_kpc', 500.0)
    q          = params.get('q', 2.0)
    x          = (np.maximum(r_kpc,1e-9) / Rboost)**q
    R_gate     = x / (1.0 + x)
    # Optional curvature factor (0.. ~1)
    c_curv     = params.get('c_curv', 0.0)
    nu         = params.get('nu', 1.0)
    curv_fac   = 1.0
    if kappa_norm is not None and c_curv > 0.0:
        curv_fac = (kappa_norm / (1.0 + kappa_norm))**nu
    return S_sigma * R_gate * curv_fac

def lensing_amplifier(r_kpc, Sigma_Msun_pc2, params, m_test_Msun, logr=None, logrho=None, tracer="photon"):
    # Achromatic photon branch: use m_eff = m_floor for photons
    mref   = params.get('mref_Msun', 1.0)
    mfloor = params.get('mfloor_Msun', 1e-10)
    chi    = params.get('chi', 1.5)
    if tracer == "photon":
        m_eff = mfloor
    else:
        m_eff = m_test_Msun + mfloor
    mass_fac = (mref / (m_eff))**chi
    kappa_norm = None
    if logr is not None and logrho is not None:
        kappa_norm = curvature_proxy(logr, logrho)
    E = env_gate(r_kpc, Sigma_Msun_pc2, params, kappa_norm=kappa_norm)
    A0 = params.get('A0', 0.1)
    return 1.0 + A0 * E * mass_fac

def g_lens_from_gdyn(g_dyn, r_kpc, Sigma_Msun_pc2, params, m_test_Msun,
                     logr=None, logrho=None, tracer="photon"):
    amp = lensing_amplifier(r_kpc, Sigma_Msun_pc2, params, m_test_Msun,
                            logr=logr, logrho=logrho, tracer=tracer)
    return g_dyn * amp

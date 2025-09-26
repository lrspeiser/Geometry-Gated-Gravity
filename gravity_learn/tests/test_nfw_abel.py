import numpy as np
from gravity_learn.physics.poisson_abel import nfw_rho, nfw_sigma_analytic, sigma_from_rho_abel


def test_nfw_sigma_matches_analytic_light():
    # Define an NFW profile
    rho_s = 1.0
    r_s = 1.0
    # radial grid for 3D density
    r = np.geomspace(1e-2, 50.0, 1024)
    rho = nfw_rho(r, rho_s, r_s)
    # project to Σ(R)
    R = np.geomspace(1e-2, 20.0, 256)
    Sigma_num = sigma_from_rho_abel(R, r, rho)
    Sigma_ana = nfw_sigma_analytic(R, rho_s, r_s)
    # Compare over a safe interior range
    sel = (R > 1e-2) & (R < 10.0)
    rel_err = np.abs(Sigma_num[sel] - Sigma_ana[sel]) / np.maximum(np.abs(Sigma_ana[sel]), 1e-12)
    assert np.median(rel_err) < 0.1, f"median rel err {np.median(rel_err):.3g} too high"
    assert np.max(rel_err) < 0.25, f"max rel err {np.max(rel_err):.3g} too high"

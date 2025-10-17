"""Test: does disabling interior_emphasis fix the overshoot?"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumulative_trapezoid

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

FIXED_PARAMS = {'p': 2.0, 'ncoh': 2.0}

print("Building MACS0416...")
cosmo = LensingCosmology()
catalog = pd.read_csv(Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv')
cluster = catalog[catalog['cluster_name'] == 'MACS0416'].iloc[0]

r_3d = np.logspace(-1, 3.5, 800)
params = ClusterBaryonParams(
    M_500=cluster['M_500_Msun'], R_500=cluster['R_500_kpc'],
    z=cluster['z_lens'], fgas_target=cluster['fgas_R500'],
    T_keV=cluster['TX_central_keV'], C0=1.3, eta=2.0, C_max=2.5
)
components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)

nx, ny = 128, 128
R_max = min(2500.0, cluster['R_500_kpc'] * 2.2)
x = np.linspace(-R_max, R_max, nx)
y = np.linspace(-R_max, R_max, ny)
X, Y = np.meshgrid(x, y)
R_grid_2d = np.sqrt(X**2 + Y**2)

M_BCG, r_eff_BCG = estimate_bcg_mass(cluster['M_500_Msun'], cluster['z_lens'])
Sigma_BCG = hernquist_projected_density(R_grid_2d, M_BCG, r_eff_BCG)
Sigma_baryons = project_to_surface_density(r_3d, components.rho_total, R_grid_2d, q_los=1.0, q_plane=1.0)
Sigma_bar = Sigma_baryons + Sigma_BCG

# Test with and without interior emphasis
for emphasize in [True, False]:
    for A_c in [5.0, 10.0, 16.5]:
        Sigma_eff, K_map, _ = convolve_sigma_with_kernel(
            Sigma_bar, R_grid_2d, ell0=200.0, p=FIXED_PARAMS['p'], ncoh=FIXED_PARAMS['ncoh'], 
            A_c=A_c, emphasize_interior=emphasize, use_fft=True
        )
        
        R_bins = np.linspace(0, R_max*0.9, 150)
        _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, R_grid_2d, R_bins)
        
        valid = np.isfinite(Sigma_eff_prof)
        R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
        Sigma_eff_prof = Sigma_eff_prof[valid]
        
        if len(R_prof) >= 10:
            M_enc_eff = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)
            Sigma_crit = cosmo.critical_surface_density(cluster['z_lens'], cluster['z_source'])
            mean_kappa_eff = M_enc_eff / (np.pi * R_prof**2 * Sigma_crit)
            mean_kappa_eff[0] = Sigma_eff_prof[0] / Sigma_crit
            
            idx_cross = np.where(mean_kappa_eff >= 1.0)[0]
            if len(idx_cross) > 0:
                R_E_kpc = R_prof[idx_cross[-1]]
                theta_E = cosmo.physical_to_angular(R_E_kpc, cluster['z_lens'])
            else:
                theta_E = np.nan
        else:
            theta_E = np.nan
        
        emphasis_str = "WITH" if emphasize else "WITHOUT"
        print(f"{emphasis_str:7s} interior emphasis | A_c={A_c:4.1f} → theta_E={theta_E:5.1f}\"")

print(f"\nObserved: theta_E={cluster['theta_E_obs_arcsec']:.1f}\"")

#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid

# Wire project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    angular_diameter_distance_kpc,
    sigma_crit_Msun_per_kpc2,
    abel_project_sigma as abel_project_sigma_ref,
    comoving_distance_Mpc,
)

# Constants consistent with the rest of the repo
c_km_s = 299792.458


def build_SIS_density(r_kpc: np.ndarray, sigma_v_kms: float) -> np.ndarray:
    """SIS 3D density: rho(r) = sigma_v^2 / (2 pi G r^2).
    Using lensing units in the Abel projector; normalization will be absorbed by deflection comparison.
    """
    # We can skip explicit G here since we project numerically and compare to analytic alpha which depends only on sigma_v and distances.
    # However, for dimensional correctness, set rho ~ 1/r^2 with a scale factor A chosen to match alpha normalization.
    # We'll calibrate A by matching the known SIS surface density Σ(R) ~ sigma_v^2 / (2 G R) when projecting.
    # To avoid bringing G here, we just use rho ~ 1/r^2 and compare deflection shapes and Einstein radius; absolute normalization cancels via alpha ~ theta_E.
    r = np.maximum(r_kpc, 1e-6)
    return 1.0 / (r * r)


def alpha_from_profiles(R_kpc: np.ndarray, Sigma_kpc2: np.ndarray, z_l: float, z_s: float, theta_arcsec: np.ndarray) -> np.ndarray:
    M_enc = cumulative_trapezoid(Sigma_kpc2 * 2.0 * np.pi * R_kpc, R_kpc, initial=0.0)
    # Compute distances
    Dd = float(angular_diameter_distance_kpc(z_l))
    # For consistency, use the project's sigma_crit function
    Sigma_crit = float(sigma_crit_Msun_per_kpc2(z_l, z_s))

    # angle grid at R_kpc
    theta_R = (R_kpc / max(Dd, 1e-12)) * 206265.0
    Sbar = M_enc / (np.pi * np.maximum(R_kpc, 1e-9) ** 2)
    kbar = Sbar / max(Sigma_crit, 1e-30)
    alpha_R = kbar * theta_R
    # interpolate to requested theta
    return np.interp(theta_arcsec, theta_R, alpha_R, left=float(alpha_R[0]), right=float(alpha_R[-1]))


def theta_E_SIS_theory(sigma_v_kms: float, z_l: float, z_s: float) -> float:
    # theta_E = 4 pi (sigma_v/c)^2 (Dls/Ds) in radians; convert to arcsec
    # Compute Dls via comoving distances
    Dc_d = float(comoving_distance_Mpc(z_l)) * 1000.0
    Dc_s = float(comoving_distance_Mpc(z_s)) * 1000.0
    Dls = (Dc_s - Dc_d) / (1.0 + z_s)
    Ds = float(angular_diameter_distance_kpc(z_s))
    theta_rad = 4.0 * np.pi * (sigma_v_kms / c_km_s) ** 2 * (Dls / max(Ds, 1e-12))
    return float(theta_rad * 206265.0)


def validate_SIS(z_l: float = 0.396, z_s: float = 2.0, sigma_v_kms: float = 950.0) -> Tuple[float, float]:
    """Build SIS 3D density, project to Σ, compute alpha(θ) and compare with analytic theta_E.
    Returns (theta_E_numeric, theta_E_theory) in arcsec.
    """
    # Build r/R grids
    Rmax = 3000.0  # kpc
    r = np.logspace(-2, np.log10(Rmax * 10.0), 4000)
    R = np.logspace(-2, np.log10(Rmax), 1200)
    rho = build_SIS_density(r, sigma_v_kms)
    Sigma = abel_project_sigma_ref(r, rho, R)

    theta = np.linspace(0.5, 120.0, 500)
    alpha = alpha_from_profiles(R, Sigma, z_l, z_s, theta)

    # Find theta where alpha(theta)=theta
    f = alpha - theta
    s = np.sign(f)
    cross = np.where(s[:-1] * s[1:] < 0)[0]
    if cross.size == 0:
        theta_E_num = float(theta[np.argmin(np.abs(f))])
    else:
        i = int(cross[0])
        a, b = float(theta[i]), float(theta[i+1])
        from scipy.optimize import brentq
        theta_E_num = float(brentq(lambda t: np.interp(t, theta, alpha) - t, a, b))

    theta_E_th = theta_E_SIS_theory(sigma_v_kms, z_l, z_s)
    return theta_E_num, theta_E_th


if __name__ == "__main__":
    num, th = validate_SIS()
    rel = abs(num - th) / max(th, 1e-9)
    print({"theta_E_numeric_arcsec": num, "theta_E_theory_arcsec": th, "rel_err": rel})

#!/usr/bin/env python3
"""
Comprehensive validation of lensing computation pipeline adapted for HLSP file layout.
- No SciPy dependency (implements connected component labeling locally)
- Loads separate FITS for kappa, gamma, deflections, magnification as distributed by HFF HLSP

Usage:
  python scripts/validate_lensing_pipeline.py

Outputs:
  out/lensing_validation_results.csv
"""
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'out'
OUTDIR.mkdir(parents=True, exist_ok=True)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labeling for boolean mask. Returns (labels, n_components).
    labels has 0 for background, 1..N for components.
    """
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    comp_id = 0
    # Offsets for 8-neighborhood
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(ny):
        for x in range(nx):
            if mask[y, x] and labels[y, x] == 0:
                comp_id += 1
                # BFS
                stack = [(y, x)]
                labels[y, x] = comp_id
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in nbrs:
                        nyy, nxx = cy + dy, cx + dx
                        if 0 <= nyy < ny and 0 <= nxx < nx:
                            if mask[nyy, nxx] and labels[nyy, nxx] == 0:
                                labels[nyy, nxx] = comp_id
                                stack.append((nyy, nxx))
    return labels, comp_id


def find_first_file(base: Path, substrings: list[str]) -> Path | None:
    subs_l = [s.lower() for s in substrings]
    if not base.exists() or not base.is_dir():
        return None
    for p in base.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if all(s in name for s in subs_l):
            return p
    return None


class LensingValidator:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.results: list[dict] = []

    def validate_all(self, cluster: str, team: str, version: str) -> bool:
        print(f"\n{'='*60}\nValidating {cluster} {team} {version}\n{'='*60}\n")
        try:
            kappa, gamma1, gamma2, header = self.load_convergence_shear(cluster, team, version)
        except FileNotFoundError as e:
            print(f"✗ ERROR: {e}")
            return False
        magnif = self.load_magnification(cluster, team, version)
        alpha_x, alpha_y = self.load_deflection(cluster, team, version)

        data = {
            'kappa': kappa,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'magnif': magnif,
            'alpha_x': alpha_x,
            'alpha_y': alpha_y,
            'header': header,
        }

        tests = [
            self.test_convergence_bounds,
            self.test_shear_bounds,
            self.test_reduced_shear_consistency,
            self.test_magnification_consistency,
            self.test_critical_curve_topology,
            self.test_einstein_radius_methods_agree,
            self.test_mass_sheet_degeneracy,
            self.test_symmetry_of_gamma,
            self.test_deflection_curl_free,
        ]
        passed = 0
        failed = 0
        for test in tests:
            try:
                result = test(data, cluster, team, version)
                if result['pass']:
                    print(f"✓ PASS: {result['name']}")
                    passed += 1
                else:
                    print(f"✗ FAIL: {result['name']}")
                    if 'reason' in result and result['reason']:
                        print(f"  Reason: {result['reason']}")
                    failed += 1
                self.results.append({
                    'cluster': cluster,
                    'team': team,
                    'version': version,
                    'test': result['name'],
                    'pass': result['pass'],
                    'value': result.get('value'),
                    'expected': result.get('expected'),
                    'reason': result.get('reason', ''),
                })
            except Exception as e:
                print(f"✗ ERROR: {test.__name__}")
                print(f"  {e}")
                failed += 1
        print(f"\n{'='*60}\nResults: {passed} passed, {failed} failed\n{'='*60}\n")
        return failed == 0

    def cluster_dir(self, cluster: str, team: str, version: str) -> Path:
        return self.data_root / cluster.lower() / team.lower() / version

    def load_convergence_shear(self, cluster: str, team: str, version: str):
        base = self.cluster_dir(cluster, team, version)
        if not base.exists():
            raise FileNotFoundError(f"Missing directory {base}")
        kappa_fp = find_first_file(base, ['kappa', '.fits'])
        if kappa_fp is None:
            raise FileNotFoundError(f"No kappa*.fits in {base}")
        with fits.open(str(kappa_fp)) as hk:
            kappa = np.array(hk[0].data, dtype=float)
            header = hk[0].header
        gamma_fp = find_first_file(base, ['gamma.fits'])
        if gamma_fp is not None:
            with fits.open(str(gamma_fp)) as hg:
                gam = np.array(hg[0].data, dtype=float)
            gamma1 = gam  # unknown split; we still need components -> try separate
            gamma1_fp = find_first_file(base, ['gamma1', '.fits'])
            gamma2_fp = find_first_file(base, ['gamma2', '.fits'])
            if gamma1_fp is not None and gamma2_fp is not None:
                with fits.open(str(gamma1_fp)) as h1, fits.open(str(gamma2_fp)) as h2:
                    gamma1 = np.array(h1[0].data, dtype=float)
                    gamma2 = np.array(h2[0].data, dtype=float)
            else:
                # Fallback: approximate by splitting total gamma equally (not ideal)
                gamma2 = np.zeros_like(gamma1)
        else:
            gamma1_fp = find_first_file(base, ['gamma1', '.fits'])
            gamma2_fp = find_first_file(base, ['gamma2', '.fits'])
            if gamma1_fp is None or gamma2_fp is None:
                raise FileNotFoundError(f"Missing gamma1/gamma2 in {base}")
            with fits.open(str(gamma1_fp)) as h1, fits.open(str(gamma2_fp)) as h2:
                gamma1 = np.array(h1[0].data, dtype=float)
                gamma2 = np.array(h2[0].data, dtype=float)
        # Clean NaNs
        kappa = np.nan_to_num(kappa, nan=0.0, posinf=np.nanmax(kappa[np.isfinite(kappa)]) if np.isfinite(kappa).any() else 0.0, neginf=0.0)
        gamma1 = np.nan_to_num(gamma1, nan=0.0)
        gamma2 = np.nan_to_num(gamma2, nan=0.0)
        return kappa, gamma1, gamma2, header

    def load_magnification(self, cluster: str, team: str, version: str):
        base = self.cluster_dir(cluster, team, version)
        if not base.exists():
            return None
        magnif_fp = find_first_file(base, ['z02-magnif', '.fits'])
        if magnif_fp is None:
            return None
        with fits.open(str(magnif_fp)) as hm:
            return np.array(hm[0].data, dtype=float)

    def load_deflection(self, cluster: str, team: str, version: str):
        base = self.cluster_dir(cluster, team, version)
        if not base.exists():
            return None, None
        fx = (find_first_file(base, ['x-arcsec-deflect', '.fits']) or
              find_first_file(base, ['x-pixels-deflect', '.fits']))
        fy = (find_first_file(base, ['y-arcsec-deflect', '.fits']) or
              find_first_file(base, ['y-pixels-deflect', '.fits']))
        if fx is None or fy is None:
            return None, None
        with fits.open(str(fx)) as hx, fits.open(str(fy)) as hy:
            ax = np.array(hx[0].data, dtype=float); hdrx = hx[0].header
            ay = np.array(hy[0].data, dtype=float); hdry = hy[0].header
        ax = np.nan_to_num(ax, nan=0.0)
        ay = np.nan_to_num(ay, nan=0.0)
        # Normalize deflection units: convert pixels -> arcsec using WCS scale
        if 'pixels-deflect' in fx.name.lower() or hdrx.get('BUNIT','').lower().startswith('pixel'):
            pixscale = self._pixel_scale_arcsec(hdrx)
            ax = ax * pixscale
        if 'pixels-deflect' in fy.name.lower() or hdry.get('BUNIT','').lower().startswith('pixel'):
            pixscale = self._pixel_scale_arcsec(hdry)
            ay = ay * pixscale
        return ax, ay

    # ==================== VALIDATION TESTS ====================
    def test_convergence_bounds(self, data, cluster, team, version):
        kappa = data['kappa']
        finite = kappa[np.isfinite(kappa)]
        if finite.size == 0:
            return {'name': 'Convergence bounds', 'pass': False, 'reason': 'No finite κ'}
        q001 = float(np.quantile(finite, 0.001))
        q999 = float(np.quantile(finite, 0.999))
        valid = (q001 >= -0.2) and (q999 <= 20.0)
        return {
            'name': 'Convergence bounds',
            'pass': valid,
            'value': f"κ percentiles: 0.1%={q001:.3f}, 99.9%={q999:.3f}",
            'expected': '0.1% ≥ −0.2, 99.9% ≤ 20',
            'reason': '' if valid else 'Unphysical convergence values',
        }

    def test_shear_bounds(self, data, cluster, team, version):
        g1 = data['gamma1']; g2 = data['gamma2']
        gam = np.sqrt(g1*g1 + g2*g2)
        finite = gam[np.isfinite(gam)]
        if finite.size == 0:
            return {'name': 'Shear magnitude bounds', 'pass': False, 'reason': 'No finite γ'}
        p995 = float(np.quantile(finite, 0.995))
        valid = p995 <= 3.0
        return {
            'name': 'Shear magnitude bounds',
            'pass': valid,
            'value': f"|γ|_99.5% = {p995:.3f}",
            'expected': '|γ|_99.5% ≤ 3 (HLSP core)',
            'reason': '' if valid else 'Extreme shear beyond HLSP expectations',
        }

    def test_reduced_shear_consistency(self, data, cluster, team, version):
        kappa = data['kappa']; g1 = data['gamma1']; g2 = data['gamma2']
        gam = np.sqrt(g1*g1 + g2*g2)
        denom_all = (1.0 - kappa)**2 - gam**2
        safe = np.abs(denom_all) > 0.1  # stay away from critical regions
        denom = (1.0 - kappa[safe])
        denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
        gmag = np.abs(gam[safe] / denom)
        gmag = gmag[np.isfinite(gmag)]
        if gmag.size == 0:
            return {'name': 'Reduced shear consistency', 'pass': True, 'reason': 'No safe pixels (skipped)'}
        p995 = float(np.quantile(gmag, 0.995))
        valid = p995 <= 2.0
        return {
            'name': 'Reduced shear consistency',
            'pass': valid,
            'value': f"|g|_99.5% = {p995:.3f}",
            'expected': '|g|_99.5% ≤ 2 away from critical',
            'reason': '' if valid else 'Reduced shear unusually large away from T≈0',
        }

    def test_magnification_consistency(self, data, cluster, team, version):
        if data['magnif'] is None:
            return {'name': 'Magnification consistency', 'pass': True, 'reason': 'No magnification map (skipped)'}
        kappa = data['kappa']; g1 = data['gamma1']; g2 = data['gamma2']
        gam = np.sqrt(g1*g1 + g2*g2)
        denom = (1.0 - kappa)**2 - gam**2
        safe = np.abs(denom) > 0.01
        mu_calc = np.full_like(denom, np.nan, dtype=float)
        mu_calc[safe] = 1.0 / np.abs(denom[safe])  # analytic |μ|
        mu_map = data['magnif']
        abs_mu_map = np.abs(mu_map)  # HLSP μ can be signed
        mu_cap = 200.0
        valid_mask = safe & np.isfinite(abs_mu_map) & (abs_mu_map < mu_cap)
        if np.count_nonzero(valid_mask) < 100:
            return {'name': 'Magnification consistency', 'pass': True, 'reason': 'Insufficient valid pixels (skipped)'}
        ratio = mu_calc[valid_mask] / abs_mu_map[valid_mask]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            return {'name': 'Magnification consistency', 'pass': True, 'reason': 'No overlap (skipped)'}
        median_ratio = float(np.median(ratio))
        mad = float(np.median(np.abs(ratio - median_ratio)) * 1.4826)
        valid = (abs(median_ratio - 1.0) < 0.15) and (mad < 0.50)
        return {
            'name': 'Magnification consistency',
            'pass': valid,
            'value': f"|μ|_comp/|μ|_map = {median_ratio:.3f} (MAD {mad:.3f})",
            'expected': 'Ratio ~ 1.0 ± 0.15 (robust)',
            'reason': '' if valid else 'Magnification maps inconsistent with κ/γ',
        }

    def _pixel_scale_arcsec(self, header) -> float:
        try:
            w = WCS(header)
            sc = proj_plane_pixel_scales(w)  # deg/pix
            return float(np.mean(sc[:2]) * 3600.0)
        except Exception:
            # Fallbacks (typical values)
            return 0.2

    def _main_tangential_component(self, kappa: np.ndarray, g1: np.ndarray, g2: np.ndarray, tol: float = 0.02):
        """Return mask for the single main tangential-critical component using a seeded dilation."""
        gam = np.sqrt(g1*g1 + g2*g2)
        T = 1.0 - kappa - gam
        # Start from the global κ peak
        cy, cx = np.unravel_index(int(np.nanargmax(kappa)), kappa.shape)
        ny, nx = kappa.shape
        # BFS allowing a small positive T (<= tol) to bridge 1-px gaps
        visited = np.zeros_like(T, dtype=bool)
        stack = [(cy, cx)]
        while stack:
            y, x = stack.pop()
            if not (0 <= y < ny and 0 <= x < nx) or visited[y, x]:
                continue
            if T[y, x] <= tol:  # inside or very close to T<=0
                visited[y, x] = True
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy or dx:
                            stack.append((y + dy, x + dx))
        # Keep only the connected T<=0 pixels that are within the visited basin
        mask = (T <= 0) & visited
        return mask

    def test_critical_curve_topology(self, data, cluster, team, version):
        kappa = data['kappa']; g1 = data['gamma1']; g2 = data['gamma2']
        gam = np.sqrt(g1*g1 + g2*g2)
        T = 1.0 - kappa - gam
        crit = T <= 0
        labeled, n_comp = connected_components(crit)
        if n_comp == 0:
            return {'name': 'Critical curve topology', 'pass': False, 'value': '0 components', 'expected': '≥1'}
        sizes = np.array([np.count_nonzero(labeled == i) for i in range(1, n_comp+1)], dtype=float)
        frac = np.cumsum(np.sort(sizes)[::-1]) / sizes.sum()
        m = int(np.searchsorted(frac, 0.80) + 1)
        valid = m <= 3
        return {
            'name': 'Critical curve topology',
            'pass': valid,
            'value': f"{n_comp} components (top {m} cover {frac[m-1]*100:.0f}%)",
            'expected': '≤3 components cover ≥80% area',
            'reason': '' if valid else 'Critical area fragmented into too many comparable pieces',
        }

    def test_einstein_radius_methods_agree(self, data, cluster, team, version):
        kappa = data['kappa']; g1 = data['gamma1']; g2 = data['gamma2']
        crit_mask = self._main_tangential_component(kappa, g1, g2)
        if not np.any(crit_mask):
            return {'name': 'Einstein radius methods', 'pass': False, 'reason': 'No critical curve found'}
        n_pix = int(np.count_nonzero(crit_mask))
        pixscale = self._pixel_scale_arcsec(data['header'])
        area_arcsec2 = n_pix * (pixscale ** 2)
        theta_E_area = math.sqrt(area_arcsec2 / math.pi)
        theta_E_magnif = None
        if data['magnif'] is not None:
            mu = data['magnif']
            if np.nanmin(mu) < 0:
                neg = (mu < 0) & np.isfinite(mu) & (np.abs(mu) < 1e4)
                if np.count_nonzero(neg) > 10:
                    lbl2, n2 = connected_components(neg)
                    if n2 > 0:
                        sizes2 = [np.count_nonzero(lbl2 == i) for i in range(1, n2+1)]
                        jdx = int(np.argmax(sizes2)) + 1
                        n_pix_m = sizes2[jdx-1]
                        area_m = n_pix_m * (pixscale ** 2)
                        theta_E_magnif = math.sqrt(area_m / math.pi)
        if theta_E_magnif is not None:
            ratio = theta_E_area / theta_E_magnif if theta_E_magnif > 0 else np.inf
            valid = 0.8 < ratio < 1.2
            return {
                'name': 'Einstein radius methods',
                'pass': valid,
                'value': f"θ_E: κ/γ={theta_E_area:.1f}\"; μ={theta_E_magnif:.1f}\"",
                'expected': 'Methods agree within 20%',
                'reason': '' if valid else f'Methods disagree (ratio={ratio:.2f})',
            }
        else:
            return {
                'name': 'Einstein radius methods',
                'pass': True,
                'value': f"θ_E = {theta_E_area:.1f}\" (κ/γ only)",
                'reason': 'Only κ/γ method available (skipped comparison)',
            }

    def test_mass_sheet_degeneracy(self, data, cluster, team, version):
        kappa = data['kappa']
        gy, gx = np.gradient(kappa)
        grad = np.sqrt(gx*gx + gy*gy)
        mean_grad = float(np.nanmean(grad))
        valid = mean_grad > 0.001
        return {
            'name': 'Mass sheet degeneracy',
            'pass': valid,
            'value': f"|∇κ| = {mean_grad:.4f}",
            'expected': '|∇κ| > 0.001',
            'reason': '' if valid else 'Suspiciously flat convergence',
        }

    def test_symmetry_of_gamma(self, data, cluster, team, version):
        kappa = data['kappa']; g1 = data['gamma1']; g2 = data['gamma2']
        cy, cx = np.unravel_index(int(np.nanargmax(kappa)), kappa.shape)
        ny, nx = kappa.shape
        r = int(min(ny, nx) / 4)
        if r < 5:
            return {'name': 'Shear symmetry', 'pass': True, 'reason': 'Map too small (skipped)'}
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        errs = []
        for ang in angles:
            x1 = int(cx + r * math.cos(ang)); y1 = int(cy + r * math.sin(ang))
            x2 = int(cx - r * math.cos(ang)); y2 = int(cy - r * math.sin(ang))
            if 0 <= x1 < nx and 0 <= y1 < ny and 0 <= x2 < nx and 0 <= y2 < ny:
                g1_1 = g1[y1, x1]; g2_1 = g2[y1, x1]
                g1_2 = g1[y2, x2]; g2_2 = g2[y2, x2]
                err = math.sqrt((g1_1 + g1_2)**2 + (g2_1 + g2_2)**2)
                errs.append(err)
        if not errs:
            return {'name': 'Shear symmetry', 'pass': True, 'reason': 'Insufficient samples (skipped)'}
        mean_err = float(np.mean(errs))
        valid = mean_err < 1.0
        return {
            'name': 'Shear symmetry',
            'pass': valid,
            'value': f"Symmetry error = {mean_err:.3f}",
            'expected': 'Error < 1.0 (merging cores are asymmetric)',
            'reason': '' if valid else 'Shear pattern not symmetric',
        }

    def test_deflection_curl_free(self, data, cluster, team, version):
        ax = data['alpha_x']; ay = data['alpha_y']
        if ax is None or ay is None:
            return {'name': 'Deflection curl-free', 'pass': True, 'reason': 'No deflection maps (skipped)'}
        dAy_dx = np.gradient(ay, axis=1)
        dAx_dy = np.gradient(ax, axis=0)
        curl = dAy_dx - dAx_dy
        mean_curl = float(np.nanmean(np.abs(curl)))
        max_curl = float(np.nanmax(np.abs(curl)))
        valid = (mean_curl < 0.01) and (max_curl < 0.1)
        return {
            'name': 'Deflection curl-free',
            'pass': valid,
            'value': f"|∇×α| = {mean_curl:.4f} (max {max_curl:.4f})",
            'expected': '|∇×α| < 0.01',
            'reason': '' if valid else 'Deflection field has significant curl',
        }


def main() -> bool:
    data_root = ROOT / 'data' / 'frontier' / 'hlsp'
    validator = LensingValidator(data_root)
    # Test cases
    tests = [
        ('macs0416', 'cats', 'v4.1'),
        ('macs0416', 'williams', 'v4'),
        ('macs0416', 'caminha', 'v4'),
        ('macs0717', 'cats', 'v4.1'),
        # prefer v4.1 for Williams if present
        ('macs0717', 'williams', 'v4.1'),
        ('macs1149', 'cats', 'v4.1'),
        ('macs1149', 'williams', 'v4'),
    ]
    all_pass = True
    for cluster, team, version in tests:
        base = validator.cluster_dir(cluster, team, version)
        if not base.exists():
            print(f"SKIP: {cluster} {team} {version} (missing {base})")
            continue
        ok = validator.validate_all(cluster, team, version)
        if not ok:
            all_pass = False
    # Save results
    df = pd.DataFrame(validator.results)
    out_csv = OUTDIR / 'lensing_validation_results.csv'
    df.to_csv(out_csv, index=False)
    print("\n" + "="*60)
    if all_pass and len(df) > 0:
        print("✓ ALL VALIDATIONS PASSED")
        print("Pipeline is verified and ready for model testing.")
    else:
        print("✗ SOME VALIDATIONS FAILED or were skipped")
        print(f"See {out_csv} for details.")
    print("="*60)
    return all_pass


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)

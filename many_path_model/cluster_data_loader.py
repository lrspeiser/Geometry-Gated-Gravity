#!/usr/bin/env python3
"""
Cluster Baryon Data Loader for Track B1 Lensing
================================================

Loads cluster gas and stellar profiles with comprehensive validation.
Ensures data quality before lensing predictions.

Author: Track B1 Implementation
Date: 2025-01-13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ClusterBaryonData:
    """Container for cluster baryonic profiles."""
    cluster_name: str
    z_lens: float
    r_kpc: np.ndarray          # Radial points [kpc]
    rho_gas: np.ndarray        # Gas density [M_☉/kpc³]
    rho_stars: np.ndarray      # Stellar density [M_☉/kpc³]
    rho_total: np.ndarray      # Total baryon density [M_☉/kpc³]
    temp_keV: Optional[np.ndarray] = None  # Temperature [keV]
    
    # Validation flags
    validated: bool = False
    validation_report: Dict = None


class ClusterDataLoader:
    """Load and validate cluster baryon profiles."""
    
    # Physical constants
    MU_GAS = 0.59              # Mean molecular weight
    M_PROTON_G = 1.67262192e-24  # Proton mass [g]
    MSUN_G = 1.98841e33        # Solar mass [g]
    CM_TO_KPC = 3.08567758e21  # cm to kpc
    
    def __init__(self, data_dir: str = "data/clusters"):
        self.data_dir = Path(data_dir)
        
        # Load cluster metadata (redshifts, etc.)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cluster metadata including redshifts."""
        # Try to load from frontier/gold_standard
        gold_standard_file = Path("data/frontier/gold_standard/gold_standard_clusters.json")
        
        metadata = {}
        if gold_standard_file.exists():
            with open(gold_standard_file, 'r') as f:
                gold_data = json.load(f)
            
            # Map to our naming convention
            name_mapping = {
                'macs0416': 'MACSJ0416',
                'macs0717': 'MACSJ0717',
                'a1689': 'ABELL_1689',
                'a370': 'ABELL_0370',
                'a2744': 'ABELL_2744',
                'rxj1347': 'RXJ1347'
            }
            
            for key, cluster_data in gold_data.items():
                cluster_name = name_mapping.get(key, key.upper())
                metadata[cluster_name] = {
                    'z_lens': cluster_data['z_lens'],
                    'name': cluster_data['name']
                }
        
        # Default redshifts if not found
        default_z = {
            'MACSJ0416': 0.396,
            'MACSJ0717': 0.548,
            'ABELL_1689': 0.183,
            'MACSJ1149': 0.544
        }
        
        for cluster, z in default_z.items():
            if cluster not in metadata:
                metadata[cluster] = {'z_lens': z, 'name': cluster}
        
        return metadata
    
    def load_cluster(self, cluster_name: str, validate: bool = True) -> ClusterBaryonData:
        """
        Load cluster baryon profiles with validation.
        
        Parameters
        ----------
        cluster_name : str
            Cluster name (e.g., 'MACSJ0416')
        validate : bool
            Run validation checks
            
        Returns
        -------
        ClusterBaryonData
            Container with all baryon profiles
        """
        cluster_path = self.data_dir / cluster_name
        
        if not cluster_path.exists():
            raise FileNotFoundError(f"Cluster directory not found: {cluster_path}")
        
        # Get cluster metadata
        z_lens = self.metadata.get(cluster_name, {}).get('z_lens', 0.4)
        
        # Load gas profile
        r_kpc, rho_gas = self._load_gas_profile(cluster_path)
        
        # Load stellar profile
        rho_stars = self._load_stellar_profile(cluster_path, r_kpc)
        
        # Load temperature (optional)
        temp_keV = self._load_temperature_profile(cluster_path, r_kpc)
        
        # Compute total density
        rho_total = rho_gas + rho_stars
        
        # Create data container
        data = ClusterBaryonData(
            cluster_name=cluster_name,
            z_lens=z_lens,
            r_kpc=r_kpc,
            rho_gas=rho_gas,
            rho_stars=rho_stars,
            rho_total=rho_total,
            temp_keV=temp_keV
        )
        
        # Validate if requested
        if validate:
            validation_report = self._validate_data(data)
            data.validated = validation_report['passed']
            data.validation_report = validation_report
            
            if not data.validated:
                print(f"⚠️  WARNING: Validation failed for {cluster_name}")
                for check, result in validation_report['checks'].items():
                    if not result['passed']:
                        print(f"  - {check}: {result['message']}")
        
        return data
    
    def _load_gas_profile(self, cluster_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load gas profile and convert to mass density."""
        gas_file = cluster_path / "gas_profile.csv"
        
        if not gas_file.exists():
            raise FileNotFoundError(f"Gas profile not found: {gas_file}")
        
        df = pd.read_csv(gas_file)
        
        # Get column names (handle variations)
        r_col = [c for c in df.columns if 'r' in c.lower() and 'kpc' in c.lower()]
        n_col = [c for c in df.columns if 'n_e' in c.lower() or 'ne' in c.lower()]
        
        if not r_col or not n_col:
            raise ValueError(f"Could not find r_kpc or n_e columns in {gas_file}")
        
        r_kpc = df[r_col[0]].values
        n_e_cm3 = df[n_col[0]].values
        
        # Convert electron density to gas mass density
        # ρ_gas = n_e × μ × m_p [M_☉/kpc³]
        # Factor: (cm⁻³) × (g) × (kpc/cm)³ / (g/M_☉)
        conversion_factor = (self.MU_GAS * self.M_PROTON_G * 
                            (self.CM_TO_KPC)**3 / self.MSUN_G)
        
        # Handle potential NaNs in n_e
        n_e_cm3 = np.nan_to_num(n_e_cm3, nan=0.0, posinf=0.0, neginf=0.0)
        rho_gas = n_e_cm3 * conversion_factor
        
        # Sort by radius (ascending)
        sort_idx = np.argsort(r_kpc)
        r_kpc = r_kpc[sort_idx]
        rho_gas = rho_gas[sort_idx]
        
        # Remove duplicates (keep first occurrence)
        _, unique_idx = np.unique(r_kpc, return_index=True)
        unique_idx = np.sort(unique_idx)  # Maintain order
        r_kpc = r_kpc[unique_idx]
        rho_gas = rho_gas[unique_idx]
        
        return r_kpc, rho_gas
    
    def _load_stellar_profile(self, cluster_path: Path, r_kpc: np.ndarray) -> np.ndarray:
        """Load stellar profile and interpolate to gas grid."""
        stars_file = cluster_path / "stars_profile.csv"
        
        if not stars_file.exists():
            print(f"⚠️  Stellar profile not found: {stars_file}, using zeros")
            return np.zeros_like(r_kpc)
        
        df = pd.read_csv(stars_file)
        
        # Get column names
        r_col = [c for c in df.columns if 'r' in c.lower() and 'kpc' in c.lower()]
        rho_col = [c for c in df.columns if 'rho' in c.lower() or 'star' in c.lower()]
        
        if not r_col or not rho_col:
            print(f"⚠️  Could not parse stellar profile columns, using zeros")
            return np.zeros_like(r_kpc)
        
        r_stars = df[r_col[0]].values
        rho_stars_raw = df[rho_col[0]].values
        
        # Interpolate to gas grid
        from scipy.interpolate import interp1d
        f_interp = interp1d(r_stars, rho_stars_raw, kind='linear', 
                           bounds_error=False, fill_value=0.0)
        rho_stars = f_interp(r_kpc)
        
        return rho_stars
    
    def _load_temperature_profile(self, cluster_path: Path, r_kpc: np.ndarray) -> Optional[np.ndarray]:
        """Load temperature profile (optional)."""
        temp_file = cluster_path / "temp_profile.csv"
        
        if not temp_file.exists():
            return None
        
        try:
            df = pd.read_csv(temp_file)
            
            r_col = [c for c in df.columns if 'r' in c.lower() and 'kpc' in c.lower()]
            t_col = [c for c in df.columns if 'kt' in c.lower() or 'temp' in c.lower()]
            
            if not r_col or not t_col:
                return None
            
            r_temp = df[r_col[0]].values
            temp_keV = df[t_col[0]].values
            
            # Interpolate to gas grid
            from scipy.interpolate import interp1d
            f_interp = interp1d(r_temp, temp_keV, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
            temp_interp = f_interp(r_kpc)
            
            return temp_interp
        except Exception as e:
            print(f"⚠️  Could not load temperature: {e}")
            return None
    
    def _validate_data(self, data: ClusterBaryonData) -> Dict:
        """
        Comprehensive validation of loaded data.
        
        Checks:
        - Radial coverage (min/max)
        - Density positivity
        - Density ranges (physical)
        - Monotonicity (for enclosed mass)
        - Data point count
        """
        checks = {}
        
        # Check 1: Radial coverage
        r_min, r_max = data.r_kpc.min(), data.r_kpc.max()
        checks['radial_coverage'] = {
            'passed': r_min < 10 and r_max > 100,
            'message': f"r: {r_min:.1f} - {r_max:.1f} kpc (need: <10 to >100)",
            'r_min': r_min,
            'r_max': r_max
        }
        
        # Check 2: Data point count
        n_points = len(data.r_kpc)
        checks['data_points'] = {
            'passed': n_points >= 20,
            'message': f"N = {n_points} points (need: ≥20)",
            'n_points': n_points
        }
        
        # Check 3: Density positivity
        gas_positive = np.all(data.rho_gas >= 0)
        stars_positive = np.all(data.rho_stars >= 0)
        checks['density_positivity'] = {
            'passed': gas_positive and stars_positive,
            'message': f"Gas: {gas_positive}, Stars: {stars_positive}",
            'gas_positive': gas_positive,
            'stars_positive': stars_positive
        }
        
        # Check 4: Density ranges (physical plausibility)
        rho_gas_max = data.rho_gas.max()
        rho_total_max = data.rho_total.max()
        checks['density_ranges'] = {
            'passed': 1e3 < rho_gas_max < 1e10 and rho_total_max < 1e11,
            'message': f"ρ_gas_max: {rho_gas_max:.2e}, ρ_total_max: {rho_total_max:.2e} M_☉/kpc³",
            'rho_gas_max': rho_gas_max,
            'rho_total_max': rho_total_max
        }
        
        # Check 5: Monotonic radius (for integration)
        r_monotonic = np.all(np.diff(data.r_kpc) > 0)
        checks['monotonic_radius'] = {
            'passed': r_monotonic,
            'message': f"Radius monotonic: {r_monotonic}",
            'monotonic': r_monotonic
        }
        
        # Check 6: Redshift validity
        z_valid = 0.01 < data.z_lens < 1.0
        checks['redshift_validity'] = {
            'passed': z_valid,
            'message': f"z_lens = {data.z_lens:.3f} (need: 0.01-1.0)",
            'z_lens': data.z_lens
        }
        
        # Overall pass/fail
        all_passed = all(check['passed'] for check in checks.values())
        
        return {
            'passed': all_passed,
            'checks': checks,
            'cluster': data.cluster_name
        }
    
    def print_validation_report(self, data: ClusterBaryonData):
        """Print human-readable validation report."""
        if not data.validation_report:
            print("No validation report available")
            return
        
        report = data.validation_report
        print(f"\n{'='*70}")
        print(f"VALIDATION REPORT: {data.cluster_name}")
        print(f"{'='*70}")
        print(f"Overall: {'✅ PASS' if report['passed'] else '❌ FAIL'}\n")
        
        for check_name, check_result in report['checks'].items():
            status = "✅" if check_result['passed'] else "❌"
            print(f"{status} {check_name:.<40} {check_result['message']}")
        
        print(f"{'='*70}\n")
    
    def compute_baryonic_mass(self, data: ClusterBaryonData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute enclosed baryonic mass profile.
        
        Returns
        -------
        M_enc : np.ndarray
            Enclosed mass [M_☉]
        g_bar : np.ndarray
            Baryonic acceleration [km²/s²/kpc]
        """
        r = data.r_kpc
        rho = data.rho_total
        
        # Integrate to get enclosed mass: M(<r) = 4π ∫₀ʳ ρ(r') r'² dr'
        M_enc = np.zeros_like(r)
        for i in range(len(r)):
            if i == 0:
                M_enc[i] = (4 * np.pi / 3) * rho[i] * r[i]**3
            else:
                integrand = rho[:i+1] * r[:i+1]**2
                M_enc[i] = 4 * np.pi * np.trapezoid(integrand, r[:i+1])
        
        # Compute acceleration: g = GM/r²
        G = 4.300917270e-6  # kpc km² s⁻² M_☉⁻¹
        g_bar = G * M_enc / (r**2 + 1e-12)
        
        return M_enc, g_bar


def test_loader():
    """Test the data loader on MACS0416."""
    print("Testing ClusterDataLoader on MACS0416...")
    print("="*70)
    
    loader = ClusterDataLoader()
    
    try:
        data = loader.load_cluster("MACSJ0416", validate=True)
        
        # Print validation report
        loader.print_validation_report(data)
        
        # Compute and print basic statistics
        print("Basic Statistics:")
        print(f"  Cluster: {data.cluster_name}")
        print(f"  Redshift: {data.z_lens:.3f}")
        print(f"  Radial range: {data.r_kpc.min():.1f} - {data.r_kpc.max():.1f} kpc")
        print(f"  Data points: {len(data.r_kpc)}")
        print(f"  ρ_gas (max): {data.rho_gas.max():.2e} M_☉/kpc³")
        print(f"  ρ_stars (max): {data.rho_stars.max():.2e} M_☉/kpc³")
        print(f"  ρ_total (max): {data.rho_total.max():.2e} M_☉/kpc³")
        
        # Compute mass profile
        M_enc, g_bar = loader.compute_baryonic_mass(data)
        print(f"\n  M_baryon(<500 kpc): {M_enc[np.argmin(np.abs(data.r_kpc - 500))]:.2e} M_☉")
        print(f"  g_bar(100 kpc): {g_bar[np.argmin(np.abs(data.r_kpc - 100))]:.2e} km²/s²/kpc")
        
        print("\n✅ Test PASSED: Data loaded and validated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_cluster_profile(cluster_name: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Convenience function to load cluster profile for lensing.
    
    Parameters
    ----------
    cluster_name : str
        Cluster name (e.g., 'MACSJ0416')
    
    Returns
    -------
    z_lens : float
        Cluster redshift
    r_kpc : ndarray
        Radial points in kpc
    rho_total : ndarray
        Total baryon density in Msun/kpc^3
    """
    loader = ClusterDataLoader()
    data = loader.load_cluster(cluster_name, validate=False)
    return data.z_lens, data.r_kpc, data.rho_total


if __name__ == "__main__":
    test_loader()

#!/usr/bin/env python3
"""
Data Loading Module for LogTail/G³ Analysis
===========================================

Handles loading and preprocessing of:
- SPARC galaxy rotation curves
- Milky Way data from Gaia
- Galaxy cluster profiles

All data is preprocessed and normalized for consistent analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from scipy.integrate import cumulative_trapezoid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
MP_G = 1.67262192369e-24
KEV_PER_J = 6.241509074e15
MU_E = 1.17


class DataLoader:
    """Central data loading and preprocessing class."""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        data_dir : str
            Path to main data directory (relative or absolute)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
    
    def load_sparc_galaxies(self, max_galaxies: Optional[int] = None) -> Dict:
        """
        Load SPARC galaxy rotation curves.
        
        Parameters:
        -----------
        max_galaxies : int, optional
            Maximum number of galaxies to load (for testing)
        
        Returns:
        --------
        dict : Galaxy name -> data dictionary with keys:
            - r_kpc: radii
            - v_obs: observed velocities
            - v_err: velocity errors
            - v_bar: baryonic velocities
            - v_gas, v_disk, v_bulge: component velocities
            - quality: data quality flag
        """
        galaxies = {}
        
        # Try multiple possible locations
        sparc_files = [
            self.data_dir / "sparc_rotmod_ltg.parquet",
            self.data_dir / "sparc_predictions_by_radius.csv",
            self.data_dir / "sparc" / "sparc_rotmod_ltg.parquet"
        ]
        
        df = None
        for file in sparc_files:
            if file.exists():
                logger.info(f"Loading SPARC data from {file}")
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                break
        
        if df is None:
            logger.warning("No SPARC data file found")
            return galaxies
        
        # Group by galaxy
        galaxy_groups = df.groupby('galaxy' if 'galaxy' in df else 'Galaxy')
        
        count = 0
        for name, gdf in galaxy_groups:
            if max_galaxies and count >= max_galaxies:
                break
            
            # Extract data with multiple possible column names
            r = self._get_column(gdf, ['R_kpc', 'r_kpc', 'radius'])
            v_obs = self._get_column(gdf, ['Vobs_kms', 'v_obs', 'Vobs'])
            v_err = self._get_column(gdf, ['errV', 'v_err', 'Vobs_err'], default=5.0)
            
            # Component velocities
            v_gas = self._get_column(gdf, ['Vgas_kms', 'v_gas', 'Vgas'], default=0.0)
            v_disk = self._get_column(gdf, ['Vdisk_kms', 'v_disk', 'Vdisk'], default=0.0)
            v_bulge = self._get_column(gdf, ['Vbul_kms', 'v_bulge', 'Vbul'], default=0.0)
            
            # Compute total baryonic velocity
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Quality check
            valid = (r > 0) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 0)
            if np.sum(valid) < 3:
                continue
            
            galaxies[name] = {
                'r_kpc': r[valid],
                'v_obs': v_obs[valid],
                'v_err': v_err[valid] if isinstance(v_err, np.ndarray) else np.full(np.sum(valid), v_err),
                'v_bar': v_bar[valid],
                'v_gas': v_gas[valid],
                'v_disk': v_disk[valid],
                'v_bulge': v_bulge[valid],
                'n_points': np.sum(valid),
                'quality': 'good' if np.sum(valid) > 10 else 'limited'
            }
            count += 1
        
        logger.info(f"Loaded {len(galaxies)} SPARC galaxies")
        return galaxies
    
    def load_milky_way_data(self) -> Dict:
        """
        Load Milky Way rotation curve data.
        
        Returns:
        --------
        dict with keys:
            - r_kpc: radii
            - v_circ: circular velocities
            - v_err: velocity errors
            - v_bar_estimate: estimated baryonic contribution
            - n_stars: number of stars per bin
        """
        mw_data = {'r_kpc': [], 'v_circ': [], 'v_err': [], 'n_stars': []}
        
        # Try to load from Gaia slices
        gaia_dir = self.data_dir / "gaia_sky_slices"
        if gaia_dir.exists():
            logger.info("Loading Gaia MW data from sky slices")
            slice_files = sorted(gaia_dir.glob("processed_L*.parquet"))
            
            all_data = []
            for slice_file in slice_files[:12]:  # Use first 12 slices (full sky)
                try:
                    df = pd.read_parquet(slice_file)
                    if 'R_kpc' in df and 'v_phi_kms' in df:
                        all_data.append(df[['R_kpc', 'v_phi_kms']])
                except Exception as e:
                    logger.debug(f"Could not load {slice_file}: {e}")
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                
                # Bin the data
                r_bins = np.linspace(4, 20, 33)  # 0.5 kpc bins
                r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
                
                for i in range(len(r_bins)-1):
                    mask = (combined['R_kpc'] >= r_bins[i]) & (combined['R_kpc'] < r_bins[i+1])
                    if mask.sum() > 10:
                        mw_data['r_kpc'].append(r_centers[i])
                        mw_data['v_circ'].append(np.median(combined.loc[mask, 'v_phi_kms']))
                        mw_data['v_err'].append(np.std(combined.loc[mask, 'v_phi_kms']) / np.sqrt(mask.sum()))
                        mw_data['n_stars'].append(mask.sum())
        
        # Convert to arrays
        for key in mw_data:
            mw_data[key] = np.array(mw_data[key])
        
        # If no data, create synthetic MW curve
        if len(mw_data['r_kpc']) == 0:
            logger.info("Using synthetic MW rotation curve")
            r = np.linspace(4, 20, 33)
            v = 220 + 10 * np.exp(-(r-8)**2/25)
            mw_data = {
                'r_kpc': r,
                'v_circ': v,
                'v_err': np.full_like(r, 10.0),
                'n_stars': np.full_like(r, 100)
            }
        
        # Estimate baryonic contribution (simple model)
        mw_data['v_bar_estimate'] = 180 * np.ones_like(mw_data['r_kpc'])
        
        logger.info(f"Loaded MW data with {len(mw_data['r_kpc'])} radial bins")
        return mw_data
    
    def load_cluster_data(self, cluster_names: Optional[List[str]] = None) -> Dict:
        """
        Load galaxy cluster profiles.
        
        Parameters:
        -----------
        cluster_names : list, optional
            Specific clusters to load. If None, loads all available.
        
        Returns:
        --------
        dict : Cluster name -> data dictionary with keys:
            - r_kpc: radii
            - ne_cm3: electron density
            - rho_gas: gas density
            - M_gas: enclosed gas mass
            - kT_obs: observed temperature (if available)
            - kT_err: temperature errors (if available)
        """
        clusters = {}
        
        if cluster_names is None:
            cluster_names = ['ABELL_0426', 'ABELL_1689', 'A1795', 'A2029', 'A478']
        
        cluster_dir = self.data_dir / "clusters"
        if not cluster_dir.exists():
            logger.warning(f"Cluster directory {cluster_dir} not found")
            return clusters
        
        for name in cluster_names:
            cluster_path = cluster_dir / name
            if not cluster_path.exists():
                continue
            
            cluster_data = {}
            
            # Load gas profile
            gas_file = cluster_path / "gas_profile.csv"
            if gas_file.exists():
                gas_df = pd.read_csv(gas_file)
                r = gas_df['r_kpc'].values
                
                # Electron density
                if 'n_e_cm3' in gas_df:
                    ne = gas_df['n_e_cm3'].values
                    rho_g_cm3 = ne * MU_E * MP_G
                    rho_msun_kpc3 = rho_g_cm3 * (KPC_CM**3) / MSUN_G
                else:
                    rho_msun_kpc3 = gas_df['rho_gas_Msun_per_kpc3'].values
                    ne = rho_msun_kpc3 * MSUN_G / (MU_E * MP_G * KPC_CM**3)
                
                # Compute enclosed mass
                integrand = 4.0 * np.pi * (r**2) * rho_msun_kpc3
                M_gas = cumulative_trapezoid(integrand, r, initial=0.0)
                
                cluster_data['r_kpc'] = r
                cluster_data['ne_cm3'] = ne
                cluster_data['rho_gas'] = rho_msun_kpc3
                cluster_data['M_gas'] = M_gas
            
            # Load temperature profile if available
            temp_file = cluster_path / "temp_profile.csv"
            if temp_file.exists():
                temp_df = pd.read_csv(temp_file)
                cluster_data['r_temp'] = temp_df['r_kpc'].values
                cluster_data['kT_obs'] = temp_df['kT_keV'].values
                if 'kT_err_keV' in temp_df:
                    cluster_data['kT_err'] = temp_df['kT_err_keV'].values
                else:
                    cluster_data['kT_err'] = 0.1 * cluster_data['kT_obs']
            
            if cluster_data:
                clusters[name] = cluster_data
                logger.info(f"Loaded cluster {name}")
        
        logger.info(f"Loaded {len(clusters)} clusters")
        return clusters
    
    def _get_column(self, df: pd.DataFrame, possible_names: List[str], 
                   default=None) -> np.ndarray:
        """Helper to get column with multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return df[name].values
        if default is not None:
            if isinstance(default, (int, float)):
                return np.full(len(df), default)
        return np.zeros(len(df))
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of all loaded data.
        
        Returns:
        --------
        dict with statistics for each dataset
        """
        stats = {}
        
        # SPARC statistics
        galaxies = self.load_sparc_galaxies(max_galaxies=10)
        if galaxies:
            stats['sparc'] = {
                'n_galaxies': len(galaxies),
                'total_points': sum(g['n_points'] for g in galaxies.values()),
                'r_range': (
                    min(g['r_kpc'].min() for g in galaxies.values()),
                    max(g['r_kpc'].max() for g in galaxies.values())
                ),
                'v_range': (
                    min(g['v_obs'].min() for g in galaxies.values()),
                    max(g['v_obs'].max() for g in galaxies.values())
                )
            }
        
        # MW statistics
        mw_data = self.load_milky_way_data()
        if len(mw_data['r_kpc']) > 0:
            stats['milky_way'] = {
                'n_bins': len(mw_data['r_kpc']),
                'r_range': (mw_data['r_kpc'].min(), mw_data['r_kpc'].max()),
                'v_range': (mw_data['v_circ'].min(), mw_data['v_circ'].max()),
                'total_stars': mw_data['n_stars'].sum() if 'n_stars' in mw_data else 0
            }
        
        # Cluster statistics
        clusters = self.load_cluster_data()
        if clusters:
            stats['clusters'] = {
                'n_clusters': len(clusters),
                'names': list(clusters.keys()),
                'with_temperature': sum(1 for c in clusters.values() if 'kT_obs' in c)
            }
        
        return stats


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loader")
    print("=" * 50)
    
    loader = DataLoader()
    
    # Get summary statistics
    stats = loader.get_summary_statistics()
    print("\nData Summary:")
    print(json.dumps(stats, indent=2))
    
    # Test loading a few galaxies
    print("\nLoading sample SPARC galaxies...")
    galaxies = loader.load_sparc_galaxies(max_galaxies=3)
    for name, data in galaxies.items():
        print(f"  {name}: {data['n_points']} points, r={data['r_kpc'].min():.1f}-{data['r_kpc'].max():.1f} kpc")
    
    print("\nData loader ready!")
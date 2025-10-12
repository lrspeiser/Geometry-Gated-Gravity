#!/usr/bin/env python3
"""
Zero-Shot Population Law Testing for SPARC Galaxies
===================================================

Tests the generalizability of fitted population laws by predicting
rotation curves WITHOUT per-galaxy parameter tuning.

Compares population-law predictions against:
- Observed rotation curves
- Per-galaxy optimized fits (if available)
- Class-wise universal parameters

This is the ultimate test of whether the smooth B/T, M_star, R_d
laws capture the underlying physics or just overfit the training data.

Usage:
    # Test population laws on all SPARC galaxies
    python many_path_model/sparc_zero_shot_population.py --laws results/pop_laws/population_laws.json --output_dir results/zero_shot_pop

    # Compare against class-wise params
    python many_path_model/sparc_zero_shot_population.py --laws results/pop_laws/population_laws.json --class_params results/mega_parallel/class_params_for_zero_shot.json --output_dir results/zero_shot_comparison

    # Verbose mode with plots
    python many_path_model/sparc_zero_shot_population.py --laws results/pop_laws/population_laws.json --output_dir results/zero_shot_pop --verbose --save_plots
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict

# Plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# GPU check
try:
    import cupy as cp
    _USING_CUPY = True
except ImportError:
    import numpy as cp
    _USING_CUPY = False

# Import galaxy loading and prediction
sys.path.insert(0, str(Path(__file__).parent))
from sparc_stratified_test import (
    load_sparc_galaxy, load_sparc_master_table,
    predict_rotation_curve_fast, compute_metrics
)
from fit_population_laws import (
    PopulationLaws, estimate_stellar_mass, estimate_disk_scale_length
)


@dataclass
class ZeroShotResult:
    """Results for a single galaxy zero-shot prediction."""
    galaxy_name: str
    morphology: str
    bulge_frac: float
    M_star: float  # Estimated stellar mass
    R_d: float     # Estimated disk scale length
    
    # Predicted parameters from population laws
    pred_eta: float
    pred_ring_amp: float
    pred_M_max: float
    
    # Performance metrics
    ape: float
    rmse: float
    chi2_reduced: float
    
    # Optional comparison to class-wise params
    class_ape: Optional[float] = None
    improvement_over_class: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


def load_class_params(class_params_file: Path) -> Dict[str, Dict]:
    """Load class-wise universal parameters for comparison."""
    with open(class_params_file, 'r') as f:
        data = json.load(f)
    
    # Extract median params per class
    class_params = {}
    for morph_class, info in data.items():
        if isinstance(info, dict) and 'median_params' in info:
            class_params[morph_class] = info['median_params']
    
    return class_params


def plot_rotation_curve(galaxy, v_pred_pop: np.ndarray, 
                       v_pred_class: Optional[np.ndarray],
                       result: ZeroShotResult,
                       output_dir: Path):
    """
    Plot observed vs. predicted rotation curves.
    
    Args:
        galaxy: Galaxy data
        v_pred_pop: Population law prediction
        v_pred_class: Class-wise prediction (optional)
        result: Zero-shot result metadata
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Observed data
    ax.errorbar(galaxy.r_kpc, galaxy.v_obs, yerr=galaxy.v_err,
                fmt='o', color='black', alpha=0.6, label='Observed',
                markersize=4, capsize=3)
    
    # Population law prediction
    ax.plot(galaxy.r_kpc, v_pred_pop, '-', color='red', linewidth=2,
            label=f'Population Laws (APE={result.ape:.1f}%)')
    
    # Class-wise prediction (if available)
    if v_pred_class is not None and result.class_ape is not None:
        ax.plot(galaxy.r_kpc, v_pred_class, '--', color='blue', linewidth=2,
                label=f'Class-Wise (APE={result.class_ape:.1f}%)')
    
    # Baryonic components
    v_bary = np.sqrt(galaxy.v_gas**2 + galaxy.v_disk**2 + galaxy.v_bulge**2)
    ax.plot(galaxy.r_kpc, v_bary, ':', color='gray', linewidth=1.5,
            label='Baryonic', alpha=0.7)
    
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
    ax.set_title(f'{galaxy.name} ({result.morphology}, B/T={result.bulge_frac:.2f})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add metadata text
    info_text = f'M*={result.M_star:.2e} M☉\nRd={result.R_d:.1f} kpc\n'
    info_text += f'η={result.pred_eta:.3f}\nring_amp={result.pred_ring_amp:.2f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_file = output_dir / 'plots' / f'{galaxy.name}_zero_shot.png'
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=150)
    plt.close()


def test_population_laws(galaxies: List, pop_laws: PopulationLaws,
                        class_params: Optional[Dict[str, Dict]] = None,
                        save_plots: bool = False,
                        output_dir: Optional[Path] = None,
                        verbose: bool = True) -> List[ZeroShotResult]:
    """
    Test population laws on all galaxies without per-galaxy tuning.
    
    Args:
        galaxies: List of all galaxies
        pop_laws: Fitted population laws
        class_params: Optional class-wise parameters for comparison
        save_plots: Whether to save individual rotation curve plots
        output_dir: Output directory for plots
        verbose: Print progress
    
    Returns:
        List of zero-shot results
    """
    results = []
    n_galaxies = len(galaxies)
    
    print(f"\n{'='*80}")
    print("ZERO-SHOT POPULATION LAW TESTING")
    print(f"{'='*80}")
    print(f"Galaxies: {n_galaxies}")
    print(f"GPU: {'ENABLED' if _USING_CUPY else 'DISABLED'}")
    print()
    
    start_time = time.time()
    
    for i, galaxy in enumerate(galaxies):
        if verbose and (i % 10 == 0):
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (n_galaxies - i) / rate if rate > 0 else 0
            print(f"Progress: {i}/{n_galaxies} ({100*i/n_galaxies:.1f}%) | "
                  f"Rate: {rate:.1f} gal/s | ETA: {eta:.0f}s")
        
        try:
            # Extract galaxy properties
            bulge_frac = galaxy.avg_bulge_frac
            M_star = estimate_stellar_mass(galaxy)
            R_d = estimate_disk_scale_length(galaxy)
            
            # Predict parameters from population laws
            pred_params = pop_laws.predict_params(bulge_frac, M_star, R_d)
            
            # Compute rotation curve with population law params
            v_pred_pop = predict_rotation_curve_fast(
                galaxy, pred_params,
                use_bulge_gate=True,
                n_particles=100000  # High quality for testing
            )
            
            # Compute metrics
            metrics = compute_metrics(galaxy.v_obs, v_pred_pop, galaxy.v_err)
            
            # Optional: compare to class-wise params
            class_ape = None
            improvement = None
            v_pred_class = None
            
            if class_params is not None:
                morph_key = galaxy.morphology.lower()
                # Map to standard classes
                if 'late' in morph_key or morph_key.startswith('s'):
                    class_key = 'late_type'
                elif 'early' in morph_key or morph_key.startswith('e'):
                    class_key = 'early_type'
                else:
                    class_key = 'intermediate_type'
                
                if class_key in class_params:
                    v_pred_class = predict_rotation_curve_fast(
                        galaxy, class_params[class_key],
                        use_bulge_gate=True,
                        n_particles=100000
                    )
                    class_metrics = compute_metrics(galaxy.v_obs, v_pred_class, galaxy.v_err)
                    class_ape = class_metrics['ape']
                    improvement = class_ape - metrics['ape']  # Positive = better
            
            # Store result
            result = ZeroShotResult(
                galaxy_name=galaxy.name,
                morphology=galaxy.morphology,
                bulge_frac=bulge_frac,
                M_star=M_star,
                R_d=R_d,
                pred_eta=pred_params['eta'],
                pred_ring_amp=pred_params['ring_amp'],
                pred_M_max=pred_params['M_max'],
                ape=metrics['ape'],
                rmse=metrics['rmse'],
                chi2_reduced=metrics['chi2_reduced'],
                class_ape=class_ape,
                improvement_over_class=improvement
            )
            results.append(result)
            
            # Optionally save plot
            if save_plots and output_dir is not None:
                plot_rotation_curve(galaxy, v_pred_pop, v_pred_class,
                                  result, output_dir)
            
        except Exception as e:
            print(f"Warning: Failed to process {galaxy.name}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {len(results)}/{n_galaxies} galaxies in {elapsed:.1f}s")
    
    return results


def analyze_results(results: List[ZeroShotResult], 
                   has_class_comparison: bool = False) -> Dict:
    """
    Analyze zero-shot test results and generate summary statistics.
    
    Args:
        results: List of zero-shot results
        has_class_comparison: Whether class-wise comparison is available
    
    Returns:
        Dictionary of summary statistics
    """
    if len(results) == 0:
        return {}
    
    # Convert to arrays for analysis
    apes = np.array([r.ape for r in results])
    rmses = np.array([r.rmse for r in results])
    chi2s = np.array([r.chi2_reduced for r in results])
    
    summary = {
        'n_galaxies': len(results),
        'ape_median': float(np.median(apes)),
        'ape_mean': float(np.mean(apes)),
        'ape_std': float(np.std(apes)),
        'ape_q25': float(np.percentile(apes, 25)),
        'ape_q75': float(np.percentile(apes, 75)),
        'rmse_median': float(np.median(rmses)),
        'chi2_reduced_median': float(np.median(chi2s)),
        'success_rate': float(np.sum(apes < 30) / len(apes) * 100)  # % with APE < 30%
    }
    
    # Class-wise comparison
    if has_class_comparison:
        class_apes = np.array([r.class_ape for r in results if r.class_ape is not None])
        improvements = np.array([r.improvement_over_class for r in results 
                                if r.improvement_over_class is not None])
        
        if len(class_apes) > 0:
            summary['class_ape_median'] = float(np.median(class_apes))
            summary['improvement_median'] = float(np.median(improvements))
            summary['improvement_mean'] = float(np.mean(improvements))
            summary['n_better_than_class'] = int(np.sum(improvements > 0))
            summary['pct_better_than_class'] = float(np.sum(improvements > 0) / len(improvements) * 100)
    
    # Binned by bulge fraction
    bulge_fracs = np.array([r.bulge_frac for r in results])
    bins = [0.0, 0.15, 0.5, 1.0]
    bin_labels = ['Pure Disk (B/T<0.15)', 'Mixed (0.15≤B/T<0.5)', 'Bulge-Dominated (B/T≥0.5)']
    
    summary['performance_by_morphology'] = {}
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (bulge_fracs >= low) & (bulge_fracs < high)
        if np.sum(mask) > 0:
            summary['performance_by_morphology'][bin_labels[i]] = {
                'n': int(np.sum(mask)),
                'ape_median': float(np.median(apes[mask])),
                'ape_mean': float(np.mean(apes[mask]))
            }
    
    return summary


def print_summary(summary: Dict, has_class_comparison: bool = False):
    """Print formatted summary of zero-shot test results."""
    print(f"\n{'='*80}")
    print("ZERO-SHOT TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Galaxies Tested: {summary['n_galaxies']}")
    print()
    
    print("Overall Performance:")
    print(f"  Median APE:  {summary['ape_median']:.2f}%")
    print(f"  Mean APE:    {summary['ape_mean']:.2f}% ± {summary['ape_std']:.2f}%")
    print(f"  IQR:         [{summary['ape_q25']:.2f}%, {summary['ape_q75']:.2f}%]")
    print(f"  Success Rate (APE<30%): {summary['success_rate']:.1f}%")
    print()
    
    if has_class_comparison and 'class_ape_median' in summary:
        print("Comparison to Class-Wise Parameters:")
        print(f"  Class-wise median APE: {summary['class_ape_median']:.2f}%")
        print(f"  Population laws median APE: {summary['ape_median']:.2f}%")
        print(f"  Median improvement: {summary['improvement_median']:.2f}%")
        print(f"  Galaxies improved: {summary['n_better_than_class']}/{summary['n_galaxies']} "
              f"({summary['pct_better_than_class']:.1f}%)")
        print()
    
    if 'performance_by_morphology' in summary:
        print("Performance by Morphology:")
        for label, stats in summary['performance_by_morphology'].items():
            print(f"  {label}:")
            print(f"    N = {stats['n']}, Median APE = {stats['ape_median']:.2f}%, "
                  f"Mean APE = {stats['ape_mean']:.2f}%")
        print()
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Testing of Population Laws')
    parser.add_argument('--laws', required=True, help='Path to fitted population_laws.json')
    parser.add_argument('--sparc_dir', default='data/Rotmod_LTG', help='SPARC data directory')
    parser.add_argument('--class_params', help='Optional: class-wise params for comparison')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--save_plots', action='store_true', help='Save individual RC plots')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load population laws
    print(f"Loading population laws from {args.laws}")
    with open(args.laws, 'r') as f:
        laws_data = json.load(f)
    
    # Reconstruct PopulationLaws object
    laws_dict = laws_data['population_laws']
    pop_laws = PopulationLaws(
        eta_a0=laws_dict['eta_a0'],
        eta_a1=laws_dict['eta_a1'],
        eta_a2=laws_dict['eta_a2'],
        eta_a3=laws_dict['eta_a3'],
        ring_amp_b0=laws_dict['ring_amp_b0'],
        ring_amp_b1=laws_dict['ring_amp_b1'],
        M_max_c0=laws_dict['M_max_c0'],
        M_max_c1=laws_dict['M_max_c1'],
        M_max_c2=laws_dict['M_max_c2'],
        lambda_hat_fixed=laws_dict.get('lambda_hat_fixed', 20.0),
        bulge_gate_power_fixed=laws_dict.get('bulge_gate_power_fixed', 32.9)
    )
    
    print("Population Laws Loaded:")
    print(f"  eta(B/T, M*) = {pop_laws.eta_a0:.3f} + {pop_laws.eta_a1:.3f}*log(M*) + "
          f"{pop_laws.eta_a2:.3f}*B/T + {pop_laws.eta_a3:.3f}*(B/T)^2")
    print(f"  ring_amp(B/T) = {pop_laws.ring_amp_b0:.3f} * exp(-{pop_laws.ring_amp_b1:.3f}*B/T)")
    print(f"  M_max(M*, Rd) = {pop_laws.M_max_c0:.3f} + {pop_laws.M_max_c1:.3f}*log(M*) + "
          f"{pop_laws.M_max_c2:.3f}*log(Rd)")
    
    # Load class-wise params if provided
    class_params = None
    if args.class_params:
        print(f"\nLoading class-wise params from {args.class_params}")
        class_params = load_class_params(Path(args.class_params))
        print(f"Loaded {len(class_params)} morphological classes")
    
    # Load galaxies
    data_dir = Path(args.sparc_dir)
    master_file = data_dir / "MasterSheet_SPARC.mrt"
    
    print(f"\nLoading SPARC galaxies from {args.sparc_dir}")
    master_info = load_sparc_master_table(master_file)
    
    rotmod_files = sorted(data_dir.glob('*_rotmod.dat'))
    galaxies = []
    for rotmod_file in rotmod_files:
        try:
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            galaxies.append(galaxy)
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to load {rotmod_file.name}: {e}")
    
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Run zero-shot tests
    results = test_population_laws(
        galaxies,
        pop_laws,
        class_params=class_params,
        save_plots=args.save_plots,
        output_dir=output_dir,
        verbose=args.verbose
    )
    
    # Analyze and summarize
    summary = analyze_results(results, has_class_comparison=(class_params is not None))
    print_summary(summary, has_class_comparison=(class_params is not None))
    
    # Save detailed results
    results_file = output_dir / 'zero_shot_results.json'
    output_data = {
        'population_laws': laws_dict,
        'summary': summary,
        'per_galaxy_results': [r.to_dict() for r in results]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save CSV for easy analysis
    csv_file = output_dir / 'zero_shot_results.csv'
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")


if __name__ == '__main__':
    main()

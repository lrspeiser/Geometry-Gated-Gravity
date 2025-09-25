#!/usr/bin/env python3
"""
Zero-shot testing framework for unified G³ model.
Train on one source, freeze parameters, test on completely different targets.
NO per-galaxy tuning allowed.
"""

import numpy as np
import json
import hashlib
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from g3_unified_global import UnifiedG3Model, optimize_global_theta
import os
import sys

class ZeroShotTestSuite:
    """
    Zero-shot test harness that ensures no per-galaxy parameter tuning.
    """
    
    def __init__(self, freeze_file=None):
        """Initialize with optional frozen parameter file"""
        self.frozen_theta = None
        self.theta_hash = None
        self.run_manifest = {}
        
        if freeze_file:
            self.load_frozen_parameters(freeze_file)
            
    def load_frozen_parameters(self, filepath):
        """Load and lock parameters - no modification allowed after this"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.frozen_theta = data['theta']
        self.theta_hash = data['theta_hash']
        
        print(f"\n{'='*60}")
        print(f" PARAMETERS FROZEN ")
        print(f"{'='*60}")
        print(f"Hash: {self.theta_hash}")
        print("NO modifications allowed from this point!")
        
        # Write to manifest
        self.run_manifest['frozen_theta_hash'] = self.theta_hash
        self.run_manifest['frozen_at'] = datetime.now().isoformat()
        
    def train_model(self, train_data, name="Training"):
        """Train model on source data - parameters will be frozen after this"""
        
        if self.frozen_theta is not None:
            raise ValueError("Parameters already frozen! Cannot retrain!")
            
        print(f"\n{'='*60}")
        print(f" TRAINING ON: {name} ")
        print(f"{'='*60}")
        
        # Optimize global parameters
        best_theta, history = optimize_global_theta(
            train_data, 
            max_iter=500,
            pop_size=64,
            use_gpu=True
        )
        
        # Create model and freeze
        model = UnifiedG3Model(best_theta)
        self.frozen_theta = model.theta
        self.theta_hash = model.theta_hash
        
        # Save frozen parameters
        freeze_file = f'out/mw_orchestrated/frozen_theta_{self.theta_hash}.json'
        model.save_parameters(freeze_file)
        
        print(f"\nParameters FROZEN with hash: {self.theta_hash}")
        print(f"Saved to: {freeze_file}")
        
        # Update manifest
        self.run_manifest['training_source'] = name
        self.run_manifest['frozen_theta_hash'] = self.theta_hash
        self.run_manifest['training_complete'] = datetime.now().isoformat()
        
        return model
        
    def test_zero_shot(self, test_data, name="Test", save_results=True):
        """
        Test on completely unseen data with frozen parameters.
        NO retuning allowed!
        """
        
        if self.frozen_theta is None:
            raise ValueError("No frozen parameters! Train first or load frozen file.")
            
        print(f"\n{'='*60}")
        print(f" ZERO-SHOT TEST ON: {name} ")
        print(f"{'='*60}")
        print(f"Using FROZEN parameters (hash: {self.theta_hash})")
        
        # Create model with frozen parameters
        model = UnifiedG3Model(self.frozen_theta)
        
        # Verify hash hasn't changed
        if model.theta_hash != self.theta_hash:
            raise ValueError(f"Parameter tampering detected! Expected {self.theta_hash}, got {model.theta_hash}")
            
        # Compute baryon properties from test data
        r_half, Sigma_mean = model.compute_baryon_properties(
            test_data['R'], test_data['z'], test_data['Sigma_loc']
        )
        
        # Audit log
        print(f"\nTest galaxy properties (computed from baryons):")
        print(f"  r_half = {r_half:.2f} kpc")
        print(f"  Σ_mean = {Sigma_mean:.1f} Msun/pc²")
        
        # Make predictions with frozen model
        v_pred = model.predict_velocity(
            test_data['R'], test_data['z'],
            test_data['Sigma_loc'], test_data['gN'],
            r_half, Sigma_mean
        )
        
        # Calculate metrics
        v_obs = test_data['v_obs']
        residuals = v_pred - v_obs
        rel_error = np.abs(residuals / v_obs) * 100
        
        metrics = {
            'name': name,
            'theta_hash': self.theta_hash,
            'n_points': len(v_obs),
            'r_half': float(r_half),
            'Sigma_mean': float(Sigma_mean),
            'median_error_pct': float(np.median(rel_error)),
            'mean_error_pct': float(np.mean(rel_error)),
            'std_error_pct': float(np.std(rel_error)),
            'frac_within_5pct': float(np.mean(rel_error < 5)),
            'frac_within_10pct': float(np.mean(rel_error < 10)),
            'frac_within_20pct': float(np.mean(rel_error < 20)),
            'percentiles': {
                '25th': float(np.percentile(rel_error, 25)),
                '50th': float(np.percentile(rel_error, 50)),
                '75th': float(np.percentile(rel_error, 75)),
                '95th': float(np.percentile(rel_error, 95))
            }
        }
        
        print(f"\nZERO-SHOT RESULTS:")
        print(f"  Median error: {metrics['median_error_pct']:.2f}%")
        print(f"  Mean error: {metrics['mean_error_pct']:.2f}%")
        print(f"  Stars <10%: {metrics['frac_within_10pct']*100:.1f}%")
        print(f"  Stars <20%: {metrics['frac_within_20pct']*100:.1f}%")
        
        # Check for systematic biases
        R = test_data['R']
        R_bins = np.percentile(R, [0, 25, 50, 75, 100])
        
        print(f"\nError by radius quartile:")
        for i in range(len(R_bins)-1):
            mask = (R >= R_bins[i]) & (R < R_bins[i+1])
            if np.any(mask):
                quartile_error = np.median(rel_error[mask])
                print(f"  Q{i+1} (R={R_bins[i]:.1f}-{R_bins[i+1]:.1f} kpc): {quartile_error:.1f}%")
                
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f'out/mw_orchestrated/zero_shot_{name}_{timestamp}.json'
            
            output = {
                'test_name': name,
                'timestamp': datetime.now().isoformat(),
                'frozen_theta_hash': self.theta_hash,
                'metrics': metrics,
                'NO_PER_GALAXY_PARAMS': True
            }
            
            with open(results_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to: {results_file}")
            
        return metrics, v_pred
        
    def cross_validate_sparc(self, sparc_data, k_folds=5):
        """
        K-fold cross validation on SPARC galaxies.
        Each fold trains on k-1 galaxies and tests on held-out galaxies.
        """
        
        print(f"\n{'='*70}")
        print(f" K-FOLD CROSS VALIDATION (k={k_folds}) ")
        print(f"{'='*70}")
        
        # Implementation would split SPARC by galaxy
        # For now, placeholder
        print("SPARC cross-validation not yet implemented")
        print("Would split by galaxy, stratified by morphology")
        
        return {}
        
    def leave_one_morphology_out(self, data_by_morphology):
        """
        Train on all morphologies except one, test on held-out morphology.
        Ultimate test of generalization.
        """
        
        print(f"\n{'='*70}")
        print(f" LEAVE-ONE-MORPHOLOGY-OUT TEST ")
        print(f"{'='*70}")
        
        results = {}
        
        for test_morph in data_by_morphology.keys():
            print(f"\nHolding out: {test_morph}")
            
            # Combine training morphologies
            train_data = {'R': [], 'z': [], 'v_obs': [], 'Sigma_loc': [], 'gN': []}
            
            for morph, data in data_by_morphology.items():
                if morph != test_morph:
                    for key in train_data.keys():
                        train_data[key].extend(data[key])
                        
            # Convert to arrays
            for key in train_data.keys():
                train_data[key] = np.array(train_data[key])
                
            # Train on combined data
            self.frozen_theta = None  # Reset for new training
            model = self.train_model(train_data, f"All except {test_morph}")
            
            # Test on held-out morphology
            test_data = data_by_morphology[test_morph]
            metrics, _ = self.test_zero_shot(test_data, test_morph, save_results=False)
            
            results[test_morph] = metrics
            
        return results
        
    def ablation_test(self, test_data):
        """
        Test with components turned off to show their contribution.
        """
        
        if self.frozen_theta is None:
            raise ValueError("No frozen parameters!")
            
        print(f"\n{'='*70}")
        print(f" ABLATION TESTS ")
        print(f"{'='*70}")
        
        # Base model
        model_base = UnifiedG3Model(self.frozen_theta)
        r_half, Sigma_mean = model_base.compute_baryon_properties(
            test_data['R'], test_data['z'], test_data['Sigma_loc']
        )
        
        v_base = model_base.predict_velocity(
            test_data['R'], test_data['z'],
            test_data['Sigma_loc'], test_data['gN'],
            r_half, Sigma_mean
        )
        
        error_base = np.median(np.abs((v_base - test_data['v_obs']) / test_data['v_obs']) * 100)
        print(f"Full model: {error_base:.2f}% error")
        
        # Ablation 1: No inner/outer blend (p=1 everywhere)
        theta_ablate1 = self.frozen_theta.copy()
        theta_ablate1['p_in'] = 1.0
        theta_ablate1['p_out'] = 1.0
        model_ablate1 = UnifiedG3Model(theta_ablate1)
        
        v_ablate1 = model_ablate1.predict_velocity(
            test_data['R'], test_data['z'],
            test_data['Sigma_loc'], test_data['gN'],
            r_half, Sigma_mean
        )
        
        error_ablate1 = np.median(np.abs((v_ablate1 - test_data['v_obs']) / test_data['v_obs']) * 100)
        print(f"No p(R) blend: {error_ablate1:.2f}% error (Δ={error_ablate1-error_base:+.1f}%)")
        
        # Ablation 2: No geometry scaling (fixed rc)
        theta_ablate2 = self.frozen_theta.copy()
        theta_ablate2['gamma'] = 0.0
        theta_ablate2['beta'] = 0.0
        model_ablate2 = UnifiedG3Model(theta_ablate2)
        
        v_ablate2 = model_ablate2.predict_velocity(
            test_data['R'], test_data['z'],
            test_data['Sigma_loc'], test_data['gN'],
            r_half, Sigma_mean
        )
        
        error_ablate2 = np.median(np.abs((v_ablate2 - test_data['v_obs']) / test_data['v_obs']) * 100)
        print(f"No rc(geometry): {error_ablate2:.2f}% error (Δ={error_ablate2-error_base:+.1f}%)")
        
        # Ablation 3: No screening
        theta_ablate3 = self.frozen_theta.copy()
        theta_ablate3['alpha'] = 0.0
        model_ablate3 = UnifiedG3Model(theta_ablate3)
        
        v_ablate3 = model_ablate3.predict_velocity(
            test_data['R'], test_data['z'],
            test_data['Sigma_loc'], test_data['gN'],
            r_half, Sigma_mean
        )
        
        error_ablate3 = np.median(np.abs((v_ablate3 - test_data['v_obs']) / test_data['v_obs']) * 100)
        print(f"No screening: {error_ablate3:.2f}% error (Δ={error_ablate3-error_base:+.1f}%)")
        
        return {
            'full': error_base,
            'no_p_blend': error_ablate1,
            'no_geometry': error_ablate2,
            'no_screening': error_ablate3
        }
        
    def save_manifest(self):
        """Save run manifest with SHA verification"""
        manifest_file = 'out/mw_orchestrated/run_manifest.json'
        
        self.run_manifest['timestamp'] = datetime.now().isoformat()
        self.run_manifest['NO_PER_GALAXY_PARAMS'] = True
        
        with open(manifest_file, 'w') as f:
            json.dump(self.run_manifest, f, indent=2)
            
        print(f"\nRun manifest saved to: {manifest_file}")
        

def main():
    """Main test runner"""
    
    parser = argparse.ArgumentParser(description='Zero-shot G³ model testing')
    parser.add_argument('--train', type=str, help='Training data source')
    parser.add_argument('--test', type=str, help='Test data target')
    parser.add_argument('--freeze', type=str, help='Use frozen parameters from file')
    parser.add_argument('--ablation', action='store_true', help='Run ablation tests')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" ZERO-SHOT TEST SUITE ")
    print("="*80)
    print("Enforcing GLOBAL parameters only - NO per-galaxy tuning!")
    
    # Create test suite
    suite = ZeroShotTestSuite(freeze_file=args.freeze)
    
    # Load Milky Way data for testing
    print("\nLoading Milky Way data...")
    data = np.load('data/mw_gaia_144k.npz')
    
    mw_data = {
        'R': data['R_kpc'],
        'z': data['z_kpc'],
        'v_obs': data['v_obs_kms'],
        'Sigma_loc': data['Sigma_loc_Msun_pc2'],
        'gN': data['gN_kms2_per_kpc']
    }
    
    if not args.freeze:
        # Train on MW
        print("\nTraining on Milky Way...")
        model = suite.train_model(mw_data, "Milky Way 144k stars")
        
    # Test zero-shot on same data (sanity check)
    print("\nTesting on training data (sanity check)...")
    metrics, v_pred = suite.test_zero_shot(mw_data, "MW_self_test")
    
    # Run ablation if requested
    if args.ablation:
        ablation_results = suite.ablation_test(mw_data)
        
    # Save manifest
    suite.save_manifest()
    
    print("\n" + "="*80)
    print(" TEST SUITE COMPLETE ")
    print("="*80)
    print(f"All tests used frozen Θ with hash: {suite.theta_hash}")
    print("NO per-galaxy parameters were used!")
    

if __name__ == '__main__':
    main()
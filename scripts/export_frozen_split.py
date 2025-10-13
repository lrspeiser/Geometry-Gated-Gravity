"""
Export Frozen Train/Test Split and Optimal Hyperparameters

This creates a JSON file with:
- Train/test galaxy indices (stratified 80/20)
- Optimal hyperparameters from v-pathspec-0.9-rar0p087
- Data provenance and metadata

This split will be used for all subsequent blind prediction tests.
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from validation_suite import ValidationSuite
from path_spectrum_kernel_track2 import PathSpectrumHyperparams

def main():
    print("="*80)
    print("EXPORTING FROZEN TRAIN/TEST SPLIT")
    print("="*80)
    print("\nTag: v-pathspec-0.9-rar0p087")
    print("Date: 2025-10-13\n")
    
    # Load SPARC data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Perform stratified split (same seed=42 as optimization)
    train_df, test_df = suite.perform_train_test_split()
    
    # Extract galaxy names/indices
    train_galaxies = []
    for idx, row in train_df.iterrows():
        train_galaxies.append({
            'index': int(idx),
            'galaxy': row['Galaxy'],
            'type': row['type']
        })
    
    test_galaxies = []
    for idx, row in test_df.iterrows():
        test_galaxies.append({
            'index': int(idx),
            'galaxy': row['Galaxy'],
            'type': row['type']
        })
    
    # Optimal hyperparameters from 200-iter optimization
    optimal_hp = PathSpectrumHyperparams(
        L_0=4.992525,
        beta_bulge=1.759351,
        alpha_shear=0.149265,
        gamma_bar=1.931874,
        A_0=0.590634,
        p=0.756591,
        n_coh=0.500019,
        g_dagger=1.2e-10  # Fixed literature value
    )
    
    # Create export structure
    export_data = {
        'metadata': {
            'tag': 'v-pathspec-0.9-rar0p087',
            'export_date': datetime.now().isoformat(),
            'total_galaxies': len(df),
            'train_galaxies': len(train_df),
            'test_galaxies': len(test_df),
            'split_method': 'stratified_by_morphology',
            'random_seed': 42,
            'test_fraction': 0.2
        },
        'performance': {
            'train_rar_scatter': 0.084,
            'test_rar_scatter': 0.087,
            'test_rar_bias': -0.078,
            'train_median_ape': 17.5,
            'test_median_ape': 19.1,
            'target_rar': 0.15,
            'achieved': True
        },
        'hyperparameters': optimal_hp.to_dict(),
        'train_set': train_galaxies,
        'test_set': test_galaxies,
        'morphology_distribution': {
            'train': {k: int(v) for k, v in train_df['type'].value_counts().items()},
            'test': {k: int(v) for k, v in test_df['type'].value_counts().items()}
        }
    }
    
    # Save to splits directory
    splits_dir = Path("C:/Users/henry/dev/GravityCalculator/splits")
    splits_dir.mkdir(exist_ok=True)
    
    output_path = splits_dir / "sparc_split_v1.json"
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Frozen split exported to: {output_path}")
    print(f"\nSummary:")
    print(f"  Training galaxies: {len(train_galaxies)}")
    print(f"  Test galaxies: {len(test_galaxies)}")
    print(f"  Morphology types: {len(df['type'].unique())}")
    print(f"\nOptimal hyperparameters:")
    for key, value in optimal_hp.to_dict().items():
        print(f"  {key:<15} = {value:.6f}")
    
    print(f"\n✅ This split is now FROZEN for all blind prediction tests")
    print(f"   No retraining allowed on test set!")

if __name__ == "__main__":
    main()

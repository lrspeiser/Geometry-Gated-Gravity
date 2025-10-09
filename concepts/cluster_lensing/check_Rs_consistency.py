#!/usr/bin/env python3
"""
Diagnostic: Check Rs_kpc consistency with learned rule.

Reads universal_model.json and verifies that fitted Rs_kpc values
match the learned rule Rs ≈ 0.9·R_edge within tolerance.

Flags any deviations >10% and generates diagnostic plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def check_Rs_consistency(model_path: str, tolerance: float = 0.10):
    """
    Check if fitted Rs_kpc matches learned rule Rs ≈ 0.9·R_edge.
    
    Args:
        model_path: Path to universal_model.json
        tolerance: Fractional tolerance for flagging deviations (default 10%)
    
    Returns:
        dict: Results with deviations and diagnostic info
    """
    with open(model_path, 'r') as f:
        model = json.load(f)
    
    results = {
        'clusters': [],
        'max_deviation': 0.0,
        'all_pass': True
    }
    
    print("\n" + "="*70)
    print("Rs_kpc CONSISTENCY CHECK")
    print("Learned rule: Rs ≈ 0.9·R_edge")
    print("="*70 + "\n")
    
    # Build lookup dictionaries
    features_dict = {f['cluster_name']: f for f in model['features']}
    params_dict = {p['cluster_name']: p for p in model['parameters']}
    
    for name in features_dict.keys():
        features = features_dict[name]
        params = params_dict[name]
        
        R_edge = features['R_edge']
        Rs_expected = 0.9 * R_edge
        Rs_fitted = params['Rs_kpc']
        
        deviation = abs(Rs_fitted - Rs_expected) / Rs_expected
        passed = deviation <= tolerance
        
        results['clusters'].append({
            'name': name,
            'R_edge': R_edge,
            'Rs_expected': Rs_expected,
            'Rs_fitted': Rs_fitted,
            'deviation': deviation,
            'passed': passed
        })
        
        results['max_deviation'] = max(results['max_deviation'], deviation)
        if not passed:
            results['all_pass'] = False
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}:")
        print(f"  R_edge      = {R_edge:7.1f} kpc")
        print(f"  Rs (expect) = {Rs_expected:7.1f} kpc  (0.9·R_edge)")
        print(f"  Rs (fitted) = {Rs_fitted:7.1f} kpc")
        print(f"  Deviation   = {deviation*100:5.1f}%  {status}")
        print()
    
    print("="*70)
    if results['all_pass']:
        print(f"✅ ALL CHECKS PASSED (max deviation: {results['max_deviation']*100:.1f}%)")
    else:
        print(f"❌ SOME CHECKS FAILED (max deviation: {results['max_deviation']*100:.1f}%)")
    print("="*70 + "\n")
    
    return results


def plot_Rs_diagnostic(results: dict, output_dir: str = "out/universal_lensing_training"):
    """Generate diagnostic plot showing Rs consistency."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    clusters = results['clusters']
    names = [c['name'] for c in clusters]
    Rs_expected = [c['Rs_expected'] for c in clusters]
    Rs_fitted = [c['Rs_fitted'] for c in clusters]
    deviations = [c['deviation'] * 100 for c in clusters]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Expected vs Fitted
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, Rs_expected, width, label='Expected (0.9·R_edge)', 
            color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, Rs_fitted, width, label='Fitted', 
            color='coral', alpha=0.8)
    
    ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rs [kpc]', fontsize=12, fontweight='bold')
    ax1.set_title('Rs Consistency: Expected vs Fitted', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Deviation percentages
    colors = ['green' if c['passed'] else 'red' for c in clusters]
    bars = ax2.bar(x, deviations, color=colors, alpha=0.7)
    
    ax2.axhline(10, color='red', linestyle='--', linewidth=2, label='10% tolerance')
    ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Deviation [%]', fontsize=12, fontweight='bold')
    ax2.set_title('Rs Deviation from Learned Rule', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, dev in zip(bars, deviations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{dev:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "Rs_consistency_diagnostic.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Diagnostic plot saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    model_path = "out/universal_lensing_training/universal_model.json"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Run train_universal_lensing_model.py first.")
        exit(1)
    
    results = check_Rs_consistency(model_path)
    plot_Rs_diagnostic(results)

#!/usr/bin/env python3
"""
Extract Bar Classification from SPARC Hubble Types
=================================================

Parses Hubble type strings to extract bar strength/presence.
Creates bar gating factors for ring amplitude and lambda suppression.

Bar classification:
- SA (unbarred): g_bar = 1.0 (no suppression)
- SAB (weakly barred): g_bar = 0.7-0.8 (moderate suppression)
- SB (strongly barred): g_bar = 0.4-0.5 (strong suppression)

Physical interpretation:
Bars destroy long, phase-coherent azimuthal loops by introducing
non-axisymmetric torques. The ring term should be suppressed.
"""
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def classify_bar_strength(hubble_type):
    """
    Classify bar strength from Hubble type string.
    
    Args:
        hubble_type: String like "SABc", "SBb", "Scd", etc.
    
    Returns:
        tuple: (bar_class, bar_strength)
            bar_class: 'SA', 'SAB', 'SB', 'S', or 'unknown'
            bar_strength: float in [0, 1] where 1 = strong bar, 0 = no bar
    """
    if not hubble_type or hubble_type == 'Unknown':
        return 'unknown', 0.5  # Neutral default
    
    ht = str(hubble_type).upper().strip()
    
    # Check for bar indicators
    if 'SB' in ht and 'SAB' not in ht:
        # Strongly barred
        return 'SB', 1.0
    elif 'SAB' in ht:
        # Weakly/moderately barred
        return 'SAB', 0.6
    elif 'SA' in ht:
        # Explicitly unbarred
        return 'SA', 0.0
    elif ht.startswith('S') or ht.startswith('I'):
        # Generic spiral or irregular (assume unbarred unless specified)
        return 'S', 0.2  # Slight uncertainty
    else:
        # Elliptical or unknown
        return 'unknown', 0.5


def compute_bar_gate(bar_strength, g_SA=1.0, g_SAB=0.75, g_SB=0.45, steepness=3.0):
    """
    Compute smooth bar gating factor.
    
    g_bar = g_SB + (g_SA - g_SB) * (1 - B_strength)^n
    
    Args:
        bar_strength: Bar strength in [0, 1] from classify_bar_strength
        g_SA: Gate value for unbarred (default 1.0 = no suppression)
        g_SAB: Gate value for weakly barred (default 0.75)
        g_SB: Gate value for strongly barred (default 0.45)
        steepness: Transition steepness
    
    Returns:
        float: Bar gating factor in [g_SB, g_SA]
    """
    # Smooth interpolation
    g_bar = g_SB + (g_SA - g_SB) * (1.0 - bar_strength) ** steepness
    return np.clip(g_bar, g_SB, g_SA)


def main():
    parser = argparse.ArgumentParser(description='Extract bar classifications from SPARC')
    parser.add_argument('--disk_params', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'),
                       help='Path to disk parameters with Hubble types')
    parser.add_argument('--output', type=Path,
                       default=Path('many_path_model/bt_law/sparc_bar_classification.json'),
                       help='Output JSON with bar classifications')
    parser.add_argument('--g_SA', type=float, default=1.0,
                       help='Gate value for unbarred spirals')
    parser.add_argument('--g_SAB', type=float, default=0.75,
                       help='Gate value for weakly barred spirals')
    parser.add_argument('--g_SB', type=float, default=0.45,
                       help='Gate value for strongly barred spirals')
    parser.add_argument('--steepness', type=float, default=3.0,
                       help='Transition steepness')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXTRACTING BAR CLASSIFICATIONS FROM SPARC HUBBLE TYPES")
    print("="*80)
    
    # Load disk parameters
    print(f"\nLoading disk parameters from: {args.disk_params}")
    with open(args.disk_params, 'r') as f:
        disk_params = json.load(f)
    
    print(f"  Loaded {len(disk_params)} galaxies")
    
    # Process each galaxy
    bar_data = {}
    bar_counts = {'SA': 0, 'SAB': 0, 'SB': 0, 'S': 0, 'unknown': 0}
    
    print("\nClassifying bar strength...")
    for name, params in disk_params.items():
        hubble_type = params.get('hubble_type', 'Unknown')
        
        bar_class, bar_strength = classify_bar_strength(hubble_type)
        g_bar = compute_bar_gate(bar_strength, 
                                 g_SA=args.g_SA,
                                 g_SAB=args.g_SAB,
                                 g_SB=args.g_SB,
                                 steepness=args.steepness)
        
        bar_data[name] = {
            'name': name,
            'hubble_type': hubble_type,
            'bar_class': bar_class,
            'bar_strength': float(bar_strength),
            'g_bar': float(g_bar)
        }
        
        bar_counts[bar_class] += 1
    
    # Print statistics
    print("\n" + "-"*80)
    print("BAR CLASSIFICATION STATISTICS")
    print("-"*80)
    total = len(bar_data)
    for bar_class, count in sorted(bar_counts.items()):
        pct = 100 * count / total
        print(f"  {bar_class:10s}: {count:3d} ({pct:5.1f}%)")
    
    # Distribution of g_bar values
    g_bars = [d['g_bar'] for d in bar_data.values()]
    print(f"\nBar gating factor (g_bar) distribution:")
    print(f"  Mean:   {np.mean(g_bars):.3f}")
    print(f"  Median: {np.median(g_bars):.3f}")
    print(f"  Range:  [{np.min(g_bars):.3f}, {np.max(g_bars):.3f}]")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(bar_data, f, indent=2)
    
    print(f"\n[OK] Bar classifications saved to: {args.output}")
    
    # Also save summary CSV
    csv_file = args.output.with_suffix('.csv')
    df = pd.DataFrame(bar_data.values())
    df.to_csv(csv_file, index=False)
    print(f"[OK] Summary CSV saved to: {csv_file}")
    
    # Show examples
    print("\n" + "-"*80)
    print("EXAMPLE CLASSIFICATIONS")
    print("-"*80)
    examples = list(bar_data.values())[:10]
    for ex in examples:
        print(f"  {ex['name']:12s} {ex['hubble_type']:8s} → {ex['bar_class']:8s} (g_bar={ex['g_bar']:.2f})")
    
    print("="*80)


if __name__ == '__main__':
    main()

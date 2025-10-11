#!/usr/bin/env python3
"""
Analyze and visualize SPARC zero-shot test results.

This script compares the standard kernel vs bulge-gated kernel performance
across different galaxy types.

Usage:
    python analyze_sparc_results.py \
        --standard results/sparc_standard_kernel.csv \
        --bulge_gated results/sparc_bulge_gated_kernel.csv \
        --output results/sparc_comparison.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(filepath):
    """Load results CSV file."""
    return pd.read_csv(filepath)


def compare_results(df_standard, df_bulge_gated):
    """
    Compare standard and bulge-gated results.
    
    Returns a merged dataframe with comparison metrics.
    """
    # Merge on galaxy name
    df_merged = df_standard.merge(
        df_bulge_gated, 
        on='name', 
        suffixes=('_standard', '_bulge_gated')
    )
    
    # Compute improvement
    df_merged['ape_improvement'] = df_merged['ape_standard'] - df_merged['ape_bulge_gated']
    df_merged['ape_improvement_pct'] = (df_merged['ape_improvement'] / df_merged['ape_standard']) * 100
    
    return df_merged


def create_comparison_plot(df_merged, output_path):
    """
    Create comprehensive comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: APE comparison scatter
    ax = axes[0, 0]
    colors = {'disk_dominated': 'blue', 'intermediate': 'green', 'bulge_dominated': 'red'}
    
    for gtype in ['disk_dominated', 'intermediate', 'bulge_dominated']:
        mask = df_merged['type_standard'] == gtype
        if mask.sum() > 0:
            ax.scatter(
                df_merged.loc[mask, 'ape_standard'],
                df_merged.loc[mask, 'ape_bulge_gated'],
                c=colors.get(gtype, 'gray'),
                label=gtype.replace('_', ' ').title(),
                alpha=0.7,
                s=100
            )
    
    # Diagonal line (no improvement)
    max_ape = max(df_merged['ape_standard'].max(), df_merged['ape_bulge_gated'].max())
    ax.plot([0, max_ape], [0, max_ape], 'k--', alpha=0.3, label='No change')
    
    ax.set_xlabel('APE Standard Kernel (%)', fontsize=12)
    ax.set_ylabel('APE Bulge-Gated Kernel (%)', fontsize=12)
    ax.set_title('APE Comparison: Standard vs Bulge-Gated', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: APE improvement by galaxy type
    ax = axes[0, 1]
    types_present = df_merged['type_standard'].unique()
    improvements_by_type = []
    labels = []
    
    for gtype in ['disk_dominated', 'intermediate', 'bulge_dominated']:
        if gtype in types_present:
            mask = df_merged['type_standard'] == gtype
            improvements = df_merged.loc[mask, 'ape_improvement']
            if len(improvements) > 0:
                improvements_by_type.append(improvements.values)
                labels.append(gtype.replace('_', ' ').title())
    
    if improvements_by_type:
        bp = ax.boxplot(improvements_by_type, labels=labels, patch_artist=True)
        for patch, gtype in zip(bp['boxes'], labels):
            patch.set_facecolor(colors.get(gtype.lower().replace(' ', '_'), 'gray'))
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('APE Improvement (%)', fontsize=12)
        ax.set_title('APE Improvement by Galaxy Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    # Plot 3: Histogram of APE values
    ax = axes[1, 0]
    ax.hist(df_merged['ape_standard'], bins=20, alpha=0.5, label='Standard', color='blue', edgecolor='black')
    ax.hist(df_merged['ape_bulge_gated'], bins=20, alpha=0.5, label='Bulge-Gated', color='red', edgecolor='black')
    ax.axvline(x=15, color='green', linestyle='--', linewidth=2, label='Success Threshold (15%)')
    ax.set_xlabel('APE (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of APE Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Per-galaxy improvement bar chart (top 10 by absolute improvement)
    ax = axes[1, 1]
    df_sorted = df_merged.sort_values('ape_improvement', ascending=False).head(10)
    
    bar_colors = [colors.get(t, 'gray') for t in df_sorted['type_standard']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['ape_improvement'], color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['name'], fontsize=9)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('APE Improvement (%)', fontsize=12)
    ax.set_title('Top 10 Galaxies by APE Improvement', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")


def print_summary_statistics(df_merged):
    """Print detailed summary statistics."""
    print("\n" + "="*80)
    print("SPARC ZERO-SHOT TEST: STANDARD vs BULGE-GATED KERNEL COMPARISON")
    print("="*80)
    
    print(f"\nTotal galaxies analyzed: {len(df_merged)}")
    
    # Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Standard Kernel:")
    print(f"  Mean APE: {df_merged['ape_standard'].mean():.2f}%")
    print(f"  Median APE: {df_merged['ape_standard'].median():.2f}%")
    print(f"  Success rate (APE < 15%): {(df_merged['ape_standard'] < 15).mean()*100:.1f}%")
    
    print(f"\nBulge-Gated Kernel:")
    print(f"  Mean APE: {df_merged['ape_bulge_gated'].mean():.2f}%")
    print(f"  Median APE: {df_merged['ape_bulge_gated'].median():.2f}%")
    print(f"  Success rate (APE < 15%): {(df_merged['ape_bulge_gated'] < 15).mean()*100:.1f}%")
    
    print(f"\nImprovement:")
    print(f"  Mean APE improvement: {df_merged['ape_improvement'].mean():.2f}%")
    print(f"  Median APE improvement: {df_merged['ape_improvement'].median():.2f}%")
    print(f"  Galaxies improved: {(df_merged['ape_improvement'] > 0).sum()} ({(df_merged['ape_improvement'] > 0).mean()*100:.1f}%)")
    print(f"  Galaxies worsened: {(df_merged['ape_improvement'] < 0).sum()} ({(df_merged['ape_improvement'] < 0).mean()*100:.1f}%)")
    
    # By galaxy type
    print("\n--- Statistics by Galaxy Type ---")
    for gtype in ['disk_dominated', 'intermediate', 'bulge_dominated']:
        mask = df_merged['type_standard'] == gtype
        if mask.sum() > 0:
            print(f"\n{gtype.replace('_', ' ').title()}:")
            print(f"  N = {mask.sum()}")
            print(f"  Standard APE: {df_merged.loc[mask, 'ape_standard'].mean():.2f}%")
            print(f"  Bulge-gated APE: {df_merged.loc[mask, 'ape_bulge_gated'].mean():.2f}%")
            print(f"  Mean improvement: {df_merged.loc[mask, 'ape_improvement'].mean():.2f}%")
            print(f"  Improved: {(df_merged.loc[mask, 'ape_improvement'] > 0).sum()}/{mask.sum()}")
    
    # Top improvers and decliners
    print("\n--- Top 5 Improved Galaxies ---")
    top_improved = df_merged.nlargest(5, 'ape_improvement')
    for _, row in top_improved.iterrows():
        print(f"  {row['name']}: {row['ape_standard']:.1f}% → {row['ape_bulge_gated']:.1f}% "
              f"(Δ={row['ape_improvement']:.1f}%, Type={row['type_standard']})")
    
    print("\n--- Top 5 Declined Galaxies ---")
    top_declined = df_merged.nsmallest(5, 'ape_improvement')
    for _, row in top_declined.iterrows():
        print(f"  {row['name']}: {row['ape_standard']:.1f}% → {row['ape_bulge_gated']:.1f}% "
              f"(Δ={row['ape_improvement']:.1f}%, Type={row['type_standard']})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze SPARC zero-shot test results")
    parser.add_argument('--standard', type=str, required=True,
                       help='Path to standard kernel results CSV')
    parser.add_argument('--bulge_gated', type=str, required=True,
                       help='Path to bulge-gated kernel results CSV')
    parser.add_argument('--output', type=str, default='sparc_comparison.png',
                       help='Output path for comparison plot')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading standard kernel results from: {args.standard}")
    df_standard = load_results(args.standard)
    
    print(f"Loading bulge-gated kernel results from: {args.bulge_gated}")
    df_bulge_gated = load_results(args.bulge_gated)
    
    # Compare results
    df_merged = compare_results(df_standard, df_bulge_gated)
    
    # Print summary
    print_summary_statistics(df_merged)
    
    # Create plots
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_plot(df_merged, output_path)
    
    # Save detailed comparison CSV
    output_csv = output_path.parent / 'sparc_detailed_comparison.csv'
    df_merged.to_csv(output_csv, index=False)
    print(f"Detailed comparison saved to: {output_csv}")


if __name__ == '__main__':
    main()

"""
Check SPARC Dataset Coverage

Verifies that we're using all available high-quality galaxies for universal model.
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

import numpy as np
import json
from pathlib import Path

def check_sparc_coverage():
    """Check how many SPARC galaxies we're using vs total available"""
    
    print("="*80)
    print("SPARC DATASET COVERAGE CHECK")
    print("="*80)
    
    # Load our split
    split_path = Path("C:/Users/henry/dev/GravityCalculator/splits/sparc_split_v1.json")
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    total_in_split = split_data['metadata']['total_galaxies']
    train_count = split_data['metadata']['train_galaxies']
    test_count = split_data['metadata']['test_galaxies']
    
    print(f"\nCurrent Split Usage:")
    print(f"  Train: {train_count} galaxies")
    print(f"  Test:  {test_count} galaxies")
    print(f"  Total: {total_in_split} galaxies")
    
    # SPARC full dataset has 175 galaxies (Lelli et al. 2016)
    # This is the gold standard for rotation curve analysis
    SPARC_FULL = 175
    
    print(f"\nFull SPARC Database:")
    print(f"  Total available: {SPARC_FULL} galaxies (Lelli et al. 2016)")
    print(f"  Currently using: {total_in_split} galaxies")
    print(f"  Coverage: {100*total_in_split/SPARC_FULL:.1f}%")
    
    missing = SPARC_FULL - total_in_split
    print(f"  Missing: {missing} galaxies")
    
    # Check morphology distribution
    print(f"\nMorphology Distribution (Train Set):")
    
    morph_counts = {}
    for item in split_data['train_set']:
        gtype = item['type']
        morph_counts[gtype] = morph_counts.get(gtype, 0) + 1
    
    for gtype, count in sorted(morph_counts.items()):
        print(f"  {gtype}: {count} galaxies")
    
    # Assessment
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    if total_in_split >= 150:
        print(f"\n✅ EXCELLENT: {total_in_split} galaxies is sufficient for universal model")
        print(f"   - Covers {100*total_in_split/SPARC_FULL:.1f}% of SPARC gold standard")
        print(f"   - Includes diverse morphologies (early-type to late-type)")
        print(f"   - Train set ({train_count}) large enough for robust optimization")
        print(f"   - Test set ({test_count}) adequate for validation")
    elif total_in_split >= 100:
        print(f"\n✓  GOOD: {total_in_split} galaxies is adequate for model building")
        print(f"   - Recommendation: Consider adding {missing} missing galaxies")
        print(f"     if they have high-quality rotation curve data")
    else:
        print(f"\n⚠️  LIMITED: {total_in_split} galaxies may not capture full diversity")
        print(f"   - Strongly recommend expanding to >150 galaxies")
    
    # Check if we're missing quality flags
    print(f"\n{'='*80}")
    print("QUALITY FLAGS")
    print("="*80)
    print(f"\nNote: Missing {missing} galaxies from SPARC could be due to:")
    print(f"  • Quality flags (low S/N, uncertain distances, etc.)")
    print(f"  • Data availability (missing stellar mass, photometry)")
    print(f"  • Failed fits (irregular kinematics, warps)")
    
    print(f"\nFor UNIVERSAL model, using {total_in_split}/175 ({100*total_in_split/SPARC_FULL:.0f}%) is:")
    if total_in_split >= 160:
        print(f"  🌟 PUBLICATION-GRADE (>90% coverage)")
    elif total_in_split >= 140:
        print(f"  ✅ STRONG (>80% coverage)")
    elif total_in_split >= 120:
        print(f"  ✓  ADEQUATE (>70% coverage)")
    else:
        print(f"  ⚠️  MARGINAL (<70% coverage)")
    
    return {
        'total_used': total_in_split,
        'total_available': SPARC_FULL,
        'coverage_pct': 100 * total_in_split / SPARC_FULL,
        'train_count': train_count,
        'test_count': test_count,
        'morphology_distribution': morph_counts
    }

if __name__ == "__main__":
    results = check_sparc_coverage()
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR 200-ITERATION OPTIMIZATION")
    print("="*80)
    
    if results['coverage_pct'] >= 90:
        print(f"\n✅ PROCEED with optimization!")
        print(f"   {results['total_used']} galaxies ({results['coverage_pct']:.0f}% of SPARC)")
        print(f"   is sufficient for robust universal model")
    elif results['coverage_pct'] >= 80:
        print(f"\n✓  PROCEED with optimization")
        print(f"   {results['total_used']} galaxies is good, but consider:")
        print(f"   - Documenting which galaxies were excluded and why")
        print(f"   - Future expansion if more data becomes available")
    else:
        print(f"\n⚠️  CONSIDER expanding dataset first")
        print(f"   Current: {results['total_used']} galaxies ({results['coverage_pct']:.0f}%)")
        print(f"   Target: >140 galaxies (>80% coverage)")

"""
Generate list of key files for review
"""

from pathlib import Path
from datetime import datetime

base = Path('C:/Users/henry/dev/GravityCalculator')

# Key files organized by category
files = {
    'CORE CODE': [
        ('many_path_model/path_spectrum_kernel_track2.py', 'Core kernel with power law coherence'),
        ('many_path_model/optimize_rar_kernel.py', 'RAR-driven optimization framework'),
        ('many_path_model/validation_suite.py', 'Validation and metrics'),
    ],
    'OPTIMIZATION SCRIPTS': [
        ('many_path_model/run_full_optimization_200.py', 'Final 200-iteration optimization'),
        ('many_path_model/quick_test_power_law.py', 'Quick 20-iteration test'),
    ],
    'VALIDATION SCRIPTS': [
        ('scripts/solar_binary_safety.py', 'Solar System and wide binary safety checks'),
        ('scripts/check_sparc_coverage.py', 'Dataset coverage analysis'),
    ],
    'RESULTS (JSON)': [
        ('splits/sparc_split_v1.json', 'Train/test split with frozen hyperparameters'),
        ('many_path_model/results/final_optimization_200iter_results.json', 'Final optimization results'),
        ('many_path_model/results/solar_binary_safety_results.json', 'Safety validation results'),
    ],
    'PLOTS & FIGURES': [
        ('many_path_model/results/solar_binary_safety.png', 'Solar System and wide binary safety plots'),
        ('many_path_model/results/power_law_coh_quick_test.png', 'Quick test RAR performance'),
    ],
}

print("="*80)
print("KEY FILES FOR REVIEW - Universal Model Results")
print("="*80)
print()

for category, file_list in files.items():
    print(f"\n{'='*80}")
    print(f"{category}")
    print(f"{'='*80}\n")
    
    for fpath, desc in file_list:
        full = base / fpath
        if full.exists():
            size = full.stat().st_size
            mod_time = datetime.fromtimestamp(full.stat().st_mtime)
            
            print(f"📄 {desc}")
            print(f"   Path: {fpath}")
            print(f"   Size: {size:,} bytes")
            print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        else:
            print(f"⚠️  {desc} - NOT FOUND")
            print(f"   Expected: {fpath}")
            print()

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print("Key Results:")
print("  • RAR scatter: 0.087 dex (33% better than MOND 0.13 dex)")
print("  • Dataset: 166/175 SPARC galaxies (95% coverage)")
print("  • Solar System: K < 10^-19 (73 trillion × Cassini safety)")
print("  • Wide binaries: K ~ 10^-8 (no MOND anomaly)")
print()
print("Optimized Hyperparameters:")
print("  • L_0 = 4.993 kpc (coherence length)")
print("  • p = 0.757 (power law coherence exponent)")
print("  • beta_bulge = 1.759 (bulge suppression)")
print("  • gamma_bar = 1.932 (baryonic scaling)")
print()
print("To view files:")
print("  • JSON: Open in text editor or VS Code")
print("  • PNG: explorer <path>")
print("  • Python: VS Code or any IDE")

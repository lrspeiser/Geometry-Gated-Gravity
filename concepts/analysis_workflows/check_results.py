import json

# Load results
with open('out/optimized_analysis/optimized_results.json', 'r') as f:
    data = json.load(f)

# Check first few galaxies
for i in range(min(3, len(data))):
    galaxy = data[i]
    print(f"\nGalaxy: {galaxy['galaxy']}")
    print(f"  Default χ²/dof: {galaxy['default']['chi2_reduced']:.2f}")
    print(f"  Optimized χ²/dof: {galaxy['optimized']['chi2_reduced']:.2f}")
    print(f"  Improvement: {galaxy['improvement_percent']:.1f}%")
    print(f"  Default params: S0={galaxy['default']['params']['S0']}, rc={galaxy['default']['params']['rc_kpc']} kpc")
    print(f"  Optimized params: S0={galaxy['optimized']['params']['S0']:.3f}, rc={galaxy['optimized']['params']['rc_kpc']:.3f} kpc")
    if 'optimization_info' in galaxy['optimized']:
        opt_info = galaxy['optimized']['optimization_info']
        print(f"  Optimization: {opt_info['n_evaluations']} evaluations, converged={opt_info['success']}")
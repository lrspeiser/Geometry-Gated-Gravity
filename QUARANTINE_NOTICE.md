# ⚠️ QUARANTINE NOTICE - DO NOT USE

## Date: 2024-01-09
## Affected Files: solve_g3_working.py and related outputs

### CRITICAL ISSUES FOUND:

1. **UNPHYSICAL VELOCITIES**: Produces velocities ~10^9 km/s (should be 100-300 km/s)
2. **UNIT ERRORS**: Massive unit conversion problems throughout
3. **NON-CONVERGED**: All solver runs marked as not converged
4. **WRONG METRIC**: Uses incompatible χ² instead of paper's median fractional closeness
5. **PARAMETER MISMATCH**: S0≈0.10 vs paper's S0≈1.4×10^-4

### QUARANTINED FILES:
```
solve_g3_working.py              ❌ DO NOT USE - Unit errors
analyze_with_working_solver.py   ❌ DO NOT USE - Wrong metrics
out/working_analysis/*           ❌ DO NOT USE - Invalid results
fix_solver_5_ways.py             ❌ DO NOT USE - Flawed approach
test_working_solver_galaxy.py    ❌ DO NOT USE - Produces wrong output
```

### CORRECT APPROACH:

Use the validated paper pipeline:

**For SPARC galaxies:**
```bash
py -u root-m\pde\run_sparc_pde.py --axisym_maps --cv 5 \
   --rotmod_parquet data\sparc_rotmod_ltg.parquet \
   --NR 128 --NZ 128 --Rmax 80 --Zmax 80 \
   --S0 1.4e-4 --rc_kpc 22 --g0_kms2_per_kpc 1200 \
   --rc_gamma 0.5 --sigma_beta 0.10 --rc_ref_kpc 30 --sigma0_Msun_pc2 150
```

**For Clusters:**
Use PDE+HSE with total-baryon geometry parity as documented in paper.md

### LESSONS LEARNED:
- Always verify physical units (velocities should be 100-300 km/s for galaxies)
- Check convergence flags before claiming optimization success
- Use the paper's validated metrics (median fractional closeness)
- Don't mix different parameter normalizations

---
**This directory contains experimental code with fundamental errors.**
**DO NOT use for paper or production analysis.**
# Regression matrix for SigmaGravity replication

Scope
- End-to-end checks matching the paper’s reported pipelines and guardrails.

Checks
1. Data verifier
   - PASS if master_catalog.csv present (MD5 recorded) and SPARC MasterSheet noted.
2. SPARC validation suite
   - PASS if script exits 0; spot-check BTFR/RAR figures generated.
3. MACS0416 diagnostics
   - PASS if plots generated without exceptions.
4. Simple Einstein check
   - PASS if consistency check script exits 0.
5. Calibration (emcee)
   - PASS if posterior saved at output/mass_scaled_emcee_paper/flat_samples.npz and summary.txt written.
6. Holdout validation
   - PASS if summary.pass == true in output/holdout_validation_mass_scaled/holdout_results.json.
7. Triaxial validation
   - PASS if all three tests report ✓ PASSED.
8. Provenance aggregation
   - PASS if provenance/manifests_index.json exists and includes the posterior with valid catalog_md5.

Run
- Windows PowerShell: projects/SigmaGravity/scripts/run_replication.ps1
- Bash: projects/SigmaGravity/scripts/run_replication.sh

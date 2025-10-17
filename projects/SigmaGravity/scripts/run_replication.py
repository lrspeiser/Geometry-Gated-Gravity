#!/usr/bin/env python3
"""
SigmaGravity end-to-end replication harness.
- Reads config/paper_settings.yaml
- Runs: data verify, SPARC validation, MACS0416 diagnostics, simple Einstein check,
        calibration (emcee), holdout validation, triaxial validation, manifest aggregation.
- Prints PASS/FAIL summary from holdout validator.
"""
import sys, os, subprocess, json, time
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[3]  # repo root
SG = ROOT / 'projects' / 'SigmaGravity'
CFG = SG / 'config' / 'paper_settings.yaml'


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("\n>>> ", ' '.join(cmd))
    return subprocess.call(cmd, cwd=str(cwd or ROOT))


def main():
    if not CFG.exists():
        print(f"FATAL: Missing config: {CFG}")
        return 2
    if yaml is None:
        print("FATAL: PyYAML not installed. pip install pyyaml")
        return 2

    with open(CFG, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Config
    catalog_rel = cfg['clusters']['catalog_path']
    catalog = (ROOT / catalog_rel).resolve()
    tiers_train = ','.join(str(t) for t in cfg['clusters']['tiers_train'])
    exclude = ','.join(cfg['clusters']['exclude'])
    holdout = ','.join(cfg['clusters']['holdout'])
    pzs = cfg['source_redshift']['mode']

    out_cal = ROOT / 'output' / 'mass_scaled_emcee_paper'
    out_hold = ROOT / 'output' / 'holdout_validation_mass_scaled'

    steps = [
        ("verify_data", [sys.executable, str(SG / 'data' / 'verify_data.py')]),
        ("sparc_validation", [sys.executable, str(ROOT / 'many_path_model' / 'validation_suite.py'), '--all']),
        ("macs0416_diagnostics", [sys.executable, str(ROOT / 'scripts' / 'plot_macs0416_diagnostics.py')]),
        ("simple_einstein_check", [sys.executable, str(ROOT / 'scripts' / 'simple_einstein_check.py')]),
        ("calibration_emcee", [sys.executable, str(ROOT / 'scripts' / 'run_mass_scaled_emcee.py'),
                                '--catalog', str(catalog), '--tiers', tiers_train,
                                '--exclude', exclude, '--holdout', holdout, '--pzs', pzs,
                                '--outdir', str(out_cal)]),
        ("holdout_validation", [sys.executable, str(ROOT / 'scripts' / 'validate_holdout_mass_scaled.py'),
                                 '--posterior', str(out_cal / 'flat_samples.npz'), '--catalog', str(catalog),
                                 '--clusters', holdout, '--pzs', pzs, '--outdir', str(out_hold)]),
        ("validate_triaxial_lensing", [sys.executable, str(ROOT / 'scripts' / 'validate_triaxial_lensing.py')]),
        ("collect_manifests", [sys.executable, str(SG / 'provenance' / 'collect_manifests.py')])
    ]

    failures = []
    for name, cmd in steps:
        print("\n" + "=" * 70)
        print(f"STEP: {name}")
        print("=" * 70)
        rc = run(cmd)
        if rc != 0:
            print(f"FATAL: step '{name}' failed with exit code {rc}")
            failures.append(name)
            break

    # Summary
    print("\n" + "=" * 70)
    print("REPLICATION SUMMARY")
    print("=" * 70)
    if failures:
        print("FAILED at steps:", ', '.join(failures))
        return 1

    # Load holdout results
    results_path = out_hold / 'holdout_results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            res = json.load(f)
        print(json.dumps(res.get('summary', {}), indent=2))
        if not res.get('summary', {}).get('pass', False):
            print("Overall: FAIL (holdout summary)")
            return 1
    else:
        print("WARN: missing holdout_results.json; cannot summarize")

    print("Overall: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())

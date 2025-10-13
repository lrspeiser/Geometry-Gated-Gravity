#!/usr/bin/env python3
"""
run_full_roadmap.py - Master Orchestration Script for Publication Pipeline

Executes Phases A-E of the execution roadmap with automatic artifact collection,
success criteria checking, and decision points.

Usage:
    python run_full_roadmap.py --all              # Run all phases
    python run_full_roadmap.py --phase A          # V2.3b engineering fix
    python run_full_roadmap.py --phase B          # Track-2 kernel fitting
    python run_full_roadmap.py --phase C          # Hybrid training
    python run_full_roadmap.py --phase D          # Publication metrics
    python run_full_roadmap.py --phase E          # Gaia benchmark
    python run_full_roadmap.py --check-status     # Show current progress
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
from datetime import datetime

# Add project root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

@dataclass
class PhaseStatus:
    """Status tracking for each phase"""
    phase_id: str
    phase_name: str
    completed: bool = False
    passed: bool = False
    artifacts: List[str] = None
    metrics: Dict = None
    timestamp: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
        if self.metrics is None:
            self.metrics = {}

class RoadmapOrchestrator:
    """Orchestrate execution of all roadmap phases"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.status_file = output_dir / "roadmap_status.json"
        self.status = self._load_status()
        
    def _load_status(self) -> Dict[str, PhaseStatus]:
        """Load existing status or initialize"""
        if self.status_file.exists():
            with open(self.status_file) as f:
                data = json.load(f)
                return {k: PhaseStatus(**v) for k, v in data.items()}
        
        # Initialize all phases
        phases = {
            'A1': PhaseStatus('A1', 'V2.3b Parameter Verification'),
            'A2': PhaseStatus('A2', 'SB Smoke Test'),
            'A3': PhaseStatus('A3', 'Full SPARC V2.3b'),
            'B1': PhaseStatus('B1', 'Track-2 Kernel Training'),
            'B2': PhaseStatus('B2', 'Track-2 Hold-Out Validation'),
            'C1': PhaseStatus('C1', 'Track-2 Residual Analysis'),
            'C2': PhaseStatus('C2', 'Hybrid Model Training'),
            'D1': PhaseStatus('D1', 'RAR & BTFR Publication Metrics'),
            'D2': PhaseStatus('D2', 'Outlier Audit'),
            'D3': PhaseStatus('D3', 'Cross-Scale Cluster Check'),
            'E1': PhaseStatus('E1', 'Gaia Benchmark Parity'),
        }
        return phases
    
    def _save_status(self):
        """Save current status to disk"""
        data = {k: asdict(v) for k, v in self.status.items()}
        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_status(self):
        """Print current roadmap status"""
        print("=" * 80)
        print("EXECUTION ROADMAP STATUS")
        print("=" * 80)
        
        phases_by_letter = {}
        for phase_id, phase in self.status.items():
            letter = phase_id[0]
            if letter not in phases_by_letter:
                phases_by_letter[letter] = []
            phases_by_letter[letter].append((phase_id, phase))
        
        phase_names = {
            'A': 'Phase A: V2.3b Engineering Fix',
            'B': 'Phase B: Track-2 Physics Kernel',
            'C': 'Phase C: Track-2+3 Hybrid',
            'D': 'Phase D: Publication Metrics',
            'E': 'Phase E: Gaia Benchmark'
        }
        
        for letter in sorted(phases_by_letter.keys()):
            print(f"\n{phase_names[letter]}")
            print("-" * 80)
            for phase_id, phase in phases_by_letter[letter]:
                status_icon = "✅" if phase.completed and phase.passed else \
                              "❌" if phase.completed and not phase.passed else \
                              "⏳" if not phase.completed else "❓"
                print(f"  {status_icon} {phase_id}: {phase.phase_name}")
                if phase.completed:
                    print(f"       Completed: {phase.timestamp}")
                    if phase.notes:
                        print(f"       Note: {phase.notes}")
        
        print("\n" + "=" * 80)
    
    def run_phase_A1(self) -> Tuple[bool, str]:
        """Phase A.1: Verify V2.3b parameter loading"""
        print("\n" + "=" * 80)
        print("PHASE A.1: V2.3b Parameter Verification")
        print("=" * 80)
        
        # Check if test file exists
        test_file = REPO_ROOT / "many_path_model" / "bt_law" / "test_v2p2_bar_gating.py"
        
        if not test_file.exists():
            return False, f"Test file not found: {test_file}"
        
        # Run parameter verification
        try:
            result = subprocess.run(
                [sys.executable, str(test_file), "--verbose"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # Check for expected parameters in output
            sab_ok = "SAB" in output and "2.0" in output  # R_bar ~ 2.0 R_d
            sb_ok = "SB" in output and ("2.5" in output or "1.5" in output)  # γ_bar ~ 2.5
            
            if sab_ok and sb_ok:
                # Save output
                output_file = self.output_dir / "v2p3b_parameter_audit.txt"
                with open(output_file, 'w') as f:
                    f.write(output)
                
                self.status['A1'].artifacts.append(str(output_file))
                return True, "Parameters verified: SAB and SB differentiated correctly"
            else:
                return False, f"Parameter verification failed. Check output: {output[:500]}"
                
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 60 seconds"
        except Exception as e:
            return False, f"Error running test: {str(e)}"
    
    def run_phase_A2(self) -> Tuple[bool, str]:
        """Phase A.2: SB Smoke Test"""
        print("\n" + "=" * 80)
        print("PHASE A.2: SB Smoke Test (5 Galaxies)")
        print("=" * 80)
        
        # This would run the smoke test on 5 SB galaxies
        # For now, return placeholder
        return False, "Phase A.2: Implementation pending - requires SPARC data access"
    
    def run_phase_A3(self) -> Tuple[bool, str]:
        """Phase A.3: Full SPARC V2.3b run"""
        print("\n" + "=" * 80)
        print("PHASE A.3: Full SPARC V2.3b Validation")
        print("=" * 80)
        
        return False, "Phase A.3: Implementation pending - requires SPARC data access"
    
    def run_phase_B1(self) -> Tuple[bool, str]:
        """Phase B.1: Train Track-2 kernel"""
        print("\n" + "=" * 80)
        print("PHASE B.1: Track-2 Kernel Training")
        print("=" * 80)
        
        return False, "Phase B.1: Implementation pending - requires fit_path_spectrum_kernel.py"
    
    def run_phase_B2(self) -> Tuple[bool, str]:
        """Phase B.2: Validate Track-2 on hold-out"""
        print("\n" + "=" * 80)
        print("PHASE B.2: Track-2 Hold-Out Validation")
        print("=" * 80)
        
        return False, "Phase B.2: Implementation pending - requires Phase B.1 completion"
    
    def run_phase_C1(self) -> Tuple[bool, str]:
        """Phase C.1: Analyze Track-2 residuals"""
        print("\n" + "=" * 80)
        print("PHASE C.1: Track-2 Residual Analysis")
        print("=" * 80)
        
        return False, "Phase C.1: Implementation pending - requires Phase B.2 completion"
    
    def run_phase_C2(self) -> Tuple[bool, str]:
        """Phase C.2: Train hybrid model"""
        print("\n" + "=" * 80)
        print("PHASE C.2: Hybrid Model Training")
        print("=" * 80)
        
        return False, "Phase C.2: Implementation pending - requires Phase C.1 completion"
    
    def run_phase_D1(self) -> Tuple[bool, str]:
        """Phase D.1: RAR & BTFR publication metrics"""
        print("\n" + "=" * 80)
        print("PHASE D.1: RAR & BTFR Publication Metrics")
        print("=" * 80)
        
        return False, "Phase D.1: Implementation pending - requires best model from B/C"
    
    def run_phase_D2(self) -> Tuple[bool, str]:
        """Phase D.2: Outlier audit"""
        print("\n" + "=" * 80)
        print("PHASE D.2: Outlier Audit")
        print("=" * 80)
        
        # Check if outlier triage script exists
        script = REPO_ROOT / "many_path_model" / "outlier_triage_analysis.py"
        if script.exists():
            return False, "Phase D.2: Ready to run after Phase D.1 completion"
        else:
            return False, "Phase D.2: Implementation pending"
    
    def run_phase_D3(self) -> Tuple[bool, str]:
        """Phase D.3: Cross-scale cluster check"""
        print("\n" + "=" * 80)
        print("PHASE D.3: Cross-Scale Cluster Check")
        print("=" * 80)
        
        return False, "Phase D.3: Implementation pending - requires cluster data"
    
    def run_phase_E1(self) -> Tuple[bool, str]:
        """Phase E.1: Gaia benchmark parity"""
        print("\n" + "=" * 80)
        print("PHASE E.1: Gaia Benchmark Parity")
        print("=" * 80)
        
        return False, "Phase E.1: Implementation pending - requires Gaia data"
    
    def run_phase(self, phase_id: str) -> bool:
        """Run a specific phase"""
        phase_methods = {
            'A1': self.run_phase_A1,
            'A2': self.run_phase_A2,
            'A3': self.run_phase_A3,
            'B1': self.run_phase_B1,
            'B2': self.run_phase_B2,
            'C1': self.run_phase_C1,
            'C2': self.run_phase_C2,
            'D1': self.run_phase_D1,
            'D2': self.run_phase_D2,
            'D3': self.run_phase_D3,
            'E1': self.run_phase_E1,
        }
        
        if phase_id not in phase_methods:
            print(f"❌ Unknown phase: {phase_id}")
            return False
        
        # Run phase
        passed, notes = phase_methods[phase_id]()
        
        # Update status
        self.status[phase_id].completed = True
        self.status[phase_id].passed = passed
        self.status[phase_id].notes = notes
        self.status[phase_id].timestamp = datetime.now().isoformat()
        
        # Save status
        self._save_status()
        
        # Print result
        if passed:
            print(f"\n✅ Phase {phase_id} PASSED")
        else:
            print(f"\n❌ Phase {phase_id} FAILED")
        print(f"   {notes}")
        
        return passed
    
    def run_all_phases(self):
        """Run all phases in sequence"""
        phases_order = ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'D3', 'E1']
        
        print("=" * 80)
        print("RUNNING FULL EXECUTION ROADMAP")
        print("=" * 80)
        
        for phase_id in phases_order:
            if not self.run_phase(phase_id):
                print(f"\n⚠️  Phase {phase_id} failed or not ready. Stopping.")
                print("   Run --check-status to see current progress")
                break
        
        self.print_status()

def main():
    parser = argparse.ArgumentParser(description="Many-Path Gravity Execution Roadmap")
    parser.add_argument('--all', action='store_true', help='Run all phases')
    parser.add_argument('--phase', type=str, help='Run specific phase (A, B, C, D, E)')
    parser.add_argument('--check-status', action='store_true', help='Show current status')
    parser.add_argument('--output', type=str, default='results/roadmap_execution',
                       help='Output directory for artifacts')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    orchestrator = RoadmapOrchestrator(output_dir)
    
    if args.check_status:
        orchestrator.print_status()
    elif args.all:
        orchestrator.run_all_phases()
    elif args.phase:
        phase_letter = args.phase.upper()
        # Run all subphases of this phase
        subphases = [k for k in orchestrator.status.keys() if k.startswith(phase_letter)]
        if not subphases:
            print(f"❌ Unknown phase: {phase_letter}")
            return
        
        for subphase in subphases:
            if not orchestrator.run_phase(subphase):
                print(f"\n⚠️  Subphase {subphase} failed or not ready. Stopping.")
                break
        
        orchestrator.print_status()
    else:
        print("Usage: python run_full_roadmap.py [--all | --phase X | --check-status]")
        print("\nPhases:")
        print("  A: V2.3b Engineering Fix")
        print("  B: Track-2 Physics Kernel")
        print("  C: Track-2+3 Hybrid")
        print("  D: Publication Metrics")
        print("  E: Gaia Benchmark")

if __name__ == "__main__":
    main()

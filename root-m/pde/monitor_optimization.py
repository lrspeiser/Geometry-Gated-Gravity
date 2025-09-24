#!/usr/bin/env python3
"""
monitor_optimization.py

Monitor the progress of GPU optimization in real-time.
Shows system resources, progress, and results.
"""

import os
import json
import time
import psutil
from pathlib import Path
from datetime import datetime
import subprocess

try:
    import GPUtil
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
    print("Note: Install gputil for GPU monitoring: pip install gputil")


def get_gpu_stats():
    """Get GPU usage statistics."""
    if not GPU_MONITORING:
        return None
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100,
                'gpu_util': gpu.load * 100,
                'temp': gpu.temperature
            }
    except:
        pass
    return None


def check_process_running(process_name="python"):
    """Check if optimization process is running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                if any('gpu_multi_optimizer' in arg for arg in cmdline):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def monitor_results(output_dir: Path):
    """Monitor optimization results."""
    
    results_file = output_dir / "optimization_results.json"
    analysis_file = output_dir / "analysis_summary.json"
    
    results = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            pass
    
    summary = {}
    if analysis_file.exists():
        try:
            with open(analysis_file, 'r') as f:
                summary = json.load(f)
        except:
            pass
    
    return results, summary


def display_status(iteration: int = 0):
    """Display current status."""
    
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*60)
    print("GPU OPTIMIZATION MONITOR")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Refresh #{iteration}")
    print()
    
    # Check if process is running
    proc = check_process_running()
    if proc:
        print("🟢 OPTIMIZATION RUNNING")
        print(f"   PID: {proc.pid}")
        print(f"   CPU: {proc.cpu_percent():.1f}%")
        print(f"   Memory: {proc.memory_info().rss / 1e9:.2f} GB")
    else:
        print("🔴 OPTIMIZATION NOT RUNNING")
    
    print()
    
    # GPU Status
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print("GPU STATUS:")
        print(f"   Device: {gpu_stats['name']}")
        print(f"   Memory: {gpu_stats['memory_used']:.1f}/{gpu_stats['memory_total']:.1f} GB ({gpu_stats['memory_percent']:.1f}%)")
        print(f"   Utilization: {gpu_stats['gpu_util']:.1f}%")
        print(f"   Temperature: {gpu_stats['temp']}°C")
    else:
        print("GPU STATUS: Not available")
    
    print()
    
    # Results status
    output_dir = Path("optimization_results")
    results, summary = monitor_results(output_dir)
    
    print("OPTIMIZATION PROGRESS:")
    
    if results:
        # Count systems
        n_galaxies = len(results.get('galaxies', {}))
        n_clusters = len(results.get('clusters', {}))
        has_mw = 'mw' in results
        
        print(f"   Galaxies: {n_galaxies} completed")
        print(f"   Clusters: {n_clusters} completed")
        print(f"   Milky Way: {'✓' if has_mw else '✗'}")
        
        # Show convergence stats
        if summary:
            print("\nCONVERGENCE STATISTICS:")
            for sys_type, stats in summary.items():
                if isinstance(stats, dict) and 'n_converged' in stats:
                    print(f"   {sys_type}: {stats['n_converged']}/{stats['n_total']} converged")
                    if stats['n_converged'] > 0:
                        print(f"      γ={stats['gamma']['median']:.3f}, λ₀={stats['lambda0']['median']:.3f}, α={stats['alpha_grad']['median']:.3f}")
    else:
        print("   No results yet...")
    
    # Check for errors
    log_file = Path("optimization.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            errors = [l for l in lines if 'ERROR' in l or 'error' in l]
            if errors:
                print(f"\n⚠️ ERRORS FOUND ({len(errors)}):")
                for err in errors[-3:]:  # Show last 3 errors
                    print(f"   {err.strip()}")
    
    print("\n" + "-"*60)
    print("Press Ctrl+C to stop monitoring")


def check_data_availability():
    """Check if required data is available."""
    
    print("\nDATA CHECK:")
    print("-"*40)
    
    data_dir = Path("C:/Users/henry/dev/GravityCalculator/data")
    
    # Check SPARC
    sparc_dir = data_dir / "sparc"
    if sparc_dir.exists():
        catalog = sparc_dir / "MassModels_Lelli2016c.txt"
        if catalog.exists():
            print("✅ SPARC catalog found")
            rotmod_dir = sparc_dir / "Rotmod_LTG"
            if rotmod_dir.exists():
                rotmod_files = list(rotmod_dir.glob("*_rotmod.txt"))
                print(f"✅ SPARC rotation curves: {len(rotmod_files)} files")
            else:
                print("❌ SPARC Rotmod_LTG directory not found")
        else:
            print("❌ SPARC catalog not found")
    else:
        print("❌ SPARC directory not found")
    
    # Check Gaia
    gaia_dir = data_dir / "gaia_sky_slices"
    if gaia_dir.exists():
        parquet_files = list(gaia_dir.glob("processed_L*.parquet"))
        print(f"✅ Gaia slices: {len(parquet_files)} files")
    else:
        print("❌ Gaia directory not found")
    
    # Check Clusters
    clusters_dir = data_dir / "clusters"
    if clusters_dir.exists():
        clusters = ['ABELL_0426', 'ABELL_1689', 'A1795', 'A2029', 'A478']
        found = sum(1 for c in clusters if (clusters_dir / c).exists())
        print(f"✅ Clusters: {found}/{len(clusters)} found")
    else:
        print("❌ Clusters directory not found")
    
    print("-"*40)


def main():
    """Main monitoring loop."""
    
    print("Starting optimization monitor...")
    
    # Initial data check
    check_data_availability()
    time.sleep(2)
    
    # Check if we should start the optimizer
    proc = check_process_running()
    if not proc:
        print("\n❓ Optimizer not running. Start it? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            print("Starting optimizer...")
            subprocess.Popen(['python', 'gpu_multi_optimizer.py'], 
                           stdout=open('optimization.log', 'w'),
                           stderr=subprocess.STDOUT)
            time.sleep(2)
    
    # Monitor loop
    iteration = 0
    try:
        while True:
            iteration += 1
            display_status(iteration)
            time.sleep(5)  # Refresh every 5 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        
        # Final results summary
        output_dir = Path("optimization_results")
        results, summary = monitor_results(output_dir)
        
        if results:
            print("\nFINAL RESULTS SUMMARY:")
            print("="*40)
            
            # Count totals
            n_galaxies = len(results.get('galaxies', {}))
            n_clusters = len(results.get('clusters', {}))
            
            print(f"Total systems optimized:")
            print(f"  Galaxies: {n_galaxies}")
            print(f"  Clusters: {n_clusters}")
            print(f"  Milky Way: {'Yes' if 'mw' in results else 'No'}")
            
            if summary:
                print("\nParameter Summary:")
                for sys_type, stats in summary.items():
                    if isinstance(stats, dict) and 'gamma' in stats:
                        print(f"\n{sys_type.upper()}:")
                        print(f"  γ: {stats['gamma']['median']:.3f} ± {stats['gamma']['std']:.3f}")
                        print(f"  λ₀: {stats['lambda0']['median']:.3f} ± {stats['lambda0']['std']:.3f}")
                        print(f"  α: {stats['alpha_grad']['median']:.3f} ± {stats['alpha_grad']['std']:.3f}")


if __name__ == "__main__":
    main()
# How to Run the G³ Optimization Code

## Main Script Location
The primary optimization script is in the root directory:
- **`optimize_g3_all_datasets.py`** (21KB)

## Prerequisites

### Required Python Packages
```bash
pip install numpy pandas scipy matplotlib cupy-cuda12x
```

### GPU Support (Optional but Recommended)
- If you have an NVIDIA GPU: `cupy-cuda12x` provides 10-15× speedup
- Without GPU: The code automatically falls back to CPU (NumPy)

## Running the Optimization

### Basic Run
```bash
python optimize_g3_all_datasets.py
```
or
```bash
py -u optimize_g3_all_datasets.py
```

### What It Does
1. **Loads Data**:
   - SPARC rotation curves from `data/sparc_rotmod_ltg.parquet`
   - Gaia MW data from `data/gaia_sky_slices/processed_*.parquet`
   - Cluster profiles from `data/clusters/*/`

2. **Optimizes Parameters**:
   - Uses differential evolution algorithm
   - Finds best v0, rc, r0, delta values
   - GPU-accelerated if available

3. **Analyzes Performance**:
   - Computes accuracy metrics for each dataset
   - Generates summary statistics

4. **Saves Results** to `out/g3_optimization/`:
   - `best_parameters.json` - Optimized parameters
   - `galaxy_results.csv` - Per-galaxy performance
   - `cluster_results.csv` - Per-cluster performance
   - `summary_table.csv` - Summary statistics
   - `mw_results.csv` - Milky Way predictions

## Key Functions in the Code

### G³ Formula Implementation
```python
def logtail_acceleration(r_kpc, v0, rc, r0, delta, use_cupy=False):
    """
    LogTail acceleration component (G³ surrogate)
    
    Parameters:
    - r_kpc: radii in kpc
    - v0: asymptotic velocity (km/s)
    - rc: core radius (kpc)  
    - r0: gate activation radius (kpc)
    - delta: transition width (kpc)
    - use_cupy: whether to use GPU
    
    Returns:
    - g_tail: additional acceleration (km²/s²/kpc)
    """
```

### Main Optimization Function
```python
def optimize_parameters(galaxies, bounds, use_gpu=False):
    """
    Optimizes G³ parameters using differential evolution
    
    Parameters:
    - galaxies: dictionary of galaxy rotation curves
    - bounds: parameter search bounds [(v0_min,v0_max), ...]
    - use_gpu: whether to use GPU acceleration
    
    Returns:
    - best_params: array [v0, rc, r0, delta]
    """
```

## Modifying the Code

### Change Parameter Bounds
Edit line 430 in `optimize_g3_all_datasets.py`:
```python
bounds = [(100, 200), (5, 30), (1, 10), (1, 10)]
# Format: [(v0_min, v0_max), (rc_min, rc_max), (r0_min, r0_max), (delta_min, delta_max)]
```

### Adjust Optimization Settings
Edit lines 296-305:
```python
result = differential_evolution(
    lambda p: compute_galaxy_chi2(p, galaxies, use_gpu),
    bounds=bounds,
    seed=42,           # Random seed for reproducibility
    maxiter=100,       # Maximum iterations (increase for better convergence)
    popsize=15,        # Population size (increase for better exploration)
    workers=1 if use_gpu else -1,
    disp=True,         # Display progress
    polish=True        # Final local optimization
)
```

### Use Different Data
Modify the data loading functions (lines 103-232):
```python
def load_sparc_data():    # Lines 103-132
def load_gaia_mw_data():  # Lines 134-178  
def load_cluster_data():  # Lines 180-232
```

## Output Interpretation

### Best Parameters (from last run)
```json
{
  "v0_kms": 134.2,      // Asymptotic velocity scale
  "rc_kpc": 30.0,       // Core radius
  "r0_kpc": 2.6,        // Gate activation radius
  "delta_kpc": 1.5,     // Transition width
  "optimization_chi2": 9.497,
  "gpu_used": true
}
```

### Performance Metrics
- **SPARC**: 88% median accuracy on outer rotation curves
- **MW**: Needs better mass model (currently overpredicts)
- **Clusters**: Temperature predictions need refinement

## Troubleshooting

### GPU Not Detected
If you see `[!!] GPU not available, using CPU`:
1. Check CUDA installation: `nvidia-smi`
2. Verify CuPy: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`
3. Reinstall CuPy for your CUDA version

### Memory Issues
If you get memory errors with GPU:
- Reduce number of galaxies processed at once
- Use smaller batch sizes
- Fall back to CPU mode

### Data Not Found
If data files are missing:
1. Ensure you're in the correct directory
2. Check that data files exist in `data/` subdirectories
3. Run data preparation scripts if needed

## Related Scripts

Other G³/LogTail implementations in the repository:
- `rigor/scripts/cluster_logtail_test.py` - Cluster-specific testing
- `rigor/scripts/add_models_and_tests.py` - Add G³ to prediction tables
- `rigor/scripts/driver_analysis.py` - Full analysis pipeline

## Contact

For questions about the code or G³ theory, see:
- Main repository: https://github.com/lrspeiser/Geometry-Gated-Gravity
- Paper: `paper.md` in the root directory
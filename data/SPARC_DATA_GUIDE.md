# SPARC Data Guide

## Summary

**ALL THE DATA YOU NEED IS ALREADY IN THE SPARC FILES!** The six required fields for `SPARCGalaxy` are all available in the existing SPARC dataset:

1. **hubble_type** - Available in `MasterSheet_SPARC.mrt`
2. **hubble_name** - Derived from hubble_type (0=S0, 1=Sa, ..., 11=BCD)
3. **distance_mpc** - Available in `MasterSheet_SPARC.mrt` AND in each `_rotmod.dat` file header
4. **sb_disk** - Column 7 in each `_rotmod.dat` file (already being loaded!)
5. **sb_bulge** - Column 8 in each `_rotmod.dat` file (already being loaded!)
6. **avg_bulge_frac** - Can be computed from v_bulge, v_disk, v_gas (already being done!)

## File Locations

### Master Table
- **Path**: `data/Rotmod_LTG/MasterSheet_SPARC.mrt`
- **Contains**: Morphological types, distances, and global properties for all 175 galaxies
- **Format**: Fixed-width text table

### Rotation Curve Files
- **Path**: `data/Rotmod_LTG/*_rotmod.dat`
- **Contains**: Rotation curves with 8 columns:
  1. Radius (kpc)
  2. Observed velocity (km/s)
  3. Velocity error (km/s)
  4. Gas velocity (km/s)
  5. Disk velocity (km/s)
  6. Bulge velocity (km/s)
  7. **Disk surface brightness (L_sun/pc²)**
  8. **Bulge surface brightness (L_sun/pc²)**

## Current Implementation

The code in `sparc_stratified_test.py` already correctly loads all this data:

```python
def load_sparc_galaxy(filepath: Path, master_info: Optional[Dict] = None) -> SPARCGalaxy:
    # Gets distance from file header OR master table
    # Parses 8 columns from rotation curve file
    data = np.array(data)  # 8 columns
    
    # Extract all the data:
    return SPARCGalaxy(
        name=name,
        hubble_type=hubble_type,              # From master table
        hubble_name=hubble_name,              # From master table
        type_group=type_group,                # Derived from hubble_name
        distance_mpc=distance_mpc,            # From file or master
        r_kpc=data[:, 0],
        v_obs=data[:, 1],
        v_err=data[:, 2],
        v_gas=data[:, 3],
        v_disk=data[:, 4],
        v_bulge=data[:, 5],
        sb_disk=data[:, 6],                   # ← Column 7!
        sb_bulge=data[:, 7],                  # ← Column 8!
        bulge_frac=bulge_frac,                # Computed
        avg_bulge_frac=avg_bulge_frac         # Computed
    )
```

## Why Multiprocessing Failed

The intensive optimization script (`sparc_intensive_optimize.py`) attempted to create `SPARCGalaxy` instances in worker processes by passing only a subset of data from a JSON file. The JSON file contained:

```json
{
  "global_params": {...},
  "galaxies": [
    {
      "name": "CamB",
      "r_kpc": [...],
      "v_obs": [...],
      // Missing: hubble_type, hubble_name, distance_mpc, sb_disk, sb_bulge, avg_bulge_frac
    }
  ]
}
```

When the worker process tried to instantiate `SPARCGalaxy(**galaxy_data)`, Python raised:
```
TypeError: __init__() missing 6 required positional arguments: 
'hubble_type', 'hubble_name', 'distance_mpc', 'sb_disk', 'sb_bulge', 'avg_bulge_frac'
```

## Solution

**Option 1: Sequential Processing (Current)**
The existing `sparc_hierarchical_search_v2.py` script works perfectly because it:
1. Loads the master table once
2. For each galaxy, loads the full rotation curve file
3. Creates complete `SPARCGalaxy` objects with all required fields
4. Optimizes sequentially (still fast with GPU)

**Option 2: Fix Multiprocessing (Future)**
To enable multiprocessing, you would need to:
1. Modify the JSON output to include all 6 missing fields
2. Update the intensive optimization script to pass complete data to workers
3. Or, refactor to load complete galaxy data within each worker process

## Recommendation

**Stick with the sequential approach** (`sparc_hierarchical_search_v2.py` with `--mode per_galaxy`). 

Why?
- ✅ It works perfectly right now
- ✅ Uses GPU acceleration for the heavy computation
- ✅ The bottleneck is the optimization algorithm (CMA-ES), not data loading
- ✅ Multiprocessing wouldn't help much since the GPU is already parallelizing the physics calculations
- ✅ Avoids complex serialization issues with numpy arrays

The optimization completed all 175 galaxies successfully with this approach!

## Data Source

SPARC database from Lelli et al. 2016:
- **Paper**: "SPARC. I. Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves"
- **ADS**: https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L
- **Data**: http://astroweb.cwru.edu/SPARC/

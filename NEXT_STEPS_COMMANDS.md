# 🎯 SPARC Analysis: Next Steps - Execution Commands

## Current Status

✅ **Mega parallel optimization COMPLETE!**
- 175 galaxies optimized
- Median APE: **7.55%**
- Mean APE: 12.45%
- 100% convergence
- Class-wise parameters extracted

---

## 1️⃣ PRIORITY 1: Zero-Shot Universality Testing

### Test with Class-Wise Parameters (3 parameter sets)
```powershell
python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode class --output_dir results/zero_shot_class
```

**Expected Results:**
- Late types: ~6-8% median APE
- Early types: ~10-12% median APE
- Intermediate: ~15-18% median APE
- Overall: ~8-12% median APE

### Test with Single Global Parameter Set (strictest test)
```powershell
python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode global --output_dir results/zero_shot_global
```

**Expected Results:**
- Overall: ~15-20% median APE (proves class-conditioning helps)

### Run Both Tests Together
```powershell
python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode both --output_dir results/zero_shot_comparison
```

---

## 2️⃣ PRIORITY 2: Clustering Analysis

### Run All Clustering Methods
```powershell
python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering --method all --n_clusters 5
```

### Just K-Means (Fast)
```powershell
python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering_kmeans --method kmeans --n_clusters 5
```

### Auto-detect Clusters with DBSCAN
```powershell
python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering_dbscan --method dbscan --dbscan_eps 0.8 --dbscan_min_samples 3
```

---

## 3️⃣ What Each Analysis Reveals

### Zero-Shot Testing Shows:
- **Universality**: Can 3 parameter sets replace 175 tailored fits?
- **Class importance**: How much does morphology-based conditioning help?
- **Robustness**: Is performance stable without per-galaxy tuning?

### Clustering Analysis Shows:
- **Parameter patterns**: Which galaxies need similar parameters?
- **Outlier identification**: Which galaxies behave differently?
- **Physical groupings**: Do clusters align with morphology?
- **Sensitivity analysis**: Which parameters matter most for each group?

---

## 📊 Expected Outcomes

### Success Criteria for Zero-Shot:
✅ Class-wise median APE < 12%
✅ Performance within 2× of optimized results
✅ No systematic failures by type

### Success Criteria for Clustering:
✅ 3-7 clear clusters identified
✅ Clusters correlate with morphology AND/OR mass
✅ Low intra-cluster variance in parameters
✅ Outliers have physical explanations (bars, interactions, etc.)

---

## 🔬 After These Complete

### Immediate Follow-ups:

1. **B/T Continuous Law** - Fit smooth function from bulge fraction to parameters
   ```powershell
   # TODO: Create bulge_fraction_law.py
   ```

2. **Outlier Analysis** - Deep dive on worst 15 galaxies
   - Check for bars, warps, interactions
   - Look for systematic issues
   - Identify needed model improvements

3. **Population Laws** - BTFR and RAR from universal parameters
   - Baryonic Tully-Fisher Relation
   - Radial Acceleration Relation
   - No per-galaxy tuning allowed!

4. **Mass Consistency** - Validate surface → mass → potential chain
   - Check energy conservation
   - Verify force balance
   - Solar system safety bounds

---

## 📁 Output Files You'll Get

### From Zero-Shot:
- `zero_shot_class_results.csv` - Per-galaxy results with class params
- `zero_shot_global_results.csv` - Per-galaxy results with global params  
- `zero_shot_class_complete.json` - Full results + statistics
- `comparison.json` - Class vs global comparison

### From Clustering:
- `features.csv` - Engineered feature matrix (~20 features × 175 galaxies)
- `clustered_kmeans.csv` - Galaxy assignments to clusters
- `cluster_summary_kmeans.csv` - Statistics for each cluster
- `pca_scatter_kmeans.png` - Visualization of clusters in 2D
- `cluster_characteristics_kmeans.png` - Parameter distributions
- `pca_variance.png` - PCA component importance

---

## 🚀 Quick Start

**Run everything in sequence (will take ~45-60 minutes total):**

```powershell
python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode both --output_dir results/zero_shot && python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering --method all --n_clusters 5
```

---

## 📈 Key Metrics to Watch

### Zero-Shot Performance:
- **Median APE by type** - Should be < 15% for all groups
- **Success rate** - Should be 100% (no failures)
- **Worst cases** - Identify which galaxies fail universality

### Clustering Quality:
- **Silhouette score** - Should be > 0.3 (good separation)
- **Cluster sizes** - Should have 15-40 galaxies each (balanced)
- **Morphology alignment** - Clusters should correlate with early/late types

---

## 🎓 Analysis Interpretation

### If Zero-Shot Works Well (< 12% median):
✅ **Model is universal** - Few knobs explain all galaxies
✅ **Class-conditioning works** - Morphology matters
✅ **Ready for prediction** - Can apply to new galaxies

### If Zero-Shot Is Mediocre (12-20% median):
⚠️ Need continuous B/T law (step 2)
⚠️ May need mass-dependent scaling
⚠️ Check for systematic biases by type

### If Clustering Finds 3-5 Clusters:
✅ **Physical groupings exist** - Beyond simple morphology
✅ **Parameter families** - Can further reduce knobs
✅ **Predictive framework** - Assign new galaxies to clusters

### If Clustering Finds Many Clusters:
⚠️ May indicate overfitting in optimization
⚠️ Need stronger regularization
⚠️ Consider fixing more parameters globally

---

## 📚 Files Ready for Analysis

Already in `results/mega_parallel/`:
- ✅ `mega_parallel_results.json` - Full optimization results
- ✅ `mega_parallel_summary.csv` - Quick CSV summary
- ✅ `class_params_for_zero_shot.json` - Class-wise median parameters
- ✅ `outliers_top15.csv` - Worst performers to investigate
- ✅ `mega_parallel_ranked.csv` - All galaxies ranked by APE

---

## 🎯 Today's Goals

1. **Run zero-shot test** (~30 minutes) - Tests universality
2. **Run clustering** (~15 minutes) - Reveals parameter structure  
3. **Review results** - Identify next physics to add

Tomorrow: B/T laws, BTFR/RAR, outlier deep-dive
